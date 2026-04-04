"""ANIMA-compliant trainer for DEF-rgbtcc.

Standards: config-driven, step-based checkpointing, early stopping,
NaN detection, warmup+cosine LR, crash protection logging.
"""
import json
import logging
import math
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from def_rgbtcc.datasets.crowd import Crowd, train_collate
from def_rgbtcc.losses.bay_loss import Bay_Loss
from def_rgbtcc.losses.post_prob import Post_Prob
from def_rgbtcc.models.dm import Net
from def_rgbtcc.training.evaluation import eval_game, eval_relative

logger = logging.getLogger("def_rgbtcc")


class CheckpointManager:
    def __init__(self, save_dir: Path, keep_top_k: int = 2, metric: str = "game0", mode: str = "min"):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(self, state: dict, metric_value: float, step: int) -> Path:
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)
        best_val, best_path = self.history[0]
        best_dest = self.save_dir / "best.pth"
        shutil.copy2(best_path, best_dest)
        return path


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state):
        self.current_step = state["current_step"]


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        improved = (
            (metric < self.best - self.min_delta)
            if self.mode == "min"
            else (metric > self.best + self.min_delta)
        )
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            logger.info(f"[EARLY STOP] No improvement for {self.patience} epochs.")
            return True
        return False


class RGBTCCTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Logging setup
        log_dir = Path(cfg["checkpoint"]["save_dir"]) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        # Metrics log (JSONL)
        self.metrics_path = log_dir / "metrics.jsonl"

        self._print_config()

    def _print_config(self):
        cfg = self.cfg
        logger.info(f"[CONFIG] {cfg.get('config_path', 'inline')}")
        logger.info(f"[BATCH] batch_size={cfg['training']['batch_size']}")
        logger.info(f"[GPU] {torch.cuda.get_device_name(0)}, "
                     f"{torch.cuda.get_device_properties(0).total_mem // 1024**2}MB")
        logger.info(f"[TRAIN] {cfg['training']['epochs']} epochs, "
                     f"lr={cfg['training']['learning_rate']}, "
                     f"optimizer=Adam")

    def setup(self):
        cfg = self.cfg
        tcfg = cfg["training"]
        dcfg = cfg["data"]

        # Datasets
        self.datasets = {
            split: Crowd(
                os.path.join(dcfg["train_path"], split),
                tcfg["crop_size"],
                tcfg["downsample_ratio"],
                split,
            )
            for split in ("train", "val", "test")
        }
        logger.info(f"[DATA] train={len(self.datasets['train'])} "
                     f"val={len(self.datasets['val'])} "
                     f"test={len(self.datasets['test'])}")

        self.train_loader = DataLoader(
            self.datasets["train"],
            collate_fn=train_collate,
            batch_size=tcfg["batch_size"],
            shuffle=True,
            num_workers=dcfg.get("num_workers", 4),
            pin_memory=dcfg.get("pin_memory", True),
        )
        self.val_loader = DataLoader(
            self.datasets["val"], batch_size=1, shuffle=False, num_workers=4, pin_memory=False
        )
        self.test_loader = DataLoader(
            self.datasets["test"], batch_size=1, shuffle=False, num_workers=4, pin_memory=False
        )

        # Model
        self.model = Net()
        self.model.to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[MODEL] {n_params / 1e6:.1f}M parameters")

        # Load pretrained
        pretrained = dcfg.get("pretrained_checkpoint")
        if pretrained and os.path.exists(pretrained):
            state = torch.load(pretrained, map_location="cpu", weights_only=True)
            # Remap VGG features keys if needed (checkpoint has "0.weight" vs model "features.0.weight")
            model_keys = set(self.model.state_dict().keys())
            if any(k not in model_keys for k in state.keys()):
                remapped = {}
                for k, v in state.items():
                    new_key = f"features.{k}"
                    if new_key in model_keys:
                        remapped[new_key] = v
                if remapped:
                    state = remapped
                    logger.info(f"[PRETRAINED] remapped {len(remapped)} keys to features.*")
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            logger.info(f"[PRETRAINED] loaded {pretrained} "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})")

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=tcfg["learning_rate"],
            weight_decay=tcfg["weight_decay"],
        )

        # Scheduler
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * tcfg["epochs"]
        warmup_steps = int(total_steps * tcfg.get("warmup_ratio", 0.05))
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_steps, total_steps, min_lr=tcfg.get("min_lr", 1e-7)
        )

        # Loss
        self.post_prob = Post_Prob(
            tcfg["sigma"],
            tcfg["crop_size"],
            tcfg["downsample_ratio"],
            tcfg["background_ratio"],
            tcfg.get("use_background", True),
            self.device,
        )
        self.criterion = Bay_Loss(tcfg.get("use_background", True), self.device)

        # Checkpoint manager
        ckpt_cfg = cfg["checkpoint"]
        self.ckpt_manager = CheckpointManager(
            save_dir=Path(ckpt_cfg["save_dir"]),
            keep_top_k=ckpt_cfg.get("keep_top_k", 2),
            metric=ckpt_cfg.get("metric", "game0"),
            mode=ckpt_cfg.get("mode", "min"),
        )

        # Early stopping
        es_cfg = cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 20),
            min_delta=es_cfg.get("min_delta", 0.001),
            mode="min",
        )

        # State
        self.start_epoch = 0
        self.global_step = 0
        self.best_game0 = float("inf")
        self.best_game3 = float("inf")

        # Resume
        resume = cfg.get("resume")
        if resume and os.path.exists(resume):
            self._resume(resume)

    def _resume(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_game0 = ckpt.get("best_game0", float("inf"))
        self.best_game3 = ckpt.get("best_game3", float("inf"))
        logger.info(f"[RESUME] from epoch {self.start_epoch}, step {self.global_step}")

    def _get_state(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_game0": self.best_game0,
            "best_game3": self.best_game3,
            "config": self.cfg,
        }

    def _log_metrics(self, metrics: dict):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def train(self, max_steps: int | None = None):
        tcfg = self.cfg["training"]
        ckpt_interval = self.cfg["checkpoint"].get("save_every_n_steps", 500)

        for epoch in range(self.start_epoch, tcfg["epochs"]):
            logger.info(f"--- Epoch {epoch}/{tcfg['epochs'] - 1} ---")
            train_loss, train_mae = self._train_epoch(epoch, max_steps, ckpt_interval)

            if max_steps is not None and self.global_step >= max_steps:
                logger.info(f"[MAX STEPS] Reached {max_steps} steps, stopping.")
                self.ckpt_manager.save(self._get_state(epoch), train_loss, self.global_step)
                return

            # Validate
            game0, game3, mse_val = self._val_epoch()

            self._log_metrics({
                "epoch": epoch,
                "step": self.global_step,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_game0": game0,
                "val_game3": game3,
                "val_mse": mse_val,
                "lr": self.optimizer.param_groups[0]["lr"],
            })

            # Checkpoint on val metric
            improved = game0 < self.best_game0 or game3 < self.best_game3
            if improved:
                self.best_game0 = min(game0, self.best_game0)
                self.best_game3 = min(game3, self.best_game3)
                logger.info(f"*** Best GAME0={self.best_game0:.3f} GAME3={self.best_game3:.3f} epoch={epoch}")
                self.ckpt_manager.save(self._get_state(epoch), game0, self.global_step)
                self._test_epoch()

            # Early stopping
            if self.early_stopping.step(game0):
                logger.info(f"[EARLY STOP] Stopping at epoch {epoch}")
                break

        logger.info("[DONE] Training complete.")
        logger.info(f"[BEST] GAME0={self.best_game0:.3f} GAME3={self.best_game3:.3f}")

    def _train_epoch(self, epoch: int, max_steps: int | None, ckpt_interval: int):
        self.model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        n_samples = 0
        t0 = time.time()

        dataloader = tqdm(self.train_loader, desc=f"Train E{epoch}", leave=False, dynamic_ncols=True)
        for rgb, t, points, targets, st_sizes in dataloader:
            rgb = rgb.to(self.device)
            t = t.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [tgt.to(self.device) for tgt in targets]

            outputs = self.model([rgb, t])
            prob_list = self.post_prob(points, st_sizes)
            loss = self.criterion(prob_list, targets, outputs)

            # NaN detection
            if torch.isnan(loss):
                logger.error("[FATAL] Loss is NaN — stopping training")
                logger.error("[FIX] Reduce lr by 10x, check data for corrupt samples")
                raise RuntimeError("NaN loss detected")

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            max_grad = self.cfg["training"].get("max_grad_norm", 1.0)
            if max_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad)

            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            N = rgb.size(0)
            pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
            res = pre_count - gd_count
            epoch_loss += loss.item() * N
            epoch_mae += np.sum(np.abs(res))
            n_samples += N

            dataloader.set_postfix(loss=f"{loss.item():.3f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

            # Step-based checkpointing
            if self.global_step % ckpt_interval == 0:
                self.ckpt_manager.save(
                    self._get_state(epoch), epoch_loss / max(n_samples, 1), self.global_step
                )

            if max_steps is not None and self.global_step >= max_steps:
                break

        dataloader.close()
        avg_loss = epoch_loss / max(n_samples, 1)
        avg_mae = epoch_mae / max(n_samples, 1)
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch} Train: loss={avg_loss:.4f} MAE={avg_mae:.2f} "
            f"lr={self.optimizer.param_groups[0]['lr']:.2e} time={elapsed:.1f}s"
        )
        return avg_loss, avg_mae

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        game = [0.0, 0.0, 0.0, 0.0]
        mse = [0.0, 0.0, 0.0, 0.0]
        total_re = 0.0

        for rgb, t, target, count, name in tqdm(self.val_loader, desc="Val", leave=False):
            rgb = rgb.to(self.device)
            t = t.to(self.device)
            outputs = self.model([rgb, t])
            for L in range(4):
                abs_err, sq_err = eval_game(outputs, target, L)
                game[L] += abs_err
                mse[L] += sq_err
            total_re += eval_relative(outputs, target)

        N = len(self.val_loader)
        game = [g / N for g in game]
        mse_vals = [np.sqrt(m / N) for m in mse]
        total_re /= N

        logger.info(
            f"Val: GAME0={game[0]:.2f} GAME1={game[1]:.2f} GAME2={game[2]:.2f} "
            f"GAME3={game[3]:.2f} MSE={mse_vals[0]:.2f} RE={total_re:.4f}"
        )
        return game[0], game[3], mse_vals[0]

    @torch.no_grad()
    def _test_epoch(self):
        self.model.eval()
        game = [0.0, 0.0, 0.0, 0.0]
        mse = [0.0, 0.0, 0.0, 0.0]
        total_re = 0.0

        for rgb, t, target, count, name in tqdm(self.test_loader, desc="Test", leave=False):
            rgb = rgb.to(self.device)
            t = t.to(self.device)
            outputs = self.model([rgb, t])
            for L in range(4):
                abs_err, sq_err = eval_game(outputs, target, L)
                game[L] += abs_err
                mse[L] += sq_err
            total_re += eval_relative(outputs, target)

        N = len(self.test_loader)
        game = [g / N for g in game]
        mse_vals = [np.sqrt(m / N) for m in mse]
        total_re /= N

        logger.info(
            f"Test: GAME0={game[0]:.2f} GAME1={game[1]:.2f} GAME2={game[2]:.2f} "
            f"GAME3={game[3]:.2f} MSE={mse_vals[0]:.2f} RE={total_re:.4f}"
        )
