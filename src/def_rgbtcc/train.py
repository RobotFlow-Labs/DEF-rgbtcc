"""DEF-rgbtcc training entry point.

Usage:
    python -m def_rgbtcc.train --config configs/paper.toml
    python -m def_rgbtcc.train --config configs/debug.toml --max-steps 5
"""
import argparse
import os
import random

import numpy as np
import toml
import torch

from def_rgbtcc.training.trainer import RGBTCCTrainer


def set_seed(seed: int = 3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="DEF-rgbtcc training")
    parser.add_argument("--config", required=True, help="TOML config file path")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps (for smoke test)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    cfg = toml.load(args.config)
    cfg["config_path"] = args.config

    if args.resume:
        cfg["resume"] = args.resume

    set_seed(cfg["training"].get("seed", 3407))

    trainer = RGBTCCTrainer(cfg)
    trainer.setup()
    trainer.train(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
