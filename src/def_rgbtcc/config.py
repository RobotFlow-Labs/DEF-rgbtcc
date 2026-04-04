from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    dataset_root: Path = Path("/mnt/forge-data/datasets/RGB-T-CC/RGBT-CC")
    drone_dataset_root: Path = Path("/mnt/forge-data/datasets/RGB-T-CC/DroneRGBT")
    pretrained_checkpoint: Path = Path("pretrained/vgg_vit_depth_2_head_6.pth")


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 1
    max_epoch: int = 400
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    crop_size: int = 224
    downsample_ratio: int = 8
    sigma: float = 8.0
    background_ratio: float = 0.15


@dataclass(slots=True)
class BenchConfig:
    batch_size: int = 1
    height: int = 224
    width: int = 224
    warmup_iters: int = 50
    measure_iters: int = 200


@dataclass(slots=True)
class RuntimeConfig:
    reference_root: Path = Path("repositories/RGBT-Crowd-Counting")
    device: str = "cuda"
