"""DEF-rgbtcc: Dual-Modulation Framework for RGB-T Crowd Counting."""

from .config import BenchConfig, DataConfig, RuntimeConfig, TrainConfig
from .models.dm import Net
from .reference_wrapper import build_reference_model, resolve_reference_root

__all__ = [
    "BenchConfig",
    "DataConfig",
    "Net",
    "RuntimeConfig",
    "TrainConfig",
    "build_reference_model",
    "resolve_reference_root",
]
