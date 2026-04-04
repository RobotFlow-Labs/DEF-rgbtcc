"""DEF-rgbtcc scaffolding package."""

from .config import BenchConfig, DataConfig, RuntimeConfig, TrainConfig
from .reference_wrapper import build_reference_model, resolve_reference_root

__all__ = [
    "BenchConfig",
    "DataConfig",
    "RuntimeConfig",
    "TrainConfig",
    "build_reference_model",
    "resolve_reference_root",
]
