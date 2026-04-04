from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import torch


def resolve_reference_root(reference_root: Optional[Path] = None) -> Path:
    if reference_root is not None:
        return reference_root
    return Path(__file__).resolve().parents[2] / "repositories" / "RGBT-Crowd-Counting"


def _load_reference_module(reference_root: Optional[Path] = None):
    root = resolve_reference_root(reference_root)
    module_path = root / "models" / "dm.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Reference model file not found: {module_path}")

    spec = importlib.util.spec_from_file_location("rgbtcc_ref_dm", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_reference_model(
    checkpoint: Optional[Path] = None,
    device: str = "cpu",
    strict: bool = True,
    reference_root: Optional[Path] = None,
) -> torch.nn.Module:
    module = _load_reference_module(reference_root)
    model = module.Net()

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=strict)

    model.to(device)
    model.eval()
    return model
