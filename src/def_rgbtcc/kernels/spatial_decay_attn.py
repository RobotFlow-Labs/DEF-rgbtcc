from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch

from def_rgbtcc.reference_wrapper import _load_reference_module


@dataclass(slots=True)
class SpatialDecayAttn:
    extension_name: str = "spatial_decay_attn_cuda"

    def __post_init__(self) -> None:
        try:
            self._ext = importlib.import_module(self.extension_name)
        except Exception:
            self._ext = None

    def __call__(self, h: int, w: int, beta_scale: torch.Tensor, beta_bias: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self._ext is not None and hasattr(self._ext, "forward"):
            return self._ext.forward(h, w, beta_scale, beta_bias)

        ref = _load_reference_module()
        dist = ref.generate_spatial_distance(h, w, device=device).unsqueeze(0)
        return ref.process_spatial_decay(dist, beta_scale, beta_bias)
