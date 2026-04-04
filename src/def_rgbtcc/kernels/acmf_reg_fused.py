from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ACMFRegFused:
    extension_name: str = "acmf_reg_fused_cuda"

    def __post_init__(self) -> None:
        try:
            self._ext = importlib.import_module(self.extension_name)
        except Exception:
            self._ext = None

    def __call__(
        self,
        rgb_feat: torch.Tensor,
        thermal_feat: torch.Tensor,
        fusion_layer: torch.nn.Module,
        reg_layer: torch.nn.Module,
    ) -> torch.Tensor:
        if self._ext is not None and hasattr(self._ext, "forward"):
            return self._ext.forward(rgb_feat, thermal_feat)

        fused = fusion_layer(rgb_feat, thermal_feat)
        density = reg_layer(fused)
        return torch.abs(density)
