"""CUDA-accelerated spatial distance decay for SMA transformer."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

# Try to load CUDA kernel
_cuda_ext = None
_kernel_path = Path(__file__).resolve().parents[3] / "kernels" / "cuda"
if _kernel_path.exists() and str(_kernel_path) not in sys.path:
    sys.path.insert(0, str(_kernel_path))
try:
    import rgbtcc_cuda_kernels
    _cuda_ext = rgbtcc_cuda_kernels
except ImportError:
    pass


def fused_spatial_distance_decay(
    h: int, w: int, beta_scale: torch.Tensor, beta_bias: torch.Tensor
) -> torch.Tensor:
    """Compute spatial distance matrix with learnable decay.

    If CUDA kernel available, uses fused GPU implementation.
    Otherwise falls back to pure PyTorch.

    Returns: (nhead, H*W, H*W) decay matrix
    """
    if _cuda_ext is not None and beta_scale.is_cuda:
        return _cuda_ext.fused_spatial_distance_decay(h, w, beta_scale, beta_bias)

    # PyTorch fallback
    device = beta_scale.device
    with torch.no_grad():
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        coords = coords.view(-1, 2).float()
        dist_matrix = torch.cdist(coords, coords, p=2).unsqueeze(0)

    beta_scale_sigmoid = torch.sigmoid(beta_scale)
    beta_bias_softplus = F.softplus(beta_bias)
    dist_matrix = dist_matrix.unsqueeze(1)
    processed_dist = F.leaky_relu(
        dist_matrix - beta_bias_softplus, negative_slope=0.1
    )
    return torch.pow(beta_scale_sigmoid, processed_dist).squeeze(0)


def fused_density_blend(
    rgb_feat: torch.Tensor, thermal_feat: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Fused RGB-T feature blending (ACMF).

    Args:
        rgb_feat: (B, C, H, W)
        thermal_feat: (B, C, H, W)
        weight: (B, 1, 1, 1) per-sample fusion weight

    Returns: (B, C, H, W) blended features
    """
    if _cuda_ext is not None and rgb_feat.is_cuda:
        return _cuda_ext.fused_density_blend(rgb_feat, thermal_feat, weight)

    return rgb_feat * weight + thermal_feat * (1 - weight)


def is_cuda_available() -> bool:
    return _cuda_ext is not None
