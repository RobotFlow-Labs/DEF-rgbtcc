from __future__ import annotations

from typing import Tuple


def mlx_available() -> bool:
    try:
        import mlx.core as _mx  # noqa: F401
        return True
    except Exception:
        return False


def spatial_decay_matrix(h: int, w: int, beta_scale: float, beta_bias: float):
    """MLX scaffold for spatial decay matrix generation.

    Returns an MLX tensor when MLX is available.
    """
    import mlx.core as mx

    ys = mx.arange(h)
    xs = mx.arange(w)
    grid_y, grid_x = mx.meshgrid(ys, xs)
    coords = mx.stack([grid_y, grid_x], axis=-1).reshape(-1, 2).astype(mx.float32)

    delta = coords[:, None, :] - coords[None, :, :]
    dist = mx.sqrt(mx.sum(delta * delta, axis=-1))

    beta_s = 1.0 / (1.0 + mx.exp(-beta_scale))
    beta_b = mx.log(1.0 + mx.exp(beta_bias))
    shifted = dist - beta_b
    processed = mx.where(shifted >= 0, shifted, shifted * 0.1)
    return mx.power(beta_s, processed)
