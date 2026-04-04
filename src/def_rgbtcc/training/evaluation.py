"""GAME (Grid Average Mean Error) evaluation metrics for crowd counting."""
import cv2
import numpy as np
import torch


def eval_game(output: torch.Tensor, target, L: int = 0):
    output_np = output[0][0].cpu().numpy()
    target_np = target[0] if isinstance(target, (list, tuple)) else target
    if isinstance(target_np, torch.Tensor):
        target_np = target_np.numpy()
    H, W = target_np.shape
    ratio = H / output_np.shape[0]
    output_np = cv2.resize(output_np, (W, H), interpolation=cv2.INTER_CUBIC) / (
        ratio * ratio
    )
    assert output_np.shape == target_np.shape

    p = pow(2, L)
    abs_error = 0.0
    square_error = 0.0
    for i in range(p):
        for j in range(p):
            out_block = output_np[i * H // p : (i + 1) * H // p, j * W // p : (j + 1) * W // p]
            tgt_block = target_np[i * H // p : (i + 1) * H // p, j * W // p : (j + 1) * W // p]
            abs_error += abs(out_block.sum() - tgt_block.sum())
            square_error += (out_block.sum() - tgt_block.sum()) ** 2
    return abs_error, square_error


def eval_relative(output: torch.Tensor, target) -> float:
    output_num = output.cpu().data.sum().item()
    if isinstance(target, torch.Tensor):
        target_num = target.sum().float().item()
    else:
        target_num = float(np.sum(target))
    if target_num == 0:
        return 0.0
    return abs(output_num - target_num) / target_num
