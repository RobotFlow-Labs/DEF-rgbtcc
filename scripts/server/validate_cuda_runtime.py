from __future__ import annotations

import argparse

import torch

from def_rgbtcc.reference_wrapper import build_reference_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CUDA runtime for DEF-rgbtcc")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()

    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    dev = torch.device("cuda")
    print(f"gpu_name={torch.cuda.get_device_name(dev)}")

    model = build_reference_model(checkpoint=args.checkpoint, device="cuda")
    rgb = torch.randn(1, 3, args.height, args.width, device="cuda")
    thermal = torch.randn(1, 3, args.height, args.width, device="cuda")

    with torch.no_grad():
        out = model([rgb, thermal])

    print(f"smoke_output_shape={tuple(out.shape)}")
    print(f"smoke_output_sum={float(out.sum().item()):.6f}")
