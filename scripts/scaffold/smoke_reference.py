from __future__ import annotations

import argparse

import torch

from def_rgbtcc.reference_wrapper import build_reference_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference model smoke runner")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()

    model = build_reference_model(checkpoint=args.checkpoint, device=args.device)
    rgb = torch.randn(1, 3, args.height, args.width, device=args.device)
    thermal = torch.randn(1, 3, args.height, args.width, device=args.device)

    with torch.no_grad():
        out = model([rgb, thermal])

    print(f"output_shape={tuple(out.shape)}")
    print(f"output_sum={float(out.sum().item()):.6f}")


if __name__ == "__main__":
    main()
