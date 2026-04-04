from __future__ import annotations

import argparse

import torch

from def_rgbtcc.reference_wrapper import build_reference_model


def run(args: argparse.Namespace) -> None:
    model = build_reference_model(checkpoint=args.checkpoint, device=args.device)
    rgb = torch.randn(args.batch_size, 3, args.height, args.width, device=args.device)
    thermal = torch.randn(args.batch_size, 3, args.height, args.width, device=args.device)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = model([rgb, thermal])

    if args.device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"peak_allocated_mb={peak:.2f}")
        print(f"peak_reserved_mb={reserved:.2f}")
    else:
        print("peak_allocated_mb=NA")
        print("peak_reserved_mb=NA")


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory benchmark for DEF-rgbtcc reference model")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
