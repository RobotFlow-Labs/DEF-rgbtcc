from __future__ import annotations

import argparse
import time

import torch

from def_rgbtcc.reference_wrapper import build_reference_model


def run(args: argparse.Namespace) -> None:
    model = build_reference_model(checkpoint=args.checkpoint, device=args.device)
    rgb = torch.randn(args.batch_size, 3, args.height, args.width, device=args.device)
    thermal = torch.randn(args.batch_size, 3, args.height, args.width, device=args.device)

    start = time.perf_counter()
    iters = 0
    with torch.no_grad():
        while True:
            _ = model([rgb, thermal])
            iters += 1
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            if (time.perf_counter() - start) >= args.seconds:
                break

    elapsed = time.perf_counter() - start
    imgs = iters * args.batch_size
    print(f"elapsed_s={elapsed:.4f}")
    print(f"iterations={iters}")
    print(f"images={imgs}")
    print(f"images_per_second={imgs / elapsed:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Throughput benchmark for DEF-rgbtcc reference model")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--seconds", type=int, default=15)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
