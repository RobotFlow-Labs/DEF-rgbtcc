from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from def_rgbtcc.reference_wrapper import build_reference_model


def run(args: argparse.Namespace) -> None:
    device = args.device
    model = build_reference_model(checkpoint=args.checkpoint, device=device)

    rgb = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    thermal = torch.randn(args.batch_size, 3, args.height, args.width, device=device)

    with torch.no_grad():
        for _ in range(args.warmup_iters):
            _ = model([rgb, thermal])

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        samples_ms = []
        for _ in range(args.measure_iters):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model([rgb, thermal])
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = float(np.mean(samples_ms))
    std_ms = float(np.std(samples_ms))
    fps = (1000.0 / mean_ms) * args.batch_size

    print(f"latency_ms_mean={mean_ms:.4f}")
    print(f"latency_ms_std={std_ms:.4f}")
    print(f"throughput_fps={fps:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency benchmark for DEF-rgbtcc reference model")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--measure-iters", type=int, default=200)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
