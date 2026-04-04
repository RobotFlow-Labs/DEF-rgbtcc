from __future__ import annotations

import argparse
from pathlib import Path

from def_rgbtcc.validation import validate_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset/checkpoint contract for DEF-rgbtcc")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sample-limit", type=int, default=5)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    results = validate_assets(Path(args.dataset_root), checkpoint_path=ckpt, sample_limit=args.sample_limit)

    failed = 0
    for item in results:
        status = "OK" if item.ok else "FAIL"
        print(f"[{status}] {item.message}")
        if not item.ok:
            failed += 1

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
