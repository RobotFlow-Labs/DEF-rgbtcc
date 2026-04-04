"""Push trained DEF-rgbtcc model to HuggingFace Hub.

Usage:
    python scripts/push_to_hf.py --checkpoint /path/to/best.pth --repo ilessio-aiflowlab/DEF-rgbtcc
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Best checkpoint path")
    parser.add_argument("--exports-dir", default="/mnt/artifacts-datai/exports/DEF-rgbtcc")
    parser.add_argument("--repo", default="ilessio-aiflowlab/DEF-rgbtcc")
    args = parser.parse_args()

    api = HfApi()
    exports = Path(args.exports_dir)

    # Create repo if it doesn't exist
    try:
        api.create_repo(args.repo, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload all export files
    files = list(exports.glob("*"))
    print(f"Uploading {len(files)} files to {args.repo}")

    for f in files:
        if f.is_file():
            print(f"  Uploading {f.name} ({f.stat().st_size / 1e6:.1f}MB)")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=args.repo,
            )

    # Upload config files
    configs_to_upload = [
        "configs/paper.toml",
        "anima_module.yaml",
    ]
    for cfg in configs_to_upload:
        p = Path(cfg)
        if p.exists():
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=p.name,
                repo_id=args.repo,
            )

    # Create model card
    model_card = """---
tags:
  - crowd-counting
  - rgb-thermal
  - defense
  - anima
datasets:
  - RGBT-CC
library_name: pytorch
---

# DEF-rgbtcc: RGB-T Crowd Counting

Dual-Modulation Framework for RGB-T Crowd Counting via Spatially Modulated Attention and Adaptive Fusion.

Paper: [ArXiv 2509.17079](https://arxiv.org/abs/2509.17079)

## Architecture
- Backbone: Shared VGG-19
- Encoder: Spatially Modulated Attention (SMA) Transformer
- Fusion: Adaptive Cross-Modal Fusion (ACMF)
- Output: Density map regression

## Available Formats
- `model.pth` — PyTorch state dict
- `model.safetensors` — SafeTensors format
- `model.onnx` — ONNX (opset 17)
- `model_fp16.trt` — TensorRT FP16
- `model_fp32.trt` — TensorRT FP32

## Usage
```python
from def_rgbtcc.serve import RGBTCCInference

model = RGBTCCInference("model.pth")
result = model.predict(rgb_image, thermal_image)
print(f"Count: {result['count']:.1f}")
```

## ANIMA Module
Part of the ANIMA Defense Module ecosystem (Wave 8).
Products: ORACLE, ATLAS, NEMESIS
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(model_card)
        f.flush()
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo="README.md",
            repo_id=args.repo,
        )

    print(f"\nDone! Model at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
