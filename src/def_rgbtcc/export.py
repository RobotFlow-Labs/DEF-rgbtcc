"""DEF-rgbtcc model export pipeline.
Exports: pth -> safetensors -> ONNX -> TRT FP16 -> TRT FP32

Usage:
    python -m def_rgbtcc.export --checkpoint best.pth --output-dir exports/
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import torch

from def_rgbtcc.models.dm import Net


def export_pth(model: torch.nn.Module, output_dir: Path):
    path = output_dir / "model.pth"
    torch.save(model.state_dict(), path)
    print(f"[EXPORT] pth: {path} ({path.stat().st_size / 1e6:.1f}MB)")
    return path


def export_safetensors(model: torch.nn.Module, output_dir: Path):
    from safetensors.torch import save_file
    path = output_dir / "model.safetensors"
    save_file(model.state_dict(), str(path))
    print(f"[EXPORT] safetensors: {path} ({path.stat().st_size / 1e6:.1f}MB)")
    return path


def export_onnx(model: torch.nn.Module, output_dir: Path, device: str = "cuda"):
    path = output_dir / "model.onnx"
    model.eval()
    rgb = torch.randn(1, 3, 224, 224, device=device)
    t = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        ([rgb, t],),
        str(path),
        input_names=["rgb", "thermal"],
        output_names=["density"],
        dynamic_axes={
            "rgb": {0: "batch", 2: "height", 3: "width"},
            "thermal": {0: "batch", 2: "height", 3: "width"},
            "density": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
    )
    print(f"[EXPORT] ONNX: {path} ({path.stat().st_size / 1e6:.1f}MB)")
    return path


def export_trt(onnx_path: Path, output_dir: Path, precision: str = "fp16"):
    trt_path = output_dir / f"model_{precision}.trt"
    toolkit = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")

    if toolkit.exists():
        cmd = [
            "python", str(toolkit),
            "--onnx", str(onnx_path),
            "--output", str(trt_path),
            "--precision", precision,
        ]
        print(f"[TRT] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[EXPORT] TRT {precision}: {trt_path}")
            return trt_path
        else:
            print(f"[TRT] Error: {result.stderr[:500]}")
    else:
        # Fallback: use trtexec directly
        trtexec = shutil.which("trtexec")
        if trtexec:
            cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={trt_path}"]
            if precision == "fp16":
                cmd.append("--fp16")
            print(f"[TRT] Running trtexec: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(f"[EXPORT] TRT {precision}: {trt_path}")
                return trt_path
            else:
                print(f"[TRT] Error: {result.stderr[:500]}")
        else:
            print(f"[TRT] Neither toolkit nor trtexec found. Skipping TRT {precision}.")
    return None


def main():
    parser = argparse.ArgumentParser(description="DEF-rgbtcc export pipeline")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="/mnt/artifacts-datai/exports/DEF-rgbtcc")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = Net()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()
    print(f"[LOAD] Model from {args.checkpoint}")

    # Export pipeline
    export_pth(model, output_dir)
    export_safetensors(model, output_dir)
    onnx_path = export_onnx(model, output_dir, device=args.device)
    export_trt(onnx_path, output_dir, precision="fp16")
    export_trt(onnx_path, output_dir, precision="fp32")

    print(f"\n[DONE] All exports in {output_dir}")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
