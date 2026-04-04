#!/bin/bash
# Post-training: re-export best checkpoint and push to HF
# Run this after training completes:
#   bash scripts/post_training.sh

set -e

CKPT="/mnt/artifacts-datai/checkpoints/DEF-rgbtcc/best.pth"
EXPORT_DIR="/mnt/artifacts-datai/exports/DEF-rgbtcc"
HF_REPO="ilessio-aiflowlab/DEF-rgbtcc"

echo "=== Post-Training Export Pipeline ==="
echo "Checkpoint: $CKPT"
echo "Export dir: $EXPORT_DIR"

# Check training is done
if pgrep -f "def_rgbtcc.train" > /dev/null; then
    echo "[WARN] Training still running! Wait for it to finish."
    exit 1
fi

# Export all formats
echo "--- Exporting pth + safetensors + ONNX ---"
PYTHONPATH="" CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m def_rgbtcc.export \
    --checkpoint "$CKPT" \
    --output-dir "$EXPORT_DIR" \
    --device cuda

# TRT export
echo "--- Exporting TRT FP16 + FP32 ---"
cd "$EXPORT_DIR"
PYTHONPATH="" CUDA_VISIBLE_DEVICES=1 /mnt/forge-data/modules/04_wave8/DEF-rgbtcc/.venv/bin/python \
    /mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py \
    --onnx model.onnx --output-dir .
cd /mnt/forge-data/modules/04_wave8/DEF-rgbtcc

# Push to HF
echo "--- Pushing to HuggingFace ---"
PYTHONPATH="" .venv/bin/python scripts/push_to_hf.py \
    --checkpoint "$CKPT" \
    --exports-dir "$EXPORT_DIR" \
    --repo "$HF_REPO"

echo "=== Done! ==="
echo "Model at: https://huggingface.co/$HF_REPO"
