#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <module_dir_on_server> [checkpoint_path_or_empty]"
  exit 1
fi

MODULE_DIR="$1"
CHECKPOINT="${2:-}"

cd "$MODULE_DIR"
source .venv/bin/activate

CKPT_ARGS=()
if [[ -n "$CHECKPOINT" ]]; then
  CKPT_ARGS=(--checkpoint "$CHECKPOINT")
fi

python -m def_rgbtcc.benchmarking.latency --device cuda --height 224 --width 224 "${CKPT_ARGS[@]}"
python -m def_rgbtcc.benchmarking.throughput --device cuda --seconds 30 "${CKPT_ARGS[@]}"
python -m def_rgbtcc.benchmarking.memory --device cuda "${CKPT_ARGS[@]}"
