#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <module_dir_on_server>"
  exit 1
fi

MODULE_DIR="$1"
cd "$MODULE_DIR"

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install --upgrade pip
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .

python -c "import def_rgbtcc; print('import_ok')"
