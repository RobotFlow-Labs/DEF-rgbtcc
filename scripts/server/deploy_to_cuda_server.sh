#!/usr/bin/env bash
set -euo pipefail

SERVER_ALIAS="${1:-datai_srv7_development}"
REMOTE_DIR="${2:-/mnt/forge-data/modules/wave-8/DEF-rgbtcc}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Deploying ${LOCAL_DIR} -> ${SERVER_ALIAS}:${REMOTE_DIR}"
ssh "$SERVER_ALIAS" "mkdir -p '$REMOTE_DIR'"

rsync -av --delete \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  "$LOCAL_DIR/" "$SERVER_ALIAS:$REMOTE_DIR/"

echo "Deployment complete"

echo "Next:"
echo "  ssh $SERVER_ALIAS 'bash $REMOTE_DIR/scripts/server/bootstrap_cuda_server.sh $REMOTE_DIR'"
