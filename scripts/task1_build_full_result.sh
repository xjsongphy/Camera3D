#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VIDEO="${1:-S1-2}"
FPS="${2:-30}"
FORCE_FLAG="${3:-}"

ARGS=(run lab1 task1 --videos "$VIDEO" --fps "$FPS" --stage all)
if [[ "$FORCE_FLAG" == "--force" ]]; then
  ARGS+=(--force)
fi

echo "Build task1 full result: video=${VIDEO}, fps=${FPS}, force=${FORCE_FLAG:-false}"
uv "${ARGS[@]}"
