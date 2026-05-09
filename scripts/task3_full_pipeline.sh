#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FPS="5"
FORCE=0
DRY_RUN=0
MASK_SOURCE="default"
METHODS=("raw")

usage() {
  cat <<'EOF' >&2
Usage: bash ./scripts/task3_full_pipeline.sh [--force] [--dry-run] [--mask-source default|motion|yolo] [--methods raw mask]
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --mask-source)
      [[ $# -ge 2 ]] || usage
      MASK_SOURCE="$2"
      shift 2
      ;;
    --methods)
      shift
      METHODS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        METHODS+=("$1")
        shift
      done
      [[ ${#METHODS[@]} -gt 0 ]] || usage
      ;;
    *)
      usage
      ;;
  esac
done

EXTRA_ARGS=(--videos S2-1 S2-2 --fps "$FPS" --stage all --methods "${METHODS[@]}")
EXTRA_ARGS+=(--mask-source "$MASK_SOURCE")
if [[ $FORCE -eq 1 ]]; then
  EXTRA_ARGS+=(--force)
fi
if [[ $DRY_RUN -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi
echo "Running Task3 for S2-1 and S2-2."
echo "FPS: ${FPS}"
echo "Methods: ${METHODS[*]}"
echo "Mask source: ${MASK_SOURCE}"
if [[ $FORCE -eq 1 ]]; then
  echo "Force: true"
fi
if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run: true"
fi
uv run lab1 task3 "${EXTRA_ARGS[@]}"

echo
echo "Task3 outputs:"
for video in S2-1 S2-2; do
  echo "- outputs/lab1/task3/${video}_fps${FPS}"
done
