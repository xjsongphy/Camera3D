#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FPS="30"
FORCE=0
DRY_RUN=0
SKIP_YOLO=0
VIDEOS=("S2-1" "S2-2")
SOURCES=("default" "motion" "yolo")

usage() {
  cat <<'EOF' >&2
Usage: bash ./scripts/task3_full_pipeline.sh [--force] [--dry-run] [--skip-yolo]
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
    --skip-yolo)
      SKIP_YOLO=1
      shift
      ;;
    *)
      usage
      ;;
  esac
done

if [[ $SKIP_YOLO -eq 1 ]]; then
  SOURCES=("default" "motion")
fi

echo "Running full Task3 pipeline for S2-1 and S2-2."
echo "FPS: ${FPS}"
echo "Mask sources: ${SOURCES[*]}"
if [[ $FORCE -eq 1 ]]; then
  echo "Force: true"
fi
if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run: true"
fi
if [[ $SKIP_YOLO -eq 1 ]]; then
  echo "Skip YOLO: true"
else
  echo "Ensuring YOLO dependency (uv extra: task3-yolo)..."
  uv sync --extra task3-yolo
fi

for source in "${SOURCES[@]}"; do
  mask_args=(lab1 task3-mask --source "$source" --videos "${VIDEOS[@]}" --fps "$FPS")
  if [[ $FORCE -eq 1 ]]; then
    mask_args+=(--force)
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    mask_args+=(--dry-run)
  fi

  echo
  echo "[1/2] Generating masks: ${source}"
  uv run "${mask_args[@]}"
done

task3_args=(lab1 task3 --videos "${VIDEOS[@]}" --fps "$FPS" --stage all --methods raw)
if [[ $FORCE -eq 1 ]]; then
  task3_args+=(--force)
fi
if [[ $DRY_RUN -eq 1 ]]; then
  task3_args+=(--dry-run)
fi

echo
echo "[2/2] Running reconstruction: raw"
uv run "${task3_args[@]}"

for source in "${SOURCES[@]}"; do
  task3_args=(lab1 task3 --videos "${VIDEOS[@]}" --fps "$FPS" --stage all --methods mask --mask-source "$source")
  if [[ $FORCE -eq 1 ]]; then
    task3_args+=(--force)
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    task3_args+=(--dry-run)
  fi

  echo
  echo "[2/2] Running reconstruction: mask + ${source}"
  uv run "${task3_args[@]}"
done

echo
echo "Task3 outputs:"
for video in "${VIDEOS[@]}"; do
  echo "- outputs/lab1/task3/${video}_fps${FPS}"
done
