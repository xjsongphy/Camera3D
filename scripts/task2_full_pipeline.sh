#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE_FPS="30"
FORCE_FLAG="${1:-}"
if [[ -n "$FORCE_FLAG" && "$FORCE_FLAG" != "--force" ]]; then
  echo "Usage: bash ./scripts/task2_full_pipeline.sh [--force]" >&2
  exit 1
fi

format_param_tag() {
  local value="$1"
  local formatted
  formatted="$(awk "BEGIN {printf \"%.3f\", $value}" | sed 's/\.0*$//; s/\./p/g')"
  echo "fps${formatted}"
}

task1_result_ready() {
  local case_root="$1"
  local required=(
    "${case_root}/images"
    "${case_root}/frame_map.csv"
    "${case_root}/sparse/0/images.txt"
    "${case_root}/sparse/0/cameras.txt"
    "${case_root}/sparse/0/points3D.txt"
    "${case_root}/trajectory.png"
  )
  for p in "${required[@]}"; do
    [[ -e "$p" ]] || return 1
  done
  compgen -G "${case_root}/images/*.jpg" > /dev/null || return 1
  return 0
}

EXTRA_ARGS=()
if [[ "$FORCE_FLAG" == "--force" ]]; then
  EXTRA_ARGS+=("--force")
fi

echo "Running Task2 full pipeline for S1-2."
echo "Source FPS: ${SOURCE_FPS}"
echo "Force: ${FORCE_FLAG:-false}"
echo ""
PARAM_TAG="$(format_param_tag "$SOURCE_FPS")"
TASK1_CASE_ROOT="outputs/lab1/task1/S1-2_${PARAM_TAG}"

echo "=== Step 1/2: Build full-sequence task1 result for S1-2 ==="
if [[ "$FORCE_FLAG" == "--force" ]] || ! task1_result_ready "$TASK1_CASE_ROOT"; then
  uv run lab1 task1 --videos S1-2 --fps "$SOURCE_FPS" --stage all "${EXTRA_ARGS[@]}"
else
  echo "Reuse existing task1 result: ${TASK1_CASE_ROOT}"
fi

echo
echo "=== Step 2/2: Run task2 with optimized default subsequences ==="
echo "Using default subsequences (return_mid, scan_stable, return_long)"
echo "Already completed subsequences will be automatically skipped."
uv run lab1 task2 --source-fps "$SOURCE_FPS" --stage all "${EXTRA_ARGS[@]}"

TASK2_ROOT="outputs/lab1/task2/S1-2_${PARAM_TAG}"
SUMMARY_CSV="${TASK2_ROOT}/summary.csv"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "Missing summary.csv: $SUMMARY_CSV" >&2
  exit 1
fi

echo
echo "=== Task2 ATE Summary ==="
awk -F',' 'NR==1 {next} {printf "%s | subset=%s, common=%s, ATE=%.6f, scale=%.6f\n", $1, $2, $3, $4, $5}' "$SUMMARY_CSV"

echo
echo "Task2 outputs:"
echo "- Root: ${TASK2_ROOT}"
echo "- Summary CSV: ${SUMMARY_CSV}"
echo "- Per-subsequence files: method_a/, method_b/, trajectory_overlay.png, metrics.txt, timing.csv"
