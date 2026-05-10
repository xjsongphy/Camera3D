#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$#" -eq 0 ]]; then
  VIDEOS=("S1-2")
else
  VIDEOS=("$@")
fi

FPS_LIST=(4 8 16 30)
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_DIR="outputs/lab1/task1/benchmarks"
SUMMARY_CSV="${SUMMARY_DIR}/task1_full_sweep_${STAMP}.csv"

mkdir -p "$SUMMARY_DIR"
echo "video,fps,extract_s,feature_extractor_s,sequential_matcher_s,hierarchical_mapper_s,model_converter_s,sfm_total_s,registered_frames,trajectory_png,timing_csv" > "$SUMMARY_CSV"

csv_value() {
  local csv_path="$1"
  local stage_name="$2"
  awk -F',' -v stage="$stage_name" 'NR>1 && $1==stage {print $2; found=1} END{if(!found) print ""}' "$csv_path"
}

registered_frames_count() {
  local images_txt="$1"
  awk '
    BEGIN {c=0}
    /^[[:space:]]*$/ {next}
    /^#/ {next}
    {
      if ($1 ~ /^[0-9]+$/) { c++ }
      getline
    }
    END {print c}
  ' "$images_txt"
}

echo "Running task1 full pipeline sweep sequentially (no parallelism)."
echo "Videos: ${VIDEOS[*]}"
echo "FPS: ${FPS_LIST[*]}"
echo "Summary: ${SUMMARY_CSV}"

for video in "${VIDEOS[@]}"; do
  for fps in "${FPS_LIST[@]}"; do
    echo
    echo "=== task1 / video=${video} / fps=${fps} / stage=all ==="
    uv run lab1 task1 --videos "$video" --fps "$fps" --stage all --force
    uv run lab1 task1 cloud --videos "$video" --fps "$fps" --force

    param_tag="fps${fps}"
    case_root="outputs/lab1/task1/${video}_${param_tag}"
    timing_csv="${case_root}/timing.csv"
    traj_png="${case_root}/trajectory.png"
    cloud_png="${case_root}/sparse_points.png"
    images_txt="${case_root}/sparse/0/images.txt"

    if [[ ! -f "$timing_csv" ]]; then
      echo "Missing timing file: ${timing_csv}" >&2
      exit 1
    fi
    if [[ ! -f "$images_txt" ]]; then
      echo "Missing sparse poses file: ${images_txt}" >&2
      exit 1
    fi
    if [[ ! -f "$cloud_png" ]]; then
      echo "Missing sparse point cloud plot: ${cloud_png}" >&2
      exit 1
    fi

    extract_s="$(csv_value "$timing_csv" "extract")"
    feat_s="$(csv_value "$timing_csv" "feature_extractor")"
    match_s="$(csv_value "$timing_csv" "sequential_matcher")"
    hmapper_s="$(csv_value "$timing_csv" "hierarchical_mapper")"
    conv_s="$(csv_value "$timing_csv" "model_converter")"
    sfm_s="$(csv_value "$timing_csv" "sfm_total")"
    reg_frames="$(registered_frames_count "$images_txt")"

    if [[ -f "$traj_png" ]]; then
      traj_ok="yes"
    else
      traj_ok="no"
    fi

    echo "${video},${fps},${extract_s},${feat_s},${match_s},${hmapper_s},${conv_s},${sfm_s},${reg_frames},${traj_ok},${timing_csv}" >> "$SUMMARY_CSV"
  done
done

echo
echo "Sweep finished."
echo "Summary CSV: ${SUMMARY_CSV}"
