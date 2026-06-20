#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

# Sweep launcher for Linux.
# Keep the main experiment settings in CONFIG_PATH; this script only points at
# a concrete input directory and overrides the fps/scene tag per run.
CONFIG_PATH="${CONFIG_PATH:-configs/lab3/extra.json}"
INPUT_DIR="${INPUT_DIR:-input/lab3_dormitory_input}"
SCENE_NAME="${SCENE_NAME:-dormitory}"
read -r -a FPS_LIST <<< "${FPS_VALUES:-4 8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
METHODS=()
DEFAULT_METHODS=(sfm nerf neus 3dgs)
IMAGE_LIMIT="${IMAGE_LIMIT:-}"
BLUR_THRESHOLD="${BLUR_THRESHOLD:-10.0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${METHODS_VALUES:-}" ]]; then
    read -r -a METHODS <<< "${METHODS_VALUES}"
fi

if [[ "${FORCE}" != "0" && "${FORCE}" != "1" ]]; then
    echo "FORCE must be 0 or 1" >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1" >&2
    exit 2
fi

RUN_SUMMARY=()

format_fps_tag() {
    local fps="$1"
    echo "${fps//./p}"
}

build_lab3_args() {
    local fps="$1"
    local dry_run_mode="$2"

    local scene_tag="${SCENE_NAME}_fps$(format_fps_tag "${fps}")"
    local args=(
        "--config" "${CONFIG_PATH}"
        "--input-dir" "${INPUT_DIR}"
        "--scene-name" "${scene_tag}"
    )

    args+=(
        "--fps" "${fps}"
    )

    if [[ -n "${OUTPUT_ROOT}" ]]; then
        args+=("--output-root" "${OUTPUT_ROOT}")
    fi
    if (( ${#METHODS[@]} > 0 )); then
        args+=("--methods")
        args+=("${METHODS[@]}")
    else
        args+=("--methods")
        args+=("${DEFAULT_METHODS[@]}")
    fi
    if [[ -n "${IMAGE_LIMIT}" ]]; then
        args+=("--image-limit" "${IMAGE_LIMIT}")
    fi
    if [[ -n "${BLUR_THRESHOLD}" ]]; then
        args+=("--blur-threshold" "${BLUR_THRESHOLD}")
    fi
    if [[ "${FORCE}" -eq 1 ]]; then args+=("--force"); fi
    if [[ "${dry_run_mode}" -eq 1 ]]; then args+=("--dry-run"); fi

    printf '%s\n' "${args[@]}"
}

find_latest_run_dir() {
    local scene_tag="$1"
    local escaped_scene_tag
    escaped_scene_tag="$(printf '%s' "${scene_tag}" | sed 's/[][(){}.^$+*?|\\/]/\\&/g')"

    local output_root="${OUTPUT_ROOT:-outputs/lab3}"
    if [[ ! -d "${output_root}" ]]; then
        return 0
    fi

    find "${output_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\t%p\n' \
        | awk -F '\t' -v pattern="^[0-9]{8}_[0-9]{6}_${escaped_scene_tag}$" '$1 ~ pattern { print $2 }' \
        | sort \
        | tail -n 1
}

invoke_lab3_sweep_run() {
    local fps="$1"
    local dry_run_mode="$2"
    local scene_tag="${SCENE_NAME}_fps$(format_fps_tag "${fps}")"
    local args=()

    mapfile -t args < <(build_lab3_args "${fps}" "${dry_run_mode}")

    echo
    echo "=== lab3 / scene=${scene_tag} / fps=${fps} ==="
    echo "uv run lab3 ${args[*]}"

    uv run lab3 "${args[@]}"

    local latest_run=""
    latest_run="$(find_latest_run_dir "${scene_tag}")"
    RUN_SUMMARY+=("${scene_tag}|${fps}|${latest_run}|${dry_run_mode}")
}

print_summary() {
    echo
    echo "Sweep finished."
    printf '%-24s %-8s %-7s %s\n' "scene_name" "fps" "dry_run" "run_dir"
    printf '%-24s %-8s %-7s %s\n' "------------------------" "--------" "-------" "-------"

    local row scene_name fps run_dir dry_run
    for row in "${RUN_SUMMARY[@]}"; do
        IFS='|' read -r scene_name fps run_dir dry_run <<< "${row}"
        printf '%-24s %-8s %-7s %s\n' "${scene_name}" "${fps}" "${dry_run}" "${run_dir}"
    done
}

echo "Running lab3 full reconstruction sweep sequentially."
echo "Config: ${CONFIG_PATH}"
echo "InputDir: ${INPUT_DIR}"
echo "Base scene: ${SCENE_NAME}"
echo "FPS list: ${FPS_LIST[*]}"
if (( ${#METHODS[@]} > 0 )); then
    echo "Methods: ${METHODS[*]}"
else
    echo "Methods: ${DEFAULT_METHODS[*]} (default order)"
fi
if [[ -n "${BLUR_THRESHOLD}" ]]; then
    echo "Blur threshold: ${BLUR_THRESHOLD}"
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY RUN - lab3 will print commands without training."
fi

for idx in "${!FPS_LIST[@]}"; do
    fps="${FPS_LIST[$idx]}"
    invoke_lab3_sweep_run "${fps}" "${DRY_RUN}"
done

print_summary
