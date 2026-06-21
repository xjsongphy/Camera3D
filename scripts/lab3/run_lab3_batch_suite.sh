#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

# Boya reconstructions.
BOYA_CLOSE_CONFIG_PATH="${BOYA_CLOSE_CONFIG_PATH:-configs/lab3/boya_close.json}"
BOYA_FAR_CONFIG_PATH="${BOYA_FAR_CONFIG_PATH:-configs/lab3/boya_far.json}"

FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${FORCE}" != "0" && "${FORCE}" != "1" ]]; then
    echo "FORCE must be 0 or 1" >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1" >&2
    exit 2
fi

RUN_SUMMARY=()

run_and_record() {
    local label="$1"
    shift
    echo
    echo "=== ${label} ==="
    echo "uv run lab3 $*"
    uv run lab3 "$@"
    RUN_SUMMARY+=("${label}|uv run lab3 $*")
}

echo "Running lab3 batch suite sequentially."
echo "1) Reconstruct boya_close via ${BOYA_CLOSE_CONFIG_PATH}"
echo "2) Reconstruct boya_far via ${BOYA_FAR_CONFIG_PATH}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY RUN - lab3 will only print commands where supported."
fi

boya_close_args=("--config" "${BOYA_CLOSE_CONFIG_PATH}")
if [[ "${FORCE}" -eq 1 ]]; then
    boya_close_args+=("--force")
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
    boya_close_args+=("--dry-run")
fi
run_and_record "reconstruct_boya_close" "${boya_close_args[@]}"

boya_far_args=("--config" "${BOYA_FAR_CONFIG_PATH}")
if [[ "${FORCE}" -eq 1 ]]; then
    boya_far_args+=("--force")
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
    boya_far_args+=("--dry-run")
fi
run_and_record "reconstruct_boya_far" "${boya_far_args[@]}"

echo
echo "Batch suite finished."
printf '%-28s %s\n' "step" "command"
printf '%-28s %s\n' "----------------------------" "-------"
for row in "${RUN_SUMMARY[@]}"; do
    IFS='|' read -r label command <<< "${row}"
    printf '%-28s %s\n' "${label}" "${command}"
done
