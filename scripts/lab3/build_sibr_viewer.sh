#!/bin/bash
set -euo pipefail

# build_sibr_viewer.sh - Linux build script for SIBR Viewer
# Adapted from build_sibr_viewer.ps1 for Linux platforms

# Default parameters
SIBR_SOURCE="${SIBR_SOURCE:-gaussian-splatting/SIBR_viewers}"
BUILD_DIR="${BUILD_DIR:-}"
GENERATOR="${GENERATOR:-Auto}"
CONFIG="${CONFIG:-RelWithDebInfo}"
CLEAN=${CLEAN:-0}
OPEN_SOLUTION=${OPEN_SOLUTION:-0}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SIBR_ROOT="$(cd "${REPO_ROOT}/${SIBR_SOURCE}" && pwd)"

# Determine build directory
if [[ -z "${BUILD_DIR}" ]]; then
    BUILD_PATH="${SIBR_ROOT}/build"
else
    BUILD_PATH="${REPO_ROOT}/${BUILD_DIR}"
fi

# Log directory
LOG_DIR="${BUILD_PATH}/logs"
LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# Logging functions
log_info() {
    echo -e "${CYAN}[build_sibr_viewer]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[build_sibr_viewer]${NC} $*"
}

log_error() {
    echo -e "${RED}[build_sibr_viewer]${NC} $*"
}

log_step() {
    echo ""
    echo -e "${CYAN}==>${NC} $*"
}

# Check for required tools
check_dependencies() {
    if ! command -v cmake &> /dev/null; then
        log_error "cmake not found in PATH"
        return 1
    fi

    if [[ "${GENERATOR}" == "Ninja" ]] && ! command -v ninja &> /dev/null; then
        log_error "Ninja generator requested but ninja not found in PATH"
        return 1
    fi
}

# Clean build directory
clean_build() {
    if [[ "${CLEAN}" -eq 1 ]] && [[ -d "${BUILD_PATH}" ]]; then
        log_info "Cleaning build directory: ${BUILD_PATH}"

        # Kill any running cmake/build processes
        local pids=$(pgrep -f "cmake.*${BUILD_PATH}" || true)
        if [[ -n "${pids}" ]]; then
            log_warn "Killing running build processes..."
            echo "${pids}" | xargs kill -9 2>/dev/null || true
            sleep 1
        fi

        rm -rf "${BUILD_PATH}"
        mkdir -p "${BUILD_PATH}"
    fi
}

# Get CMake generator
get_cmake_generator() {
    local gen="Unix Makefiles"
    if [[ "${GENERATOR}" == "Auto" ]]; then
        # Prefer Ninja if available
        if command -v ninja &> /dev/null; then
            gen="Ninja"
        fi
    else
        gen="${GENERATOR}"
    fi
    echo "${gen}"
}

# Run command with logging
run_logged_command() {
    local cmd=("$@")
    local display_name="${cmd[0]}"
    local log_file="${LOG_DIR}/build-${LOG_TIMESTAMP}.log"

    log_info "Running: ${display_name}"
    log_info "Log: ${log_file}"

    "${cmd[@]}" 2>&1 | tee "${log_file}"
    local exit_code=${PIPESTATUS[0]}

    if [[ ${exit_code} -ne 0 ]]; then
        log_error "${display_name} failed with exit code ${exit_code}"
        log_error "See log: ${log_file}"
        return ${exit_code}
    fi

    return 0
}

# Main build function
main() {
    log_info "Source: ${SIBR_ROOT}"
    log_info "Build:  ${BUILD_PATH}"
    log_info "Config: ${CONFIG}"

    # Check dependencies
    check_dependencies || exit 1

    # Clean if requested
    clean_build

    # Create build directory
    mkdir -p "${BUILD_PATH}"

    # Determine generator
    local generator=$(get_cmake_generator)
    log_info "Generator: ${generator}"

    # Configure
    log_step "Configure CMake"
    local configure_args=(
        --fresh
        -S "${SIBR_ROOT}"
        -B "${BUILD_PATH}"
        -G "${generator}"
        -DCMAKE_BUILD_TYPE="${CONFIG}"
        -DCMAKE_INSTALL_PREFIX="${SIBR_ROOT}/install"
    )

    if ! run_logged_command cmake "${configure_args[@]}"; then
        log_error "CMake configuration failed"
        exit 1
    fi

    # Build
    log_step "Build SIBR Viewer"
    local build_args=(
        --build "${BUILD_PATH}"
        --target install
        --config "${CONFIG}"
        --parallel "$(nproc)"
    )

    if ! run_logged_command cmake "${build_args[@]}"; then
        log_error "Build failed"
        exit 1
    fi

    # Check for built viewer
    local viewer_bin="${SIBR_ROOT}/install/bin/SIBR_gaussianViewer_app"
    if [[ -f "${viewer_bin}" ]]; then
        log_info "Built viewer: ${viewer_bin}"
    elif [[ -f "${viewer_bin}_config" ]]; then
        log_info "Built viewer: ${viewer_bin}_config"
    else
        log_warn "Build finished, but SIBR_gaussianViewer_app was not found under install/bin"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sibr-source)
            SIBR_SOURCE="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --generator)
            GENERATOR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --no-clean)
            CLEAN=0
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --sibr-source PATH    Path to SIBR source (default: gaussian-splatting/SIBR_viewers)"
            echo "  --build-dir PATH      Build directory (default: <sibr-source>/build)"
            echo "  --generator GEN       CMake generator (Auto, Ninja, Unix Makefiles)"
            echo "  --config CFG          Build config (default: RelWithDebInfo)"
            echo "  --clean               Clean build directory before building"
            echo "  --no-clean            Skip cleaning"
            exit 1
            ;;
    esac
done

# Run main
main
