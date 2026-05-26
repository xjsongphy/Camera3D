"""
Environment setup for Lab 2.

This module handles platform-specific dependencies and environment variables:
- macOS: LLVM library path for Mitsuba/Drjit
- All platforms: OpenEXR support for OpenCV

Usage:
    from lab2 import setup_env
    setup_env()

Or automatically import on module load:
    from lab2 import env  # Automatically calls setup_env()
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _setup_llvm_path() -> bool:
    """
    Setup DRJIT_LIBLLVM_PATH for macOS.

    On macOS, Mitsuba/Drjit need the LLVM shared library.
    DRJIT_LIBLLVM_PATH must point to the full path of libLLVM.dylib,
    not just the directory containing it.

    Returns:
        True if LLVM path was set, False otherwise
    """
    if sys.platform != "darwin":
        return False

    # Already set by user
    if os.environ.get('DRJIT_LIBLLVM_PATH'):
        return True

    # Try to find LLVM via brew
    try:
        llvm_prefix = subprocess.check_output(
            ['brew', '--prefix', 'llvm'],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Point to libLLVM.dylib (not libLLVM-C.dylib)
        llvm_lib = Path(llvm_prefix) / 'lib' / 'libLLVM.dylib'

        if llvm_lib.exists():
            os.environ['DRJIT_LIBLLVM_PATH'] = str(llvm_lib)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # brew not found

    # Fallback: try common paths
    common_paths = [
        Path('/opt/homebrew/opt/llvm/lib/libLLVM.dylib'),
        Path('/usr/local/opt/llvm/lib/libLLVM.dylib'),
        Path('/opt/homebrew/Cellar/llvm/lib/libLLVM.dylib'),
    ]

    for path in common_paths:
        if path.exists():
            os.environ['DRJIT_LIBLLVM_PATH'] = str(path)
            return True

    return False


def _setup_opencv_exr() -> None:
    """
    Enable OpenEXR support for OpenCV.

    Required for reading/writing .exr files in Mitsuba patterns.
    """
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


def setup_env() -> dict[str, str]:
    """
    Setup all environment variables for Lab 2.

    This function should be called before importing Mitsuba or
    using the renderer. It's safe to call multiple times.

    Returns:
        Dictionary of environment variables that were set
    """
    result = {}

    # Setup LLVM path (macOS only)
    if _setup_llvm_path():
        result['DRJIT_LIBLLVM_PATH'] = os.environ['DRJIT_LIBLLVM_PATH']

    # Setup OpenEXR support
    _setup_opencv_exr()
    result['OPENCV_IO_ENABLE_OPENEXR'] = os.environ['OPENCV_IO_ENABLE_OPENEXR']

    return result


def check_dependencies() -> dict[str, bool]:
    """
    Check if required dependencies are available.

    Returns:
        Dictionary with dependency names as keys and availability as values
    """
    result = {
        'torch': False,
        'mitsuba': False,
        'drjit': False,
        'opencv': False,
        'llvm': False,
    }

    try:
        import torch
        result['torch'] = True
    except ImportError:
        pass

    try:
        import mitsuba as mi
        result['mitsuba'] = True
    except ImportError:
        pass

    try:
        import drjit
        result['drjit'] = True
    except ImportError:
        pass

    try:
        import cv2
        result['opencv'] = True
    except ImportError:
        pass

    # Check LLVM library (macOS)
    if sys.platform == "darwin":
        result['llvm'] = bool(os.environ.get('DRJIT_LIBLLVM_PATH'))
        # Also verify file exists
        if result['llvm']:
            result['llvm'] = Path(os.environ['DRJIT_LIBLLVM_PATH']).exists()

    return result


def print_dependency_status() -> None:
    """Print status of all dependencies."""
    status = check_dependencies()

    print("Lab 2 Dependency Status:")
    print("-" * 40)
    for name, available in status.items():
        icon = "✓" if available else "✗"
        print(f"  {icon} {name}")

    # Print warnings for missing dependencies
    if not status['torch']:
        print("\n⚠️  PyTorch not found. Install with: uv sync --group lab2")
    if not status['mitsuba']:
        print("⚠️  Mitsuba not found. Install with: uv sync --group lab2")
    if sys.platform == "darwin" and not status['llvm']:
        print("⚠️  LLVM not found. Install with: brew install llvm")


# Auto-setup on module import
setup_env()


if __name__ == "__main__":
    # Test environment setup
    print("Testing Lab 2 environment setup...")
    print()

    env_vars = setup_env()
    print("Environment variables set:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")
    print()

    print_dependency_status()
