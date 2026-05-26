"""Lab2 structured-light renderer package.

This module automatically sets up required environment variables on import:
- macOS: LLVM library path for Mitsuba/Drjit
- All platforms: OpenEXR support for OpenCV
"""

# Import env first to setup environment variables
from . import env  # noqa: F401 (imported for side effects)

from .scene_genertor import generate_all_scene_bundles, generate_scene_bundle, list_scene_presets
from .shader import StructuredLightRenderer

__all__ = [
    "StructuredLightRenderer",
    "list_scene_presets",
    "generate_scene_bundle",
    "generate_all_scene_bundles",
    "env",
]
