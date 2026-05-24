"""Lab2 structured-light renderer package."""

from .scene_genertor import generate_all_scene_bundles, generate_scene_bundle, list_scene_presets
from .shader import StructuredLightRenderer

__all__ = [
    "StructuredLightRenderer",
    "list_scene_presets",
    "generate_scene_bundle",
    "generate_all_scene_bundles",
]
