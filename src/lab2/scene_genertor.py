from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ScenePreset:
    name: str
    description: str


SCENE_PRESETS = {
    "sl_marble_objects": ScenePreset("sl_marble_objects", "Marble-like mixed objects scene"),
    "sl_diffuse_objects": ScenePreset("sl_diffuse_objects", "Simple diffuse baseline scene"),
}


def list_scene_presets() -> list[str]:
    return list(SCENE_PRESETS.keys())


# =============================================================================
# Standard camera and projector configurations for Lab 2
# =============================================================================

def get_standard_camera_config() -> dict[str, Any]:
    """Get standard camera configuration for structured light rendering."""
    return {
        "width": 640,
        "height": 480,
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
        "R": torch.eye(3).tolist(),
        "t": [0.0, 0.0, 0.0],
    }


def get_standard_projector_config() -> dict[str, Any]:
    """Get standard projector configuration for structured light rendering."""
    return {
        "width": 640,
        "height": 480,
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
        "R": torch.eye(3).tolist(),
        "t": [0.1, 0.0, 0.0],  # Slightly offset from camera
    }


def get_standard_render_config() -> dict[str, Any]:
    """Get standard rendering configuration."""
    return {
        "device": "cpu",
        "spp": 64,
        "ambient": 0.12,
    }


def _look_at_from_rt(mi: Any, R: Any, t: Any):
    r_np = np.asarray(R, dtype=np.float32)
    t_np = np.asarray(t, dtype=np.float32).reshape(3)
    c = (-r_np.T @ t_np).reshape(3)
    forward = r_np.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    target = c + forward
    up = r_np.T @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return mi.ScalarTransform4f.look_at(origin=tuple(c.tolist()), target=tuple(target.tolist()), up=tuple(up.tolist()))


def _load_scene_definition(scene_name: str, mi: Any) -> dict[str, Any]:
    """
    Load scene definition from external file.

    Args:
        scene_name: Name of the scene preset
        mi: Mitsuba module instance

    Returns:
        dict: Scene objects (geometry + materials)
    """
    import importlib

    try:
        # Import scene definition module from lab2.scenes
        module = importlib.import_module(f"lab2.scenes.{scene_name}")
        # Call define_scene function
        return module.define_scene(mi)
    except ImportError as e:
        raise ValueError(
            f"Scene definition file not found for '{scene_name}'. "
            f"Expected file: src/lab2/scenes/{scene_name}.py"
        ) from e
    except AttributeError as e:
        raise ValueError(
            f"Scene definition file '{scene_name}' must define a 'define_scene(mi)' function."
        ) from e


def build_runtime_scene_dict(
    mi: Any,
    camera: Any,
    projector: Any,
    ambient: float,
    pattern_path: str,
    scene_name: str = "sl_plane_diffuse",
) -> dict[str, Any]:
    """Build runtime Mitsuba scene dict from a named preset."""
    if scene_name not in SCENE_PRESETS:
        raise ValueError(f"Unknown scene preset: {scene_name}. Available: {list_scene_presets()}")

    camera_fov = float(2.0 * np.degrees(np.arctan(camera.width / (2.0 * camera.fx))))
    proj_fov = float(2.0 * np.degrees(np.arctan(projector.width / (2.0 * projector.fx))))

    base: dict[str, Any] = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 6},
        "sensor": {
            "type": "perspective",
            "fov": camera_fov,
            "to_world": _look_at_from_rt(mi, camera.R, camera.t),
            "sampler": {"type": "independent", "sample_count": 64},
            "film": {
                "type": "hdrfilm",
                "width": camera.width,
                "height": camera.height,
                "pixel_format": "rgb",
                "component_format": "float32",
                "rfilter": {"type": "gaussian"},
            },
        },
        "ambient": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [ambient, ambient, ambient]},
        },
        "projector": {
            "type": "projector",
            "fov": proj_fov,
            "to_world": _look_at_from_rt(mi, projector.R, projector.t),
            "irradiance": {"type": "bitmap", "filename": pattern_path, "raw": True},
        },
        # Weak fill light to improve visibility without overwhelming projected patterns.
        "fill_light": {
            "type": "point",
            "position": [0.0, 0.45, 1.05],
            "intensity": {"type": "rgb", "value": [0.55, 0.55, 0.55]},
        },
    }

    # Load scene definition from external file
    scene_objects = _load_scene_definition(scene_name, mi)
    base.update(scene_objects)

    return base


def generate_scene_bundle(scene_root: str | Path, scene_name: str) -> Path:
    """Generate scene.xml + include files + config.yaml for a named preset."""
    if scene_name not in SCENE_PRESETS:
        raise ValueError(f"Unknown scene preset: {scene_name}. Available: {list_scene_presets()}")

    scene_root = Path(scene_root)
    base_dir = scene_root / "base"
    scene_dir = scene_root / scene_name
    base_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)

    (base_dir / "integrator.xml").write_text(
        "<integrator type=\"path\">\n    <integer name=\"max_depth\" value=\"6\"/>\n</integrator>\n",
        encoding="utf-8",
    )
    (base_dir / "sensor.xml").write_text(
        "<sensor type=\"perspective\">\n    <!-- filled at runtime by renderer config -->\n</sensor>\n",
        encoding="utf-8",
    )
    (base_dir / "emitter.xml").write_text(
        "<emitter type=\"constant\">\n    <rgb name=\"radiance\" value=\"0.05,0.05,0.05\"/>\n</emitter>\n",
        encoding="utf-8",
    )

    (scene_dir / "scene.xml").write_text(
        """<scene version="3.0.0">
    <include filename="../base/integrator.xml"/>
    <include filename="../base/sensor.xml"/>
    <include filename="../base/emitter.xml"/>
    <include filename="geometry.xml"/>
    <include filename="material.xml"/>
</scene>
""",
        encoding="utf-8",
    )

    if scene_name == "sl_plane_diffuse":
        geometry = """<shape type="rectangle">
    <transform name="to_world">
        <scale x="1.8" y="1.2" z="1"/>
    </transform>
    <ref id="mat_main"/>
</shape>
"""
        material = """<bsdf type="diffuse" id="mat_main">
    <rgb name="reflectance" value="0.7, 0.7, 0.7"/>
</bsdf>
"""
    elif scene_name == "sl_marble_objects":
        geometry = """<shape type="sphere">
    <transform name="to_world">
        <translate x="-0.35" y="-0.15" z="1.9"/>
        <scale x="0.24" y="0.24" z="0.24"/>
    </transform>
    <ref id="mat_main"/>
</shape>
<shape type="cube">
    <transform name="to_world">
        <translate x="0.35" y="-0.2" z="2.05"/>
        <scale x="0.24" y="0.24" z="0.24"/>
    </transform>
    <ref id="mat_main"/>
</shape>
"""
        material = """<bsdf type="roughplastic" id="mat_main">
    <float name="alpha" value="0.16"/>
    <rgb name="diffuse_reflectance" value="0.83, 0.83, 0.85"/>
</bsdf>
"""
    elif scene_name == "sl_diffuse_objects":
        geometry = """<shape type="sphere">
    <transform name="to_world">
        <translate x="-0.35" y="-0.15" z="1.9"/>
        <scale x="0.24" y="0.24" z="0.24"/>
    </transform>
    <ref id="mat_main"/>
</shape>
<shape type="cube">
    <transform name="to_world">
        <translate x="0.35" y="-0.2" z="2.05"/>
        <scale x="0.24" y="0.24" z="0.24"/>
    </transform>
    <ref id="mat_main"/>
</shape>
"""
        material = """<bsdf type="diffuse" id="mat_main">
    <rgb name="reflectance" value="0.72, 0.72, 0.72"/>
</bsdf>
"""
    else:
        geometry = """<shape type="sphere">
    <transform name="to_world">
        <translate x="-0.35" y="-0.15" z="1.9"/>
        <scale x="0.24" y="0.24" z="0.24"/>
    </transform>
    <ref id="mat_glass"/>
</shape>
<shape type="cube">
    <transform name="to_world">
        <translate x="0.35" y="-0.2" z="2.05"/>
        <scale x="0.24" y="0.24" z="0.24"/>
    </transform>
    <ref id="mat_wood"/>
</shape>
"""
        material = """<bsdf type="dielectric" id="mat_glass">
    <float name="int_ior" value="1.5"/>
    <float name="ext_ior" value="1.0"/>
</bsdf>
<bsdf type="roughdiffuse" id="mat_wood">
    <float name="alpha" value="0.28"/>
    <rgb name="reflectance" value="0.58, 0.4, 0.25"/>
</bsdf>
"""

    (scene_dir / "geometry.xml").write_text(geometry, encoding="utf-8")
    (scene_dir / "material.xml").write_text(material, encoding="utf-8")
    (scene_dir / "config.yaml").write_text(
        f"""name: {scene_name}
camera:
  width: 640
  height: 480
projector:
  width: 512
  height: 384
patterns:
  K: 4
  Wp: 512
renderer:
  spp: 16
  variant: cuda_ad_rgb
""",
        encoding="utf-8",
    )

    return scene_dir


def generate_all_scene_bundles(scene_root: str | Path) -> list[Path]:
    out: list[Path] = []
    for name in list_scene_presets():
        out.append(generate_scene_bundle(scene_root=scene_root, scene_name=name))
    return out


# =============================================================================
# Scene loading functions for training pipeline
# =============================================================================

def load_scene_with_standard_config(
    renderer: Any,
    scene_name: str,
    camera_config: dict[str, Any] | None = None,
    projector_config: dict[str, Any] | None = None,
) -> None:
    """
    Load a scene with standard camera and projector configurations.

    Args:
        renderer: StructuredLightRenderer instance
        scene_name: Name of the scene preset to load
        camera_config: Optional camera config (uses standard if None)
        projector_config: Optional projector config (uses standard if None)
    """
    if camera_config is None:
        camera_config = get_standard_camera_config()
    if projector_config is None:
        projector_config = get_standard_projector_config()

    renderer.set_scene_name(scene_name)
    renderer.set_camera(camera_config)
    renderer.set_projector(projector_config)
    renderer.load_scene()


def create_standard_renderer(
    scene_name: str,
    device: str = "cpu",
    spp: int = 64,
    camera_config: dict[str, Any] | None = None,
    projector_config: dict[str, Any] | None = None,
    backend: str = "pytorch",
) -> Any:
    """
    Create a renderer with standard configurations for the given scene.

    This is a convenience function for training scripts.

    Args:
        scene_name: Name of the scene preset to load
        device: Device to use for rendering
        spp: Samples per pixel
        camera_config: Optional camera config (uses standard if None)
        projector_config: Optional projector config (uses standard if None)

    Returns:
        Configured StructuredLightRenderer instance
    """
    if backend == "mitsuba":
        from lab2.shader import StructuredLightRenderer

        renderer = StructuredLightRenderer(device=device, spp=spp)
    elif backend == "pytorch":
        from lab2.pytorch_renderer import PytorchStructuredLightRenderer

        renderer = PytorchStructuredLightRenderer(device=device, spp=spp)
    else:
        raise ValueError(f"Unknown renderer backend: {backend}")

    load_scene_with_standard_config(
        renderer,
        scene_name,
        camera_config=camera_config,
        projector_config=projector_config,
    )

    return renderer
