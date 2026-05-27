from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ScenePreset:
    name: str
    description: str


SCENE_PRESETS = {
    "sl_plane_diffuse": ScenePreset("sl_plane_diffuse", "Diffuse plane baseline scene"),
    "sl_marble_objects": ScenePreset("sl_marble_objects", "Marble-like mixed objects scene"),
    "sl_wood_glass": ScenePreset("sl_wood_glass", "Wood-like + glass mixed material scene"),
}


def list_scene_presets() -> list[str]:
    return list(SCENE_PRESETS.keys())


def _look_at_from_rt(mi: Any, R: Any, t: Any):
    r_np = np.asarray(R, dtype=np.float32)
    t_np = np.asarray(t, dtype=np.float32).reshape(3)
    c = (-r_np.T @ t_np).reshape(3)
    forward = r_np.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    target = c + forward
    up = r_np.T @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return mi.ScalarTransform4f.look_at(origin=tuple(c.tolist()), target=tuple(target.tolist()), up=tuple(up.tolist()))


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
    }

    # Mitsuba rectangle is in XY plane with normal facing +Z.
    # Camera looks along +Z, so we need to rotate rectangles by -90 deg around X
    # to make the normal face -Z (toward the camera).
    _ground_rot = mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=-90)

    if scene_name == "sl_plane_diffuse":
        base["target"] = {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0.0, 0.0, 2.0)) @ _ground_rot @ mi.ScalarTransform4f.scale((1.8, 1.2, 1.0)),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.7, 0.7, 0.7]}},
        }
    elif scene_name == "sl_marble_objects":
        base["ground"] = {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0.0, -0.45, 1.6)) @ _ground_rot @ mi.ScalarTransform4f.scale((2.2, 1.2, 1.0)),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.76, 0.76, 0.76]}},
        }
        base["sphere"] = {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate((-0.35, -0.15, 1.9)) @ mi.ScalarTransform4f.scale((0.24, 0.24, 0.24)),
            "bsdf": {
                "type": "roughplastic",
                "alpha": 0.14,
                "diffuse_reflectance": {"type": "rgb", "value": [0.83, 0.83, 0.85]},
            },
        }
        base["cube"] = {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate((0.35, -0.2, 2.05)) @ mi.ScalarTransform4f.scale((0.24, 0.24, 0.24)),
            "bsdf": {
                "type": "roughplastic",
                "alpha": 0.19,
                "diffuse_reflectance": {"type": "rgb", "value": [0.84, 0.84, 0.87]},
            },
        }
    elif scene_name == "sl_wood_glass":
        base["ground"] = {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate((0.0, -0.48, 1.75)) @ _ground_rot @ mi.ScalarTransform4f.scale((2.3, 1.2, 1.0)),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [0.56, 0.42, 0.30]},
            },
        }
        base["glass_sphere"] = {
            "type": "sphere",
            "to_world": mi.ScalarTransform4f.translate((-0.28, -0.2, 1.9)) @ mi.ScalarTransform4f.scale((0.23, 0.23, 0.23)),
            "bsdf": {"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0},
        }
        base["wood_cube"] = {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.translate((0.35, -0.22, 2.03)) @ mi.ScalarTransform4f.scale((0.22, 0.22, 0.22)),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [0.58, 0.40, 0.25]},
            },
        }

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
    else:
        geometry = """<shape type="sphere">
    <transform name="to_world">
        <translate x="-0.28" y="-0.2" z="1.9"/>
        <scale x="0.23" y="0.23" z="0.23"/>
    </transform>
    <ref id="mat_glass"/>
</shape>
<shape type="cube">
    <transform name="to_world">
        <translate x="0.35" y="-0.22" z="2.03"/>
        <scale x="0.22" y="0.22" z="0.22"/>
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
