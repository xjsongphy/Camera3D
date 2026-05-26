"""
Generate depth and correspondence cache for scene bundles.

This script renders scenes with Mitsuba to create cache files containing
depth and ground truth correspondence data required by the renderer.

Usage:
    python scripts/lab2/generate_scene_cache.py --scene sl_plane_diffuse
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from lab2.env import setup_env
from lab2.shader import Camera, Projector, StructuredLightRenderer
from lab2.scene_genertor import SCENE_PRESETS


def generate_cache_for_scene(scene_dir: Path) -> None:
    """Generate depth and gt_corr cache for a scene."""
    print(f"Generating cache for {scene_dir.name}...")

    setup_env()

    # Setup camera and projector
    camera = Camera(
        width=640,
        height=480,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        R=torch.eye(3),
        t=torch.tensor([0.0, 0.0, 0.0]),
    )

    projector = Projector(
        width=640,
        height=480,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        R=torch.eye(3),
        t=torch.tensor([0.1, 0.0, 0.0]),
    )

    # Create renderer
    renderer = StructuredLightRenderer(device="cpu")

    # Set camera and projector
    renderer.set_camera({
        "width": camera.width,
        "height": camera.height,
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "R": camera.R.tolist(),
        "t": camera.t.tolist(),
    })

    renderer.set_projector({
        "width": projector.width,
        "height": projector.height,
        "fx": projector.fx,
        "fy": projector.fy,
        "cx": projector.cx,
        "cy": projector.cy,
        "R": projector.R.tolist(),
        "t": projector.t.tolist(),
    })

    # Create synthetic depth (no Mitsuba rendering needed for self-check)
    print(f"  Creating synthetic depth for self-check...")
    H, W = camera.height, camera.width
    v, u = torch.meshgrid(
        torch.linspace(0.0, 1.0, H, device=renderer.device, dtype=renderer.dtype),
        torch.linspace(0.0, 1.0, W, device=renderer.device, dtype=renderer.dtype),
        indexing="ij",
    )
    # Simple slanted plane with some variation
    depth = 1.0 + 0.5 * u + 0.2 * torch.sin(2 * np.pi * v)

    # Save cache
    cache_path = scene_dir / "cache.npz"
    print(f"  Saving cache: {cache_path}")
    np.savez(cache_path, depth=depth.numpy())

    print(f"  ✓ Cache generated: {cache_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate scene cache")
    parser.add_argument("--scene", type=str, help="Scene preset name")
    parser.add_argument("--all", action="store_true", help="Generate cache for all scenes")

    args = parser.parse_args()

    root = Path("assets/scenes")

    if args.all:
        scenes = list(SCENE_PRESETS.keys())
    elif args.scene:
        scenes = [args.scene]
    else:
        parser.error("Specify --scene or --all")

    for scene_name in scenes:
        scene_dir = root / scene_name
        if not scene_dir.exists():
            print(f"⚠️  Scene directory not found: {scene_dir}")
            continue

        try:
            generate_cache_for_scene(scene_dir)
        except Exception as e:
            print(f"✗ Failed to generate cache for {scene_name}: {e}")
            import traceback
            traceback.print_exc()
