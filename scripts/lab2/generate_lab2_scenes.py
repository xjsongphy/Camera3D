"""
Generate all scene bundles for Lab 2 experiments.

Creates scene files for:
- Diffuse plane (baseline)
- Marble objects
- Wood + glass materials
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "lab2"))
from scene_genertor import generate_all_scene_bundles


if __name__ == "__main__":
    root = Path("assets/scenes")
    root.mkdir(parents=True, exist_ok=True)
    created = generate_all_scene_bundles(root)

    print(f"Generated {len(created)} scene bundles:")
    for p in created:
        print(f"  - {p}")
