from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "lab2"))
from scene_genertor import generate_all_scene_bundles


if __name__ == "__main__":
    root = Path("assets/scenes")
    created = generate_all_scene_bundles(root)
    for p in created:
        print(p)
