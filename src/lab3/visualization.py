"""Interactive visualization for Lab 3 (assignment §8 bonus: interactive viewer).

Each representation has a natural interactive viewer:

- SfM point cloud / COLMAP dense / Poisson mesh -> Open3D window (orbit, zoom,
  normals coloring).
- 3DGS Gaussian ``.ply`` -> Open3D window (shows Gaussian centres / distribution
  and floaters; note Open3D ignores the SH colour channels, so for *true*
  splatted rendering use the repo's SIBR viewer -- documented in the README).
- NeRF (nerfstudio) -> ``ns-viewer`` web viewer.

``view_run_dir`` orchestrates the right viewer per method on an existing run.
It is opt-in and dry-run aware; the Open3D / ns-viewer launches are GUI/network
operations the user runs locally, so they are guarded and never executed under
``--dry-run``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lab3.common import run_cmd


@dataclass(frozen=True)
class ViewerConfig:
    open3d_bin: str = "python"
    script_path: Path | None = None
    dry_run: bool = False


def find_viewer_targets(run_dir: Path, methods: Iterable[str]) -> dict[str, list[Path]]:
    """Locate each method's viewable artifact: geometry files + nerf config.yml."""
    methods = set(methods)
    geometry = run_dir / "geometry"
    results = run_dir / "results"
    targets: dict[str, list[Path]] = {}

    if "sfm" in methods:
        candidates = [
            geometry / "sfm" / "dense.ply",
            geometry / "sfm" / "poisson_mesh.ply",
            results / "sfm" / "dense" / "fused.ply",
        ]
        targets["sfm"] = [c for c in candidates if c.exists()]

    if "3dgs" in methods:
        candidates: list[Path] = [geometry / "3dgs" / "gaussians.ply"]
        candidates += sorted((results / "3dgs" / "point_cloud").glob("iteration_*/point_cloud.ply"))
        targets["3dgs"] = [c for c in candidates if c.exists()]

    if "nerf" in methods:
        train_root = results / "nerf" / "train"
        configs = sorted(train_root.rglob("config.yml")) if train_root.exists() else []
        targets["nerf"] = configs[-1:] if configs else []

    return targets


def build_nerfstudio_viewer_command(viewer_bin: str, config_path: Path) -> list[str]:
    """``ns-viewer --load-config <config.yml>`` -> interactive web viewer."""
    return [viewer_bin, "--load-config", str(config_path)]


def open3d_viewer(paths: Iterable[Path], *, title: str = "lab3") -> None:
    """Open an interactive Open3D window over the given point clouds / meshes.

    Best-effort: needs the optional ``open3d`` package and a display. A ``.ply``
    is loaded as a triangle mesh when it has faces, otherwise as a point cloud
    (this is how 3DGS Gaussian ``.ply`` files are inspected -- their SH colour
    channels are ignored, so they render uncoloured; use SIBR for splatting).
    """
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(
            f"[lab3.visualization] open3d unavailable ({exc}). "
            "Install with `pip install open3d` to use the interactive viewer."
        )
        return

    geometries = []
    for path in paths:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            geometries.append(mesh)
        else:
            geometries.append(o3d.io.read_point_cloud(str(path)))

    if not geometries:
        print("[lab3.visualization] no geometry to display.")
        return
    o3d.visualization.draw_geometries(geometries, window_name=title)


def view_run_dir(
    run_dir: Path,
    methods: Iterable[str] = ("sfm", "3dgs", "nerf"),
    *,
    nerfstudio_viewer: bool = True,
    dry_run: bool = False,
    nerf_viewer_bin: str = "lab3-viewer",
) -> None:
    """Launch the appropriate interactive viewer for each requested method."""
    if not run_dir.exists():
        print(f"[lab3.visualization] run directory not found: {run_dir}")
        return

    targets = find_viewer_targets(run_dir, methods)

    geometry_paths: list[Path] = []
    for method in ("sfm", "3dgs"):
        geometry_paths.extend(targets.get(method, []))
    if geometry_paths:
        print(f"[lab3.visualization] Open3D geometry viewer: {geometry_paths}")
        if not dry_run:
            open3d_viewer(geometry_paths, title=f"lab3 geometry: {run_dir.name}")
    elif any(m in methods for m in ("sfm", "3dgs")):
        print("[lab3.visualization] no staged geometry found; run `lab3` with geometry stage on first.")

    if nerfstudio_viewer and "nerf" in methods and targets.get("nerf"):
        config_path = targets["nerf"][0]
        cmd = build_nerfstudio_viewer_command(nerf_viewer_bin, config_path)
        print("$", " ".join(cmd))
        if not dry_run:
            if shutil.which(nerf_viewer_bin) is None:
                print(
                    f"[lab3.visualization] {nerf_viewer_bin} not found in PATH; "
                    "install nerfstudio to use the web viewer."
                )
            else:
                run_cmd(cmd)  # opens web viewer; blocks until the user closes it
    elif nerfstudio_viewer and "nerf" in methods:
        print("[lab3.visualization] no nerfstudio config.yml found under results/nerf/train.")
