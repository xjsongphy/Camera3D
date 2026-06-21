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

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lab3.common import run_cmd
from lab3.reconstruction import RECONSTRUCTIONS, create_default_reconstructor
from lab3.reconstruction.base import ViewerTarget


@dataclass(frozen=True)
class ViewerConfig:
    open3d_bin: str = "python"
    script_path: Path | None = None
    dry_run: bool = False


def find_viewer_targets(run_dir: Path, methods: Iterable[str]) -> dict[str, list[Path]]:
    """Locate each method's viewable artifacts through its lifecycle adapter."""
    reconstructors = [create_default_reconstructor(method) for method in methods]
    return {
        reconstructor.name: [target.path for target in reconstructor.viewer_targets(run_dir)]
        for reconstructor in reconstructors
    }


def _viewer_target_records(run_dir: Path, methods: Iterable[str]) -> list[ViewerTarget]:
    records: list[ViewerTarget] = []
    for method in methods:
        records.extend(create_default_reconstructor(method).viewer_targets(run_dir))
    return records


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
    methods: Iterable[str] = tuple(RECONSTRUCTIONS),
    *,
    nerfstudio_viewer: bool = True,
    dry_run: bool = False,
    nerf_viewer_bin: str = "lab3-viewer",
) -> None:
    """Launch the appropriate interactive viewer for each requested method."""
    if not run_dir.exists():
        print(f"[lab3.visualization] run directory not found: {run_dir}")
        return

    target_records = _viewer_target_records(run_dir, methods)
    geometry_paths = [target.path for target in target_records if target.kind == "geometry"]
    if geometry_paths:
        print(f"[lab3.visualization] Open3D geometry viewer: {geometry_paths}")
        if not dry_run:
            open3d_viewer(geometry_paths, title=f"lab3 geometry: {run_dir.name}")
    elif not target_records:
        print("[lab3.visualization] no staged geometry found; run `lab3` with geometry stage on first.")

    if nerfstudio_viewer:
        for target in target_records:
            if target.kind != "nerfstudio":
                continue
            cmd = target.command(nerf_viewer_bin)
            print("$", " ".join(cmd))
            if dry_run:
                continue
            if shutil.which(nerf_viewer_bin) is None:
                print(
                    f"[lab3.visualization] {nerf_viewer_bin} not found in PATH; "
                    "install nerfstudio to use the web viewer."
                )
                continue
            run_cmd(cmd)

    for target in target_records:
        if target.kind != "sibr":
            continue
        _launch_sibr_viewer(target, dry_run=dry_run)


def _launch_sibr_viewer(target: ViewerTarget, *, dry_run: bool) -> None:
    repo_dir = _resolve_repo_dir(target)
    viewer_bin = _find_sibr_binary(repo_dir)
    if viewer_bin is None:
        build_cmd = _sibr_build_command(repo_dir)
        print(
            "[lab3.visualization] SIBR viewer binary SIBR_gaussianViewer_app not found; "
            f"build it first via: {' '.join(build_cmd)}"
        )
        if dry_run:
            return
        run_cmd(build_cmd)
        viewer_bin = _find_sibr_binary(repo_dir)
        if viewer_bin is None:
            print(
                "[lab3.visualization] SIBR build completed but the viewer binary is still missing "
                f"under {repo_dir / 'SIBR_viewers' / 'install' / 'bin'}"
            )
            return
    cmd = [str(viewer_bin), "-m", str(target.path)]
    print("$", " ".join(cmd))
    if not dry_run:
        run_cmd(cmd)


def _resolve_repo_dir(target: ViewerTarget) -> Path:
    configured = Path(target.launcher_args[0]) if target.launcher_args else Path("gaussian-splatting")
    if configured.is_absolute():
        return configured
    return (Path.cwd() / configured).resolve()


def _find_sibr_binary(repo_dir: Path) -> Path | None:
    bin_dir = repo_dir / "SIBR_viewers" / "install" / "bin"
    candidates = (
        "SIBR_gaussianViewer_app.exe",
        "SIBR_gaussianViewer_app",
        "SIBR_gaussianViewer_app_rwdi.exe",
        "SIBR_gaussianViewer_app_config.exe",
        "SIBR_gaussianViewer_app_config",
    )
    for name in candidates:
        path = bin_dir / name
        if path.exists():
            return path
    return None


def _sibr_build_command(repo_dir: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    sibr_source = _relative_to_workspace(repo_dir / "SIBR_viewers", repo_root)
    system = platform.system()
    if system == "Windows":
        script = repo_root / "scripts" / "lab3" / "build_sibr_viewer.ps1"
        return [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "--sibr-source",
            sibr_source,
        ]
    script = repo_root / "scripts" / "lab3" / "build_sibr_viewer.sh"
    return ["bash", str(script), "--sibr-source", sibr_source]


def _relative_to_workspace(path: Path, workspace_root: Path) -> str:
    return os.path.relpath(path.resolve(), workspace_root.resolve())
