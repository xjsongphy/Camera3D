"""Geometry staging for Lab 3 (assignment §5.3 / §4.4).

Collects each method's geometric artifacts (COLMAP dense point cloud, 3DGS
Gaussian ``.ply``, optional Poisson mesh) into a single ``geometry/`` tree so
they can be inspected and compared. Mesh reconstruction is best-effort: it uses
Open3D when available and is otherwise skipped, matching the dry-run philosophy.

Also computes *quantitative* geometry metrics (Chamfer distance + F-score,
assignment §5.3 optional) comparing each method's point cloud to the COLMAP
dense cloud as a proxy. The metric math is pure NumPy (downsampled so it stays
cheap and dependency-free); only the ``.ply`` loading shells out to Open3D as a
best-effort step. Following the assignment's honesty rule, the COLMAP cloud is a
*proxy*, not ground truth — its own holes/noise bias the numbers.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from lab3.common import copy_file

# Subsample both clouds to this many points before the brute-force nearest
# neighbour search: keeps memory/time bounded and dependency-free. Relative
# comparisons stay meaningful even on the subsample; report ``downsample_cap``.
DEFAULT_DOWNSAMPLE_CAP = 4096
GEOMETRY_METRIC_COLUMNS = [
    "method",
    "proxy",
    "chamfer_to_proxy",
    "chamfer_from_proxy",
    "chamfer_sym",
    "fscore_05pct",
    "fscore_1pct",
    "n_method",
    "n_proxy",
    "proxy_diag",
    "downsample_cap",
    "note",
]


def find_3dgs_pointcloud(model_dir: Path) -> Path | None:
    """Latest 3DGS trained point cloud (``point_cloud/iteration_*/point_cloud.ply``)."""
    if not model_dir.is_dir():
        return None
    matches = sorted((model_dir / "point_cloud").glob("iteration_*/point_cloud.ply"))
    return matches[-1] if matches else None


def find_sfm_dense(sfm_dir: Path) -> Path | None:
    fused = sfm_dir / "dense" / "fused.ply"
    return fused if fused.exists() else None


def copy_geometry(src: Path, dest_dir: Path, name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    copy_file(src, dest, overwrite=True)
    return dest


def maybe_poisson_mesh(points_ply: Path, out_ply: Path) -> Path | None:
    """Best-effort Poisson surface reconstruction from a point cloud via Open3D."""
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[lab3.geometry] open3d unavailable ({exc}); skipping Poisson mesh.")
        return None
    pcd = o3d.io.read_point_cloud(str(points_ply))
    if not pcd.has_points():
        print(f"[lab3.geometry] empty/unreadable point cloud {points_ply}; skipping Poisson mesh.")
        return None
    pcd.estimate_normals()
    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    except RuntimeError as exc:
        print(f"[lab3.geometry] Poisson reconstruction failed ({exc}); skipping mesh.")
        return None
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_ply), mesh)
    return out_ply


def stage_exported_geometry(context: Any, method: str) -> list[Path]:
    """Stage geometry files already exported inside one method result tree."""
    method_dir: Path = context.run_dir / "results" / method
    destination: Path = context.run_dir / "geometry" / method
    collected: list[Path] = []
    for suffix in ("*.ply", "*.obj", "*.glb"):
        for path in method_dir.rglob(suffix):
            collected.append(copy_geometry(path, destination, path.name))
    return collected


def stage_geometry(context: Any, reconstructors: Sequence[Any]) -> dict[str, list[Path]]:
    """Ask every reconstructor to stage its own geometric representation."""
    if getattr(context, "dry_run", False):
        return {reconstructor.name: [] for reconstructor in reconstructors}

    staged: dict[str, list[Path]] = {}
    for reconstructor in reconstructors:
        collected = reconstructor.stage_geometry(context)
        if collected:
            staged[reconstructor.name] = collected

    return staged


# --------------------------------------------------------------------------- #
# Quantitative geometry metrics (§5.3): Chamfer distance + F-score             #
# --------------------------------------------------------------------------- #
def bbox_diagonal(points: np.ndarray) -> float:
    """Length of the bounding-box diagonal — used as a scale reference."""
    pts = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def downsample_points(points: np.ndarray, cap: int = DEFAULT_DOWNSAMPLE_CAP, seed: int = 0) -> np.ndarray:
    """Uniformly subsample to at most ``cap`` points; deterministic for a given seed."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=cap, replace=False)
        pts = pts[idx]
    return pts


def _nearest_sq_distances(a: np.ndarray, b: np.ndarray, block: int = 512) -> np.ndarray:
    """Squared distance from each point in ``a`` to its nearest point in ``b``.

    Chunked over ``a`` so peak memory stays small even at the default cap.
    """
    out = np.empty(a.shape[0], dtype=np.float64)
    for start in range(0, a.shape[0], block):
        end = min(start + block, a.shape[0])
        diff = a[start:end, None, :] - b[None, :, :]
        out[start:end] = (diff * diff).sum(axis=-1).min(axis=-1)
    return out


def chamfer_distances(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Directional and symmetric mean nearest-neighbour distances (L2).

    ``to_b`` averages each point of ``a`` to its NN in ``b``; ``from_b`` is the
    reverse; ``sym`` is their mean (the usual symmetric Chamfer-L2).
    """
    d_ab = np.sqrt(_nearest_sq_distances(a, b)).mean()
    d_ba = np.sqrt(_nearest_sq_distances(b, a)).mean()
    return {"to_b": float(d_ab), "from_b": float(d_ba), "sym": float(0.5 * (d_ab + d_ba))}


def f_score(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    """F-score (Knapitsch et al.): harmonic mean of coverage of ``a`` by ``b`` and vice versa."""
    precision = float(np.mean(np.sqrt(_nearest_sq_distances(a, b)) <= threshold))  # a covered by b
    recall = float(np.mean(np.sqrt(_nearest_sq_distances(b, a)) <= threshold))  # b covered by a
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def compute_geometry_metrics(
    method_points: np.ndarray,
    proxy_points: np.ndarray,
    *,
    cap: int = DEFAULT_DOWNSAMPLE_CAP,
    seed: int = 0,
) -> dict[str, Any]:
    """Chamfer + F-score of ``method_points`` against the ``proxy_points`` cloud.

    F-score thresholds are relative to the proxy bounding-box diagonal (0.5% and
    1%) so the metric is comparable across scenes without a known metric scale.
    """
    method = downsample_points(method_points, cap, seed)
    proxy = downsample_points(proxy_points, cap, seed)
    diag = bbox_diagonal(proxy) or 1.0
    cham = chamfer_distances(method, proxy)
    return {
        "chamfer_to_proxy": cham["to_b"],
        "chamfer_from_proxy": cham["from_b"],
        "chamfer_sym": cham["sym"],
        "fscore_05pct": f_score(method, proxy, 0.005 * diag),
        "fscore_1pct": f_score(method, proxy, 0.01 * diag),
        "n_method": int(np.asarray(method_points).shape[0]),
        "n_proxy": int(np.asarray(proxy_points).shape[0]),
        "proxy_diag": diag,
        "downsample_cap": cap,
    }


def load_xyz(path: Path) -> np.ndarray | None:
    """Load a ``.ply`` point cloud as an (N, 3) array via Open3D; ``None`` if unavailable/empty."""
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[lab3.geometry] open3d unavailable ({exc}); cannot load {path}.")
        return None
    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.has_points():
        return None
    return np.asarray(pcd.points, dtype=np.float64)


def write_geometry_metrics_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GEOMETRY_METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in GEOMETRY_METRIC_COLUMNS})


def write_geometry_metrics(
    context: Any,
    staged: dict[str, list[Path]],
    proxy_method: str = "sfm",
    *,
    out_name: str = "geometry_metrics.csv",
) -> Path | None:
    """Compare each method's staged cloud to the proxy cloud; write ``geometry_metrics.csv``.

    Returns the written path, or ``None`` in dry-run / when no proxy is staged.
    """
    if getattr(context, "dry_run", False):
        return None
    run_dir: Path = context.run_dir
    out_path = run_dir / out_name

    proxy_paths = staged.get(proxy_method, [])
    proxy_xyz = _load_first_xyz(proxy_paths)
    rows: list[dict[str, Any]] = []
    if proxy_xyz is None:
        rows.append({"method": "", "note": f"proxy ({proxy_method}) dense cloud missing; metrics skipped"})
        write_geometry_metrics_csv(rows, out_path)
        return out_path

    for method, paths in staged.items():
        if method == proxy_method:
            continue  # never compare the proxy to itself
        xyz = _load_first_xyz(paths)
        if xyz is None:
            rows.append({"method": method, "proxy": proxy_method, "note": "no loadable point cloud; skipped"})
            continue
        metrics = compute_geometry_metrics(xyz, proxy_xyz)
        rows.append({"method": method, "proxy": proxy_method, **metrics, "note": "COLMAP dense used as proxy, not ground truth"})

    write_geometry_metrics_csv(rows, out_path)
    return out_path


def _load_first_xyz(paths: Iterable[Path]) -> np.ndarray | None:
    for path in paths:
        xyz = load_xyz(path)
        if xyz is not None:
            return xyz
    return None
