from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from lab3.reconstruction import DGSConfig, DGSReconstructor, SfMConfig, SfMReconstructor

import numpy as np
import pytest

from lab3.geometry import (
    bbox_diagonal,
    chamfer_distances,
    compute_geometry_metrics,
    copy_geometry,
    downsample_points,
    f_score,
    find_3dgs_pointcloud,
    find_sfm_dense,
    stage_geometry,
    write_geometry_metrics,
    write_geometry_metrics_csv,
)


def test_find_3dgs_pointcloud_picks_latest_iteration(tmp_path: Path) -> None:
    model = tmp_path / "3dgs"
    (model / "point_cloud" / "iteration_3000").mkdir(parents=True)
    (model / "point_cloud" / "iteration_7000").mkdir(parents=True)
    (model / "point_cloud" / "iteration_3000" / "point_cloud.ply").write_bytes(b"old")
    (model / "point_cloud" / "iteration_7000" / "point_cloud.ply").write_bytes(b"new")

    found = find_3dgs_pointcloud(model)

    assert found is not None
    assert found.read_bytes() == b"new"


def test_find_sfm_dense_locates_fused_ply(tmp_path: Path) -> None:
    sfm = tmp_path / "sfm"
    (sfm / "dense").mkdir(parents=True)
    (sfm / "dense" / "fused.ply").write_bytes(b"xyz")

    found = find_sfm_dense(sfm)

    assert found is not None
    assert found.name == "fused.ply"


def test_copy_geometry_copies_file(tmp_path: Path) -> None:
    src = tmp_path / "src.ply"
    src.write_bytes(b"data")
    dest_dir = tmp_path / "out"

    copied = copy_geometry(src, dest_dir, "gaussians.ply")

    assert copied.exists()
    assert copied.read_bytes() == b"data"
    assert copied.name == "gaussians.ply"


def test_stage_geometry_collects_3dgs_and_sfm(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    results = run_dir / "results"
    # 3dgs model + point cloud
    pc_dir = results / "3dgs" / "point_cloud" / "iteration_7000"
    pc_dir.mkdir(parents=True)
    (pc_dir / "point_cloud.ply").write_bytes(b"gs")
    # sfm dense
    (results / "sfm" / "dense").mkdir(parents=True)
    (results / "sfm" / "dense" / "fused.ply").write_bytes(b"sfm")

    context = SimpleNamespace(run_dir=run_dir, dry_run=False)
    reconstructors = [DGSReconstructor(DGSConfig(repo_dir=None)), SfMReconstructor(SfMConfig(dense=True))]

    staged = stage_geometry(context, reconstructors)

    assert (run_dir / "geometry" / "3dgs" / "gaussians.ply").exists()
    assert (run_dir / "geometry" / "sfm" / "dense.ply").exists()
    assert "3dgs" in staged and "sfm" in staged


def test_stage_geometry_dry_run_skips_copy(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    context = SimpleNamespace(run_dir=run_dir, dry_run=True)
    reconstructors = [DGSReconstructor(DGSConfig(repo_dir=None))]

    staged = stage_geometry(context, reconstructors)

    assert not (run_dir / "geometry").exists() or staged == {"3dgs": []}


# --------------------------------------------------------------------------- #
# Quantitative geometry metrics (assignment §5.3: Chamfer / F-score)           #
# --------------------------------------------------------------------------- #
def test_bbox_diagonal_of_unit_cube_corners() -> None:
    corners = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    assert bbox_diagonal(corners) == pytest.approx(float(np.sqrt(3.0)))


def test_chamfer_identical_clouds_is_zero() -> None:
    pts = np.random.default_rng(0).random((50, 3))
    c = chamfer_distances(pts, pts)
    assert c["sym"] == 0.0
    assert c["to_b"] == 0.0 and c["from_b"] == 0.0


def test_chamfer_translated_cloud_matches_offset() -> None:
    a = np.zeros((1, 3))
    b = np.array([[3.0, 0.0, 0.0]])
    c = chamfer_distances(a, b)
    assert c["to_b"] == 3.0
    assert c["from_b"] == 3.0
    assert c["sym"] == 3.0


def test_f_score_identical_is_one() -> None:
    pts = np.random.default_rng(1).random((40, 3))
    assert f_score(pts, pts, threshold=0.0) == 1.0


def test_f_score_far_apart_is_zero() -> None:
    a = np.zeros((10, 3))
    b = np.full((10, 3), 100.0)
    assert f_score(a, b, threshold=0.5) == 0.0


def test_downsample_caps_and_is_deterministic() -> None:
    pts = np.random.default_rng(2).random((500, 3))
    small = downsample_points(pts, cap=50, seed=0)
    assert small.shape == (50, 3)
    again = downsample_points(pts, cap=50, seed=0)
    assert np.array_equal(small, again)


def test_compute_geometry_metrics_returns_expected_keys() -> None:
    rng = np.random.default_rng(3)
    proxy = rng.random((300, 3))
    method = proxy + rng.normal(0, 0.01, proxy.shape)  # close to proxy
    result = compute_geometry_metrics(method, proxy)
    assert {"chamfer_sym", "fscore_05pct", "fscore_1pct", "n_method", "n_proxy"} <= set(result)
    assert result["chamfer_sym"] >= 0.0
    assert 0.0 <= result["fscore_05pct"] <= 1.0
    assert result["n_method"] == 300 and result["n_proxy"] == 300


def test_compute_geometry_metrics_identical_is_perfect() -> None:
    proxy = np.random.default_rng(4).random((200, 3))
    result = compute_geometry_metrics(proxy, proxy)
    assert result["chamfer_sym"] == 0.0
    assert result["fscore_05pct"] == 1.0


def test_write_geometry_metrics_csv_has_columns(tmp_path: Path) -> None:
    rows = [
        {
            "method": "3dgs",
            "proxy": "sfm",
            "chamfer_sym": 0.0123,
            "fscore_05pct": 0.41,
            "n_method": 4096,
            "n_proxy": 4096,
        }
    ]
    path = tmp_path / "geometry_metrics.csv"
    write_geometry_metrics_csv(rows, path)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "chamfer_sym" in reader.fieldnames and "fscore_05pct" in reader.fieldnames
        row = next(reader)
    assert row["method"] == "3dgs" and row["proxy"] == "sfm"


def test_write_geometry_metrics_skips_in_dry_run(tmp_path: Path) -> None:
    context = SimpleNamespace(run_dir=tmp_path, dry_run=True)
    path = write_geometry_metrics(context, {"3dgs": []})
    assert path is None
