from __future__ import annotations

from pathlib import Path

from lab3.visualization import (
    find_viewer_targets,
    view_run_dir,
)


def _seed_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "geometry" / "sfm").mkdir(parents=True)
    (run_dir / "geometry" / "3dgs").mkdir(parents=True)
    (run_dir / "results" / "nerf" / "train" / "nerfacto").mkdir(parents=True)
    (run_dir / "geometry" / "sfm" / "dense.ply").write_bytes(b"ply")
    (run_dir / "geometry" / "3dgs" / "gaussians.ply").write_bytes(b"ply")
    (run_dir / "results" / "nerf" / "train" / "nerfacto" / "config.yml").write_text("a: 1")
    return run_dir


def test_find_viewer_targets_locates_geometry_and_nerf_config(tmp_path: Path) -> None:
    run_dir = _seed_run(tmp_path)

    targets = find_viewer_targets(run_dir, ("sfm", "3dgs", "nerf"))

    assert targets["sfm"] and targets["sfm"][0].name == "dense.ply"
    assert targets["3dgs"] and targets["3dgs"][0].name == "gaussians.ply"
    assert targets["nerf"] and targets["nerf"][0].name == "config.yml"


def test_find_viewer_targets_skips_missing_methods(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # nothing staged

    targets = find_viewer_targets(run_dir, ("sfm", "3dgs", "nerf"))

    assert targets.get("sfm") == []
    assert targets.get("3dgs") == []
    assert targets.get("nerf") == []


def test_find_viewer_targets_prefers_staged_over_raw_results(tmp_path: Path) -> None:
    run_dir = _seed_run(tmp_path)
    # also create a raw results/sfm/dense/fused.ply; staged geometry/sfm wins
    (run_dir / "results" / "sfm" / "dense").mkdir(parents=True)
    (run_dir / "results" / "sfm" / "dense" / "fused.ply").write_bytes(b"raw")

    targets = find_viewer_targets(run_dir, ("sfm",))

    assert targets["sfm"][0] == run_dir / "geometry" / "sfm" / "dense.ply"


def test_build_nerfstudio_viewer_command_uses_load_config(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yml"
    from lab3.reconstruction.base import ViewerTarget

    cmd = ViewerTarget("nerf", "external", cfg, ("--load-config", str(cfg))).command(
        "ns-viewer"
    )
    assert cmd[0] == "ns-viewer"
    assert "--load-config" in cmd
    assert str(cfg) in cmd


def test_view_run_dir_dry_run_does_not_open_gui(tmp_path: Path) -> None:
    run_dir = _seed_run(tmp_path)

    # dry_run must return without launching Open3D / ns-viewer (no GUI, no network)
    view_run_dir(run_dir, ("sfm", "3dgs", "nerf"), dry_run=True)


def test_view_run_dir_without_targets_is_safe(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()

    view_run_dir(run_dir, ("sfm", "3dgs", "nerf"), dry_run=True)
