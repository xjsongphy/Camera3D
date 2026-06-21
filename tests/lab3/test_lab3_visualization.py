from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from lab3.reconstruction import create_default_reconstructor
from lab3.reconstruction.base import ViewerTarget
from lab3.visualization import (
    find_viewer_targets,
    view_run_dir,
)


sys.modules.setdefault("yaml", types.SimpleNamespace())


def _seed_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "configs").mkdir(parents=True)
    (run_dir / "geometry" / "sfm").mkdir(parents=True)
    (run_dir / "geometry" / "3dgs").mkdir(parents=True)
    (run_dir / "results" / "3dgs").mkdir(parents=True)
    (run_dir / "results" / "nerf" / "train" / "nerfacto").mkdir(parents=True)
    (run_dir / "results" / "neus" / "train" / "neus-facto").mkdir(parents=True)
    (run_dir / "geometry" / "sfm" / "dense.ply").write_bytes(b"ply")
    (run_dir / "geometry" / "3dgs" / "gaussians.ply").write_bytes(b"ply")
    (run_dir / "results" / "nerf" / "train" / "nerfacto" / "config.yml").write_text("a: 1")
    (run_dir / "results" / "neus" / "train" / "neus-facto" / "config.yml").write_text("a: 1")
    (run_dir / "configs" / "run_config.json").write_text(
        json.dumps(
            {
                "input_dir": "input",
                "scene_name": "scene",
                "reconstruction": {
                    "3dgs": {"repo_dir": str(tmp_path / "gaussian-splatting")}
                },
            }
        ),
        encoding="utf-8",
    )
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

    cmd = ViewerTarget("nerf", "nerfstudio", cfg, ("--load-config", str(cfg))).command(
        "ns-viewer"
    )
    assert cmd[0] == "ns-viewer"
    assert "--load-config" in cmd
    assert str(cfg) in cmd


def test_3dgs_reconstructor_exposes_sibr_viewer_target(tmp_path: Path) -> None:
    run_dir = _seed_run(tmp_path)

    reconstructor = create_default_reconstructor("3dgs")
    targets = reconstructor.viewer_targets(run_dir)

    assert any(target.kind == "geometry" and target.path.name == "gaussians.ply" for target in targets)
    assert any(target.kind == "sibr" and target.path == run_dir / "results" / "3dgs" for target in targets)


def test_nerf_and_neus_reconstructors_expose_nerfstudio_targets(tmp_path: Path) -> None:
    run_dir = _seed_run(tmp_path)

    nerf_targets = create_default_reconstructor("nerf").viewer_targets(run_dir)
    neus_targets = create_default_reconstructor("neus").viewer_targets(run_dir)

    assert nerf_targets and nerf_targets[0].kind == "nerfstudio"
    assert neus_targets and any(target.kind == "nerfstudio" for target in neus_targets)


def test_view_run_dir_dry_run_does_not_open_gui(tmp_path: Path) -> None:
    run_dir = _seed_run(tmp_path)

    # dry_run must return without launching Open3D / ns-viewer (no GUI, no network)
    view_run_dir(run_dir, ("sfm", "3dgs", "nerf"), dry_run=True)


def test_view_run_dir_without_targets_is_safe(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()

    view_run_dir(run_dir, ("sfm", "3dgs", "nerf"), dry_run=True)


def test_view_run_dir_dry_run_prints_sibr_build_command_when_binary_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    run_dir = _seed_run(tmp_path)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    view_run_dir(run_dir, ("3dgs",), dry_run=True)

    output = capsys.readouterr().out
    assert "build_sibr_viewer.sh" in output
    assert "SIBR_gaussianViewer_app" in output


def test_view_run_dir_dry_run_prints_sibr_launch_command_when_binary_exists(
    tmp_path: Path,
    capsys,
) -> None:
    run_dir = _seed_run(tmp_path)
    viewer_bin = tmp_path / "gaussian-splatting" / "SIBR_viewers" / "install" / "bin" / "SIBR_gaussianViewer_app"
    viewer_bin.parent.mkdir(parents=True, exist_ok=True)
    viewer_bin.write_text("", encoding="utf-8")

    view_run_dir(run_dir, ("3dgs",), dry_run=True)

    output = capsys.readouterr().out
    assert "SIBR_gaussianViewer_app" in output
    assert f"-m {run_dir / 'results' / '3dgs'}" in output
