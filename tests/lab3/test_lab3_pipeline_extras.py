from __future__ import annotations

import csv
from pathlib import Path

import pytest

from lab3.pipeline import (
    Lab3PipelineConfig,
    _build_reconstructors,
    config_from_dict,
    run_pipeline,
)
from lab3.reconstruction import DGSConfig, default_reconstruction_configs


def test_config_from_dict_parses_new_fields() -> None:
    cfg = config_from_dict(
        {
            "input_dir": "data/scene",
            "scene_name": "desk",
            "share_poses": False,
            "evaluate": False,
            "eval_size": [800, 800],
            "lpips": False,
            "reconstruction": {
                "3dgs": {
                    "repo_dir": "external/gaussian-splatting",
                    "eval_split": False,
                    "save_iterations": [4000, 8000],
                },
                "nerf": {
                    "colmap_model": "prepared/sparse/0",
                    "save_iterations": [5000, 10000],
                    "train_num_rays_per_batch": 6144,
                },
                "neus": {
                    "train_num_rays_per_batch": 3072,
                    "train_num_images_to_sample_from": 256,
                    "train_num_times_to_repeat_images": 100,
                    "export_mesh": False,
                },
            },
        }
    )

    assert cfg.share_poses is False
    assert cfg.evaluate is False
    assert cfg.lpips is False
    assert cfg.eval_size == (800, 800)
    assert cfg.dgs.eval_split is False
    assert cfg.dgs.save_iterations == (4000, 8000)
    assert cfg.nerf.colmap_model == Path("prepared/sparse/0")
    assert cfg.nerf.save_iterations == (5000, 10000)
    assert cfg.nerf.train_num_rays_per_batch == 6144
    assert cfg.neus.train_num_rays_per_batch == 3072
    assert cfg.neus.train_num_images_to_sample_from == 256
    assert cfg.neus.train_num_times_to_repeat_images == 100
    assert cfg.neus.export_mesh is False


def test_resolve_shared_configs_injects_when_sharing(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "input_dir": str(tmp_path),
            "methods": ["sfm", "3dgs", "nerf", "neus"],
            "share_poses": True,
        }
    )
    shared = tmp_path / "prepared"

    reconstructors = _build_reconstructors(cfg, shared)
    configs = {reconstructor.name: reconstructor.config for reconstructor in reconstructors}

    assert configs["3dgs"].colmap_source == shared
    assert configs["nerf"].colmap_model == shared / "sparse" / "0"
    assert configs["neus"].colmap_model == shared / "sparse" / "0"


def test_resolve_shared_configs_leaves_them_empty_when_not_sharing(tmp_path: Path) -> None:
    cfg = config_from_dict(
        {
            "input_dir": str(tmp_path),
            "methods": ["sfm", "3dgs", "nerf"],
            "share_poses": False,
        }
    )

    reconstructors = _build_reconstructors(cfg, tmp_path / "prepared")
    configs = {reconstructor.name: reconstructor.config for reconstructor in reconstructors}

    assert configs["3dgs"].colmap_source is None
    assert configs["nerf"].colmap_model is None


def test_order_methods_runs_sfm_first_when_sharing() -> None:
    cfg = config_from_dict(
        {"input_dir": "data/scene", "methods": ["3dgs", "nerf", "sfm"], "share_poses": True}
    )
    ordered = _build_reconstructors(cfg, Path("prepared"))
    assert ordered[0].name == "sfm"
    assert {reconstructor.name for reconstructor in ordered} == {"sfm", "3dgs", "nerf"}


def test_order_methods_preserves_order_when_not_sharing() -> None:
    cfg = config_from_dict(
        {"input_dir": "data/scene", "methods": ["3dgs", "nerf", "sfm"], "share_poses": False}
    )
    ordered = _build_reconstructors(cfg, Path("prepared"))
    assert [reconstructor.name for reconstructor in ordered] == ["3dgs", "nerf", "sfm"]


def test_shared_poses_fails_fast_without_sfm() -> None:
    cfg = config_from_dict(
        {"input_dir": "data/scene", "methods": ["nerf", "neus"], "share_poses": True}
    )

    with pytest.raises(Exception, match="requires a pose provider"):
        _build_reconstructors(cfg, Path("prepared"))


def test_neus_standalone_requires_explicit_colmap_model() -> None:
    cfg = config_from_dict(
        {"input_dir": "data/scene", "methods": ["neus"], "share_poses": False}
    )

    with pytest.raises(Exception, match="cannot estimate poses"):
        _build_reconstructors(cfg, Path("prepared"))


def test_dry_run_pipeline_writes_configs_and_metrics_skeleton(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "img_000001.jpg").write_bytes(b"x")
    (input_dir / "img_000002.jpg").write_bytes(b"x")

    repo = tmp_path / "gaussian-splatting"
    repo.mkdir()

    cfg = Lab3PipelineConfig(
        input_dir=input_dir,
        scene_name="smoke",
        output_root=tmp_path / "outputs",
        methods=("sfm", "3dgs", "nerf"),
        share_poses=True,
        evaluate=True,
        geometry=True,
        qualitative=True,
        dry_run=True,
        timestamp="dry",
        reconstruction={
            **default_reconstruction_configs(),
            "3dgs": DGSConfig(repo_dir=repo),
        },
    )

    run_dir = run_pipeline(cfg)

    assert (run_dir / "configs" / "run_config.json").exists()
    assert (run_dir / "configs" / "prepared_dataset.json").exists()
    assert (run_dir / "timings.json").exists()
    metrics_csv = run_dir / "metrics.csv"
    assert metrics_csv.exists()
    with metrics_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "method" in reader.fieldnames
        assert "psnr" in reader.fieldnames
        assert "render_fps" in reader.fieldnames
        # Efficiency columns required by the assignment (§5.2): iterations + GPU peak.
        assert "iterations" in reader.fieldnames
        assert "gpu_mem_peak_gb" in reader.fieldnames
        rows = list(reader)
    method_names = {row["method"] for row in rows}
    assert {"3dgs", "nerf", "sfm"} <= method_names
