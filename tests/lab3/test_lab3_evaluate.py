from __future__ import annotations

import csv
from pathlib import Path

import pytest

from lab3.evaluate import (
    METRIC_COLUMNS,
    EvaluateConfig,
    gpu_summary,
    model_size_mb,
    pair_rendered_views,
    write_metrics_csv,
)
from lab3.reconstruction.dgs import _graphdeco_command, _pair_renders
from types import SimpleNamespace


def test_build_3dgs_render_command_targets_render_py(tmp_path: Path) -> None:
    repo = tmp_path / "gaussian-splatting"
    cmd = _graphdeco_command("python", repo, "render.py", ["-m", str(tmp_path / "model")])
    assert cmd[0] == "python"
    assert cmd[1].endswith("reconstruction/dgs.py")
    assert "render.py" in cmd
    assert "-m" in cmd
    assert str(tmp_path / "model") in cmd


def test_pair_3dgs_renders_matches_gt_by_name(tmp_path: Path) -> None:
    renders = tmp_path / "test" / "ours_7000" / "renders"
    gts = tmp_path / "test" / "ours_7000" / "gt"
    renders.mkdir(parents=True)
    gts.mkdir(parents=True)
    (renders / "00000.png").write_bytes(b"x")
    (gts / "00000.png").write_bytes(b"y")
    (renders / "00001.png").write_bytes(b"x")
    # gt for 00001 intentionally missing -> dropped

    pairs = _pair_renders(tmp_path)

    assert len(pairs) == 1
    assert pairs[0][0].name == "00000.png"


def test_pair_rendered_views_uses_canonical_names_and_stems(tmp_path: Path) -> None:
    images = tmp_path / "images"
    renders = tmp_path / "renders" / "test" / "rgb"
    images.mkdir()
    renders.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"gt")
    (images / "b.jpg").write_bytes(b"gt")
    (renders / "a.jpg").write_bytes(b"pred")
    (renders / "b.png").write_bytes(b"pred")

    pairs = pair_rendered_views(images, ("a.jpg", "b.jpg"), renders)

    assert [(gt.name, pred.name) for gt, pred in pairs] == [("a.jpg", "a.jpg"), ("b.jpg", "b.png")]


def test_model_size_mb_sums_file_sizes(tmp_path: Path) -> None:
    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    a.write_bytes(b"\0" * (1024 * 1024))
    b.write_bytes(b"\0" * (512 * 1024))

    size = model_size_mb([a, b])

    assert size == pytest.approx(1.5, abs=1e-3)


def test_model_size_mb_missing_files_are_skipped(tmp_path: Path) -> None:
    real = tmp_path / "a.ply"
    real.write_bytes(b"\0" * 1024)

    size = model_size_mb([real, tmp_path / "missing.ply"])

    assert size == pytest.approx(1024 / (1024 * 1024), abs=1e-6)


def test_write_metrics_csv_has_required_columns(tmp_path: Path) -> None:
    rows = [
        {
            "method": "3dgs",
            "psnr": 28.1,
            "ssim": 0.91,
            "lpips": 0.12,
            "metric_source": "lab3.metrics",
            "held_out": "prepared/test.txt (canonical split)",
            "train_time_sec": 600.0,
            "render_fps": 142.0,
            "model_size_mb": 12.3,
            "gpu": "n/a",
            "notes": "",
        }
    ]
    path = tmp_path / "metrics.csv"
    write_metrics_csv(rows, path)

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        row = next(reader)

    assert "method" in header and "psnr" in header and "lpips" in header
    assert row["method"] == "3dgs"
    assert row["psnr"] == "28.1"


def test_gpu_summary_returns_string_without_raising() -> None:
    summary = gpu_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_evaluate_config_defaults_enable_evaluation() -> None:
    cfg = EvaluateConfig()
    assert cfg.enabled is True


def test_metric_columns_include_iterations_and_gpu_peak() -> None:
    assert "iterations" in METRIC_COLUMNS
    assert "gpu_mem_peak_gb" in METRIC_COLUMNS


def test_config_iterations_reads_dgs_iterations() -> None:
    from lab3.evaluate import _config_iterations

    assert _config_iterations(SimpleNamespace(iterations=7000)) == 7000
    assert _config_iterations(None) is None
    assert _config_iterations(SimpleNamespace(iterations=None)) is None


def test_config_iterations_reads_nerf_max_num_iterations() -> None:
    from lab3.evaluate import _config_iterations

    assert _config_iterations(SimpleNamespace(max_num_iterations=30000), key="max_num_iterations") == 30000


def test_geometry_only_row_carries_iterations_and_peak() -> None:
    from lab3.evaluate import geometry_only_row

    cfg = SimpleNamespace(iterations=None)
    peaks = {"sfm_patch_match_stereo": 3.25}
    row = geometry_only_row(
        "sfm",
        cfg,
        {"sfm_mapper": 12.0},
        peaks,
        train_timing_key="sfm_mapper",
        train_peak_key="sfm_patch_match_stereo",
    )
    assert row["method"] == "sfm"
    assert row["iterations"] == ""  # SfM has no training iterations
    assert row["gpu_mem_peak_gb"] == "3.2500"
    assert row["render_fps"] == "N/A"
