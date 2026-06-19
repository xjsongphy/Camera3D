from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from lab3.evaluate import (
    METRIC_COLUMNS,
    EvaluateConfig,
    build_3dgs_metrics_command,
    build_3dgs_render_command,
    build_nerf_eval_command,
    build_nerf_render_command,
    gpu_summary,
    model_size_mb,
    pair_3dgs_renders,
    parse_3dgs_results_json,
    parse_nerf_eval_json,
    write_metrics_csv,
)
from types import SimpleNamespace


def test_build_3dgs_render_command_targets_render_py(tmp_path: Path) -> None:
    repo = tmp_path / "gaussian-splatting"
    cmd = build_3dgs_render_command("python", repo, tmp_path / "model")
    assert cmd[0] == "python"
    assert cmd[1].endswith("render.py")
    assert "-m" in cmd
    assert str(tmp_path / "model") in cmd


def test_build_3dgs_metrics_command_targets_metrics_py(tmp_path: Path) -> None:
    repo = tmp_path / "gaussian-splatting"
    cmd = build_3dgs_metrics_command("python", repo, tmp_path / "model")
    assert cmd[1].endswith("metrics.py")
    assert "-m" in cmd


def test_build_nerf_eval_command_uses_load_config(tmp_path: Path) -> None:
    cmd = build_nerf_eval_command("ns-eval", tmp_path / "config.yml", tmp_path / "out.json")
    assert cmd[0] == "ns-eval"
    assert "--load-config" in cmd
    assert "--output-path" in cmd


def test_build_nerf_render_command_requests_rgb_dataset(tmp_path: Path) -> None:
    cmd = build_nerf_render_command("ns-render", tmp_path / "config.yml", tmp_path / "renders", "test")
    assert cmd[0] == "ns-render"
    assert "dataset" in cmd
    assert "--split" in cmd
    assert "test" in cmd
    assert "rgb" in cmd


def test_pair_3dgs_renders_matches_gt_by_name(tmp_path: Path) -> None:
    renders = tmp_path / "test" / "ours_7000" / "renders"
    gts = tmp_path / "test" / "ours_7000" / "gt"
    renders.mkdir(parents=True)
    gts.mkdir(parents=True)
    (renders / "00000.png").write_bytes(b"x")
    (gts / "00000.png").write_bytes(b"y")
    (renders / "00001.png").write_bytes(b"x")
    # gt for 00001 intentionally missing -> dropped

    pairs = pair_3dgs_renders(tmp_path)

    assert len(pairs) == 1
    assert pairs[0][0].name == "00000.png"


def test_parse_3dgs_results_json_extracts_metrics(tmp_path: Path) -> None:
    payload = {"scene": {"ours_7000": {"PSNR": 28.1, "SSIM": 0.91, "LPIPS": 0.12}}}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = parse_3dgs_results_json(path)

    assert result["psnr"] == pytest.approx(28.1)
    assert result["ssim"] == pytest.approx(0.91)
    assert result["lpips"] == pytest.approx(0.12)


def test_parse_nerf_eval_json_handles_aggregate(tmp_path: Path) -> None:
    payload = {"results": {"psnr": 27.4, "ssim": 0.88, "lpips": 0.15}}
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = parse_nerf_eval_json(path)

    assert result["psnr"] == pytest.approx(27.4)
    assert result["ssim"] == pytest.approx(0.88)


def test_parse_nerf_eval_json_averages_per_image_list(tmp_path: Path) -> None:
    payload = [
        {"psnr": 26.0, "ssim": 0.8, "lpips": 0.2},
        {"psnr": 28.0, "ssim": 0.9, "lpips": 0.1},
    ]
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = parse_nerf_eval_json(path)

    assert result["psnr"] == pytest.approx(27.0)
    assert result["ssim"] == pytest.approx(0.85)


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
            "held_out": "every-8th (train.py --eval)",
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
