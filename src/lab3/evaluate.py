"""Backend-agnostic evaluation orchestration and metric bookkeeping."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from lab3.common import Lab3Error
from lab3.reconstruction.base import ReconstructionContext, Reconstructor

METRIC_COLUMNS = [
    "method", "psnr", "ssim", "lpips", "metric_source", "held_out",
    "train_time_sec", "iterations", "gpu_mem_peak_gb", "render_fps",
    "model_size_mb", "gpu", "notes",
]


@dataclass(frozen=True)
class EvaluateConfig:
    enabled: bool = True
    eval_size: tuple[int, int] | None = None
    lpips: bool = True
    native_crosscheck: bool = False


def evaluate_run(
    context: ReconstructionContext,
    eval_cfg: EvaluateConfig,
    reconstructors: Sequence[Reconstructor],
) -> list[dict[str, Any]]:
    """Invoke the same evaluation interface on every registered backend."""
    output_dir = context.run_dir / "results" / "_eval"
    if not context.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for reconstructor in reconstructors
        if (row := reconstructor.evaluate(context, eval_cfg, output_dir)) is not None
    ]
    write_metrics_csv(rows, context.run_dir / "metrics.csv")
    return rows


def metrics_row(
    method: str,
    metrics: dict[str, Any],
    *,
    metric_source: str,
    held_out: str,
    train_time_sec: float | None,
    iterations: int | None,
    gpu_mem_peak_gb: float | None,
    render_fps: float,
    model_size_mb: float,
    notes: str,
) -> dict[str, Any]:
    """Build the repository-wide metrics schema without knowing the backend."""
    return {
        "method": method,
        "psnr": _fmt(metrics.get("psnr")),
        "ssim": _fmt(metrics.get("ssim")),
        "lpips": _fmt(metrics.get("lpips")),
        "metric_source": metric_source,
        "held_out": held_out,
        "train_time_sec": _fmt(train_time_sec),
        "iterations": "" if iterations is None else str(iterations),
        "gpu_mem_peak_gb": _fmt(gpu_mem_peak_gb),
        "render_fps": _fmt(render_fps),
        "model_size_mb": _fmt(model_size_mb),
        "gpu": gpu_summary(),
        "notes": notes,
    }


def geometry_only_row(
    method: str,
    config: Any,
    timings: dict[str, float],
    gpu_peaks: dict[str, float],
    *,
    train_timing_key: str,
    train_peak_key: str,
) -> dict[str, Any]:
    return metrics_row(
        method,
        {"psnr": "N/A", "ssim": "N/A", "lpips": "N/A"},
        metric_source="geometry-only",
        held_out="n/a",
        train_time_sec=timings.get(train_timing_key),
        iterations=config_iterations(config),
        gpu_mem_peak_gb=gpu_peaks.get(train_peak_key),
        render_fps=float("nan"),
        model_size_mb=float("nan"),
        notes="explicit point cloud; RGB novel-view PSNR not applicable",
    ) | {"render_fps": "N/A", "model_size_mb": "N/A"}


def pair_rendered_views(
    images_dir: Path, test_names: Sequence[str], render_dir: Path
) -> list[tuple[Path, Path]]:
    """Pair canonical ground truth and render files without backend assumptions."""
    rendered = [
        path for path in render_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    by_name = {path.name: path for path in rendered}
    by_stem = {path.stem: path for path in rendered}
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for name in test_names:
        ground_truth = images_dir / name
        prediction = by_name.get(name) or by_stem.get(Path(name).stem)
        if not ground_truth.is_file() or prediction is None:
            missing.append(name)
        else:
            pairs.append((ground_truth, prediction))
    if missing:
        raise Lab3Error(f"Missing canonical held-out renders: {missing[:5]}")
    return pairs


def model_size_mb(paths: Iterable[Path]) -> float:
    total = 0
    for path in paths:
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return total / (1024 * 1024)


def render_fps(image_count: int | None, elapsed: float) -> float:
    return image_count / elapsed if image_count and elapsed > 0 else float("nan")


def config_iterations(config: Any, key: str = "iterations") -> int | None:
    value = getattr(config, key, None) if config is not None else None
    return int(value) if value is not None else None


def gpu_summary() -> str:
    try:
        import torch
    except Exception:
        return "n/a"
    if not torch.cuda.is_available():
        return "n/a (CUDA not available)"
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    total = torch.cuda.get_device_properties(index).total_memory / (1024**3)
    return f"{name}, {total:.1f} GB total"


def write_metrics_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if value != value else f"{value:.4f}"
    return str(value)


# Compatibility for callers predating the uniform public name.
_config_iterations = config_iterations
