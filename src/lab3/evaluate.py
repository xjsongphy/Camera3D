"""Evaluation stage for Lab 3.

Renders each method's held-out views and scores them with the unified
:mod:`lab3.metrics` implementation (same PSNR/SSIM/LPIPS, same resolution) so
methods are compared fairly. Also records render FPS, model file size and GPU
info, and writes the assignment-required ``metrics.csv``.

The render/metric commands shell out to the official 3DGS (``render.py`` /
``metrics.py``) and nerfstudio (``ns-eval`` / ``ns-render``) CLIs, mirroring the
rest of Lab 3's design: everything is dry-run aware and the pure helpers are
unit-tested without those tools installed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from lab3.common import Lab3Error, run_cmd, timed_block
from lab3.metrics import compute_image_metrics
from lab3.reconstruction.base import ReconstructionContext

METRIC_COLUMNS = [
    "method",
    "psnr",
    "ssim",
    "lpips",
    "metric_source",
    "held_out",
    "train_time_sec",
    "iterations",
    "gpu_mem_peak_gb",
    "render_fps",
    "model_size_mb",
    "gpu",
    "notes",
]

# Stage label whose GPU-memory peak represents each method's training/MVS peak.
TRAIN_PEAK_KEY = {"3dgs": "3dgs_train", "nerf": "nerf_train", "sfm": "sfm_patch_match_stereo"}


@dataclass(frozen=True)
class EvaluateConfig:
    enabled: bool = True
    eval_size: tuple[int, int] | None = None  # (height, width); None = native
    lpips: bool = True
    rgb_methods: tuple[str, ...] = ("3dgs", "nerf")


# --------------------------------------------------------------------------- #
# Command builders (pure)                                                      #
# --------------------------------------------------------------------------- #
def build_3dgs_render_command(python_bin: str, repo_dir: Path, model_dir: Path) -> list[str]:
    return [python_bin, str(repo_dir / "render.py"), "-m", str(model_dir)]


def build_3dgs_metrics_command(python_bin: str, repo_dir: Path, model_dir: Path) -> list[str]:
    return [python_bin, str(repo_dir / "metrics.py"), "-m", str(model_dir)]


def build_nerf_eval_command(eval_bin: str, config_path: Path, out_json: Path) -> list[str]:
    return [eval_bin, "--load-config", str(config_path), "--output-path", str(out_json)]


def build_nerf_render_command(
    render_bin: str, config_path: Path, out_dir: Path, split: str = "test"
) -> list[str]:
    return [
        render_bin,
        "dataset",
        "--load-config",
        str(config_path),
        "--output-path",
        str(out_dir),
        "--split-mode",
        split,
        "--rendered-output-names",
        "rgb",
    ]


# --------------------------------------------------------------------------- #
# Output parsers (pure)                                                        #
# --------------------------------------------------------------------------- #
def pair_3dgs_renders(model_dir: Path) -> list[tuple[Path, Path]]:
    """Match 3DGS ``test/ours_*/renders`` to its ``gt`` copies by filename."""
    renders_root = _find_latest(model_dir / "test", "ours_*")
    if renders_root is None:
        return []
    renders_dir = renders_root / "renders"
    gt_dir = renders_root / "gt"
    pairs: list[tuple[Path, Path]] = []
    if renders_dir.is_dir() and gt_dir.is_dir():
        gt_names = {p.name for p in gt_dir.iterdir()}
        for render in sorted(renders_dir.iterdir()):
            gt = gt_dir / render.name
            if render.name in gt_names:
                pairs.append((gt, render))
    return pairs


def parse_3dgs_results_json(path: Path) -> dict[str, float]:
    """Extract PSNR/SSIM/LPIPS from 3DGS ``metrics.py`` ``results.json``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics: dict[str, float] = {}
    for scene in data.values():
        if not isinstance(scene, dict):
            continue
        for method_values in scene.values():
            if isinstance(method_values, dict) and "PSNR" in method_values:
                metrics = _normalize_metric_keys(method_values)
                break
        if metrics:
            break
    return metrics


def parse_nerf_eval_json(path: Path) -> dict[str, float]:
    """Extract PSNR/SSIM/LPIPS from ``ns-eval`` output (aggregate or per-image list)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        per_image = [_normalize_metric_keys(item) for item in data if isinstance(item, dict)]
        return _average_metrics(per_image)
    if isinstance(data, dict):
        results = data.get("results", data)
        if isinstance(results, list):
            per_image = [_normalize_metric_keys(item) for item in results if isinstance(item, dict)]
            return _average_metrics(per_image)
        if isinstance(results, dict):
            return _normalize_metric_keys(results)
    return {}


# --------------------------------------------------------------------------- #
# Resource helpers (pure / side-effecting but safe)                           #
# --------------------------------------------------------------------------- #
def model_size_mb(paths: Iterable[Path]) -> float:
    total = 0
    for path in paths:
        if path.exists():
            total += path.stat().st_size
    return total / (1024 * 1024)


def gpu_summary() -> str:
    """Best-effort GPU name + total VRAM via torch; ``"n/a"`` if unavailable."""
    try:
        import torch  # local import; torch is an optional project dependency
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def evaluate_run(
    context: ReconstructionContext,
    prepared_root: Path,
    test_list: Path,
    eval_cfg: EvaluateConfig,
    reconstructor_configs: dict[str, Any],
    timings: dict[str, float],
    gpu_peaks: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Render + score every RGB method; write ``metrics.csv``. Dry-run aware."""
    rows: list[dict[str, Any]] = []
    held_out_names = _read_lines(test_list)
    eval_dir = context.run_dir / "results" / "_eval"
    if not context.dry_run:
        eval_dir.mkdir(parents=True, exist_ok=True)
    gpu_peaks = gpu_peaks if gpu_peaks is not None else {}

    for method in eval_cfg.rgb_methods:
        if method == "3dgs":
            rows.append(
                _evaluate_3dgs(
                    context, eval_cfg, reconstructor_configs.get("3dgs"), timings, gpu_peaks, eval_dir
                )
            )
        elif method == "nerf":
            rows.append(
                _evaluate_nerf(
                    context, eval_cfg, reconstructor_configs.get("nerf"), timings, gpu_peaks, eval_dir
                )
            )

    # SfM is geometry-only for RGB metrics; include a row noting N/A.
    if "sfm" in reconstructor_configs:
        rows.append(_geometry_only_row("sfm", reconstructor_configs["sfm"], timings, gpu_peaks))

    csv_path = context.run_dir / "metrics.csv"
    write_metrics_csv(rows, csv_path)
    return rows


def _evaluate_3dgs(
    context: ReconstructionContext,
    eval_cfg: EvaluateConfig,
    dgs_cfg: Any,
    timings: dict[str, float],
    gpu_peaks: dict[str, float],
    eval_dir: Path,
) -> dict[str, Any]:
    model_dir = context.run_dir / "results" / "3dgs"  # where train.py -m wrote
    repo_dir = getattr(dgs_cfg, "repo_dir", None) if dgs_cfg else None
    python_bin = getattr(dgs_cfg, "python_bin", "python") if dgs_cfg else "python"
    logs = context.run_dir / "logs"

    render_cmd = build_3dgs_render_command(python_bin, repo_dir or Path("."), model_dir)
    with timed_block("3dgs_render", timings):
        run_cmd(
            render_cmd,
            dry_run=context.dry_run,
            cwd=repo_dir,
            log_path=logs / "3dgs_render.log" if logs else None,
        )

    pairs = [] if context.dry_run else pair_3dgs_renders(model_dir)
    gt_paths = [g for g, _ in pairs]
    pred_paths = [r for _, r in pairs]
    metrics = (
        compute_image_metrics(gt_paths, pred_paths, eval_size=eval_cfg.eval_size)
        if pairs
        else {"psnr": float("nan"), "ssim": float("nan"), "lpips": None, "n": 0}
    )
    render_elapsed = timings.get("3dgs_render", 0.0)
    fps = _fps(metrics["n"], render_elapsed)
    model_size = model_size_mb([model_dir / "point_cloud"]) if not context.dry_run else float("nan")

    # Native cross-check via metrics.py (writes results.json under model_dir/test).
    if repo_dir is not None and not context.dry_run:
        with timed_block("3dgs_native_metrics", timings):
            run_cmd(
                build_3dgs_metrics_command(python_bin, repo_dir, model_dir),
                dry_run=context.dry_run,
                cwd=repo_dir,
                log_path=logs / "3dgs_metrics.log" if logs else None,
            )

    return _row(
        "3dgs",
        metrics,
        metric_source="lab3.metrics",
        held_out="every-8th (train.py --eval)",
        train_time_sec=timings.get("3dgs_train"),
        iterations=_config_iterations(dgs_cfg),
        gpu_mem_peak_gb=gpu_peaks.get(TRAIN_PEAK_KEY["3dgs"]),
        render_fps=fps,
        model_size_mb=model_size,
        gpu=gpu_summary(),
        notes="3dgs native results.json written under results/3dgs/test for cross-check; "
        "render_fps timed over render.py (includes disk I/O)",
    )


def _evaluate_nerf(
    context: ReconstructionContext,
    eval_cfg: EvaluateConfig,
    nerf_cfg: Any,
    timings: dict[str, float],
    gpu_peaks: dict[str, float],
    eval_dir: Path,
) -> dict[str, Any]:
    train_bin = getattr(nerf_cfg, "train_bin", "ns-train") if nerf_cfg else "ns-train"
    config_path = _find_nerf_config(context.run_dir / "results" / "nerf" / "train")
    out_json = eval_dir / "nerf_eval.json"
    logs = context.run_dir / "logs"

    # nerfstudio's ns-eval computes PSNR/SSIM/LPIPS on its native eval split.
    eval_bin = train_bin.replace("ns-train", "ns-eval") if train_bin == "ns-train" else "ns-eval"
    eval_cmd = build_nerf_eval_command(eval_bin, config_path or Path("config.yml"), out_json)
    with timed_block("nerf_eval", timings):
        run_cmd(
            eval_cmd,
            dry_run=context.dry_run,
            log_path=logs / "nerf_eval.log" if logs else None,
        )

    # Render the eval split so qualitative panels can include nerfstudio.
    render_dir = context.run_dir / "results" / "nerf" / "renders"
    render_bin = train_bin.replace("ns-train", "ns-render") if train_bin == "ns-train" else "ns-render"
    render_cmd = build_nerf_render_command(
        render_bin, config_path or Path("config.yml"), render_dir, "test"
    )
    with timed_block("nerf_render", timings):
        run_cmd(
            render_cmd,
            dry_run=context.dry_run,
            log_path=logs / "nerf_render.log" if logs else None,
        )

    metrics = (
        parse_nerf_eval_json(out_json)
        if (not context.dry_run and out_json.exists())
        else {"psnr": float("nan"), "ssim": float("nan"), "lpips": None}
    )
    render_elapsed = timings.get("nerf_eval", 0.0)
    # ns-eval timing includes eval-image rendering; treat n as unknown -> no fps.
    fps = _fps(metrics.get("n"), render_elapsed) if metrics.get("n") else float("nan")
    model_size = model_size_mb(_nerf_model_files(context.run_dir / "results" / "nerf")) if not context.dry_run else float("nan")

    return _row(
        "nerf",
        metrics,
        metric_source="nerfstudio ns-eval",
        held_out="nerfstudio native eval split",
        train_time_sec=timings.get("nerf_train"),
        iterations=_config_iterations(nerf_cfg, key="max_num_iterations"),
        gpu_mem_peak_gb=gpu_peaks.get(TRAIN_PEAK_KEY["nerf"]),
        render_fps=fps,
        model_size_mb=model_size,
        gpu=gpu_summary(),
        notes="held-out split differs from 3dgs; see report discussion",
    )


def _geometry_only_row(
    method: str, cfg: Any, timings: dict[str, float], gpu_peaks: dict[str, float]
) -> dict[str, Any]:
    train_key = f"{method}_mapper"
    iterations = _config_iterations(cfg)
    return {
        "method": method,
        "psnr": "N/A",
        "ssim": "N/A",
        "lpips": "N/A",
        "metric_source": "geometry-only",
        "held_out": "n/a",
        "train_time_sec": _fmt(timings.get(train_key)),
        "iterations": "" if iterations is None else str(iterations),
        "gpu_mem_peak_gb": _fmt(gpu_peaks.get(TRAIN_PEAK_KEY.get(method, ""))),
        "render_fps": "N/A",
        "model_size_mb": "N/A",
        "gpu": gpu_summary(),
        "notes": "explicit point cloud; RGB novel-view PSNR not applicable",
    }


def _row(
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
    gpu: str,
    notes: str,
) -> dict[str, Any]:
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
        "gpu": gpu,
        "notes": notes,
    }


def _config_iterations(cfg: Any, key: str = "iterations") -> int | None:
    """Configured training iteration cap for a method (``None`` when N/A or unset)."""
    if cfg is None:
        return None
    value = getattr(cfg, key, None)
    return int(value) if value is not None else None


# --------------------------------------------------------------------------- #
# Small utilities                                                              #
# --------------------------------------------------------------------------- #
def _find_latest(root: Path, pattern: str) -> Path | None:
    if not root.is_dir():
        return None
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else None


def _find_nerf_config(train_dir: Path) -> Path | None:
    if not train_dir.exists():
        return None
    matches = sorted(train_dir.rglob("config.yml"))
    return matches[-1] if matches else None


def _nerf_model_files(method_dir: Path) -> list[Path]:
    files: list[Path] = []
    for suffix in ("*.pt", "*.ckpt", "*.safetensors"):
        files.extend(method_dir.rglob(suffix))
    return files


def _normalize_metric_keys(values: dict[str, Any]) -> dict[str, float]:
    lowered = {str(k).lower(): v for k, v in values.items()}
    out: dict[str, float] = {}
    for key in ("psnr", "ssim", "lpips"):
        if key in lowered and lowered[key] is not None:
            out[key] = float(lowered[key])
    return out


def _average_metrics(per_image: list[dict[str, float]]) -> dict[str, float]:
    if not per_image:
        return {}
    keys = set().union(*(m.keys() for m in per_image))
    return {key: float(sum(m[key] for m in per_image if key in m) / len(per_image)) for key in keys}


def _fps(n: int | None, elapsed: float) -> float:
    if not n or elapsed <= 0:
        return float("nan")
    return n / elapsed


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return f"{value:.4f}"
    return str(value)
