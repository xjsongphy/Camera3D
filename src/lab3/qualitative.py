"""Qualitative comparison figures for Lab 3 (assignment §5.1).

Builds GT / method-render / error-map panels so reviewers can see *where* each
representation fails, not just a single PSNR number. The figure builder is pure
(numpy arrays in, :class:`matplotlib.figure.Figure` out) and unit-tested; the
orchestration gathers each method's rendered images and is dry-run aware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
import numpy as np

from lab3.metrics import load_image


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-pixel absolute error normalized to [0, 1] for display."""
    diff = np.mean(np.abs(np.asarray(gt, dtype=np.float64) - np.asarray(pred, dtype=np.float64)), axis=-1)
    peak = float(diff.max())
    if peak <= 0:
        return np.zeros_like(diff)
    return diff / peak


def build_comparison_figure(view_name: str, gt: np.ndarray, methods: dict[str, np.ndarray]) -> plt.Figure:
    """One row: GT, then <render, error> per method."""
    n_methods = len(methods)
    n_cols = 1 + 2 * n_methods
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(np.clip(gt, 0, 1))
    axes[0].set_title("GT")
    axes[0].axis("off")

    for idx, (method, image) in enumerate(methods.items(), start=1):
        render_ax = axes[2 * idx - 1]
        error_ax = axes[2 * idx]
        render_ax.imshow(np.clip(image, 0, 1))
        render_ax.set_title(method)
        render_ax.axis("off")
        error_ax.imshow(error_map(gt, image), cmap="magma", vmin=0, vmax=1)
        error_ax.set_title(f"{method} err")
        error_ax.axis("off")

    fig.suptitle(view_name)
    fig.tight_layout()
    return fig


def save_qualitative(
    context: Any,
    eval_cfg: Any,
    prepared_images_dir: Path,
    test_names: list[str],
    method_render_dirs: dict[str, Path],
    eval_size: tuple[int, int] | None = None,
    max_views: int = 3,
) -> list[Path]:
    """Build comparison panels for up to ``max_views`` held-out views."""
    if getattr(context, "dry_run", False):
        return []
    out_dir: Path = context.run_dir / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name in test_names[:max_views]:
        gt_path = prepared_images_dir / name
        if not gt_path.exists():
            continue
        gt = load_image(gt_path, eval_size)
        methods: dict[str, np.ndarray] = {}
        for method, render_dir in method_render_dirs.items():
            render_path = _find_render(render_dir, name)
            if render_path is not None:
                methods[method] = load_image(render_path, eval_size)
        if not methods:
            continue
        fig = build_comparison_figure(name, gt, methods)
        path = out_dir / f"comparison_{Path(name).stem}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def _find_render(render_dir: Path, target_name: str) -> Path | None:
    """Locate a method's render for ``target_name`` by stem (handles index naming)."""
    if not render_dir.is_dir():
        return None
    stem = Path(target_name).stem
    for path in sorted(render_dir.rglob("*.png")):
        if stem in path.stem or path.stem == stem:
            return path
    return None
