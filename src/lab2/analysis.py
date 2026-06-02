from __future__ import annotations

from io import BytesIO
from pathlib import Path
import csv

import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch

from .OpticalSGD import DecoderType, OpticalSGDOptimizer


def decoder_label(optimizer: OpticalSGDOptimizer) -> str:
    label = optimizer.config.decoder_type.value if optimizer.config.decoder_type else "none"
    if optimizer.config.decoder_type == DecoderType.ZNCC_NN and optimizer.config.use_projector_response_curve:
        label = f"{label}_response"
    return label


def plot_patterns(optimizer: OpticalSGDOptimizer) -> Figure:
    if optimizer.patterns is None:
        raise RuntimeError("No patterns to plot")

    k, _ = optimizer.patterns.shape
    hp = optimizer.config.projector_height
    patterns_np = optimizer.patterns.detach().cpu().numpy()
    patterns_2d = np.tile(patterns_np[:, None, :], (1, hp, 1))

    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axes = [axes]

    for idx in range(k):
        im = axes[idx].imshow(
            patterns_2d[idx],
            cmap="gray",
            vmin=optimizer.config.min_value,
            vmax=optimizer.config.max_value,
        )
        axes[idx].set_title(f"Pattern {idx + 1}")
        axes[idx].axis("off")
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def plot_frequency_spectrum(optimizer: OpticalSGDOptimizer) -> Figure:
    if optimizer.patterns is None:
        raise RuntimeError("No patterns to analyze")

    k = optimizer.patterns.shape[0]
    patterns_np = optimizer.patterns.detach().cpu().numpy()
    fig, axes = plt.subplots(k, 1, figsize=(10, 2 * k))
    if k == 1:
        axes = [axes]

    freq_bins = np.fft.rfftfreq(optimizer.patterns.shape[1], d=1.0 / optimizer.patterns.shape[1])

    for idx in range(k):
        pattern_fft = np.abs(np.fft.rfft(patterns_np[idx]))
        axes[idx].plot(freq_bins, pattern_fft, linewidth=1.5)
        axes[idx].set_xlabel("Frequency")
        axes[idx].set_ylabel("Magnitude")
        axes[idx].set_title(f"Pattern {idx + 1} Spectrum")
        axes[idx].grid(True, alpha=0.3)

        if optimizer.config.max_frequency is not None:
            axes[idx].axvline(
                optimizer.config.max_frequency,
                color="r",
                linestyle="--",
                linewidth=1.5,
                label=f"Frequency limit = {optimizer.config.max_frequency:.2f}",
            )
            axes[idx].legend()

    plt.tight_layout()
    return fig


def plot_loss_curve(optimizer: OpticalSGDOptimizer) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    if optimizer.loss_history:
        ax.plot(range(1, len(optimizer.loss_history) + 1), optimizer.loss_history, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss ({decoder_label(optimizer)})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_projector_response_curve(optimizer: OpticalSGDOptimizer) -> Figure | None:
    if optimizer.decoder is None or not hasattr(optimizer.decoder, "sample_projector_response_curve"):
        return None

    curve = optimizer.decoder.sample_projector_response_curve()
    if curve is None:
        return None

    x, y = curve
    x_np = x.numpy()
    y_np = y.numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1.0, label="Identity")
    ax.plot(x_np, y_np, linewidth=2.0, color="tab:blue", label="Learned response")
    ax.set_xlabel("Input intensity")
    ax.set_ylabel("Output intensity")
    ax.set_title("Projector Response Curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def export_raw_histories(optimizer: OpticalSGDOptimizer, output_dir: str | Path | None = None) -> None:
    if not optimizer.pattern_history:
        return

    target_dir = Path(output_dir or optimizer.config.output_dir or ".")
    target_dir.mkdir(parents=True, exist_ok=True)

    pattern_tensor = torch.stack(optimizer.pattern_history, dim=0)
    torch.save(
        {
            "iterations": optimizer.history_iterations,
            "patterns": pattern_tensor,
            "loss_iterations": optimizer.history_iterations[1:],
            "loss_history": optimizer.loss_history,
        },
        target_dir / "pattern_history.pt",
    )

    if optimizer.decoder_param_history:
        param_bundle: dict[str, object] = {"iterations": optimizer.history_iterations}
        for name, history in optimizer.decoder_param_history.items():
            param_bundle[name] = torch.stack(history, dim=0)
        torch.save(param_bundle, target_dir / "decoder_parameter_history.pt")


def export_history_summaries(optimizer: OpticalSGDOptimizer, output_dir: str | Path | None = None) -> None:
    if not optimizer.pattern_history:
        return

    target_dir = Path(output_dir or optimizer.config.output_dir or ".")
    target_dir.mkdir(parents=True, exist_ok=True)

    pattern_tensor = torch.stack(optimizer.pattern_history, dim=0)
    summary_rows: list[dict[str, float | int]] = []
    for idx, iteration in enumerate(optimizer.history_iterations):
        snapshot = pattern_tensor[idx].numpy()
        for pattern_idx in range(snapshot.shape[0]):
            values = snapshot[pattern_idx]
            summary_rows.append(
                {
                    "iteration": iteration,
                    "pattern_index": pattern_idx,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "l2_norm": float(np.linalg.norm(values)),
                }
            )
    with open(target_dir / "pattern_history_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["iteration", "pattern_index", "mean", "std", "min", "max", "l2_norm"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    if optimizer.decoder_param_history:
        param_rows: list[dict[str, float | int | str]] = []
        for name, history in optimizer.decoder_param_history.items():
            stacked = torch.stack(history, dim=0)
            flat = stacked.reshape(stacked.shape[0], -1).numpy()
            for idx, iteration in enumerate(optimizer.history_iterations):
                values = flat[idx]
                param_rows.append(
                    {
                        "iteration": iteration,
                        "parameter": name,
                        "numel": int(values.size),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "l2_norm": float(np.linalg.norm(values)),
                    }
                )
        with open(target_dir / "decoder_parameter_history_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["iteration", "parameter", "numel", "mean", "std", "min", "max", "l2_norm"],
            )
            writer.writeheader()
            writer.writerows(param_rows)


def _figure_to_frame(fig: Figure) -> np.ndarray:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    frame = iio.imread(buf)
    buf.close()
    plt.close(fig)
    return frame


def _render_pattern_frame(optimizer: OpticalSGDOptimizer, pattern_snapshot: torch.Tensor, iteration: int) -> np.ndarray:
    k, _ = pattern_snapshot.shape
    hp = optimizer.config.projector_height
    patterns_np = pattern_snapshot.numpy()
    patterns_2d = np.tile(patterns_np[:, None, :], (1, hp, 1))

    n_cols = min(k, 5)
    n_rows = (k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if k == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(k):
        row, col = divmod(idx, n_cols)
        axes[row, col].imshow(
            patterns_2d[idx],
            cmap="gray",
            vmin=optimizer.config.min_value,
            vmax=optimizer.config.max_value,
        )
        axes[row, col].set_title(f"P{idx + 1}", fontsize=10)
        axes[row, col].axis("off")

    for idx in range(k, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    fig.suptitle(f"Pattern Evolution - Iteration {iteration}", fontsize=12)
    plt.tight_layout()
    return _figure_to_frame(fig)


def _render_spectrum_frame(optimizer: OpticalSGDOptimizer, pattern_snapshot: torch.Tensor, iteration: int) -> np.ndarray:
    patterns_np = pattern_snapshot.numpy()
    k, width = patterns_np.shape
    freq_bins = np.fft.rfftfreq(width, d=1.0 / width)

    fig, axes = plt.subplots(k, 1, figsize=(10, 2.5 * k))
    if k == 1:
        axes = [axes]

    for idx in range(k):
        spectrum = np.abs(np.fft.rfft(patterns_np[idx]))
        axes[idx].plot(freq_bins, spectrum, linewidth=1.5)
        axes[idx].set_xlabel("Frequency")
        axes[idx].set_ylabel("Magnitude")
        axes[idx].set_title(f"Pattern {idx + 1} Spectrum")
        axes[idx].grid(True, alpha=0.3)
        if optimizer.config.max_frequency is not None:
            axes[idx].axvline(
                optimizer.config.max_frequency,
                color="r",
                linestyle="--",
                linewidth=1.5,
                label=f"Frequency limit = {optimizer.config.max_frequency:.2f}",
            )
            axes[idx].legend()

    fig.suptitle(f"Spectrum Evolution - Iteration {iteration}", fontsize=12)
    plt.tight_layout()
    return _figure_to_frame(fig)


def _write_gif(
    optimizer: OpticalSGDOptimizer,
    path: Path,
    frame_builder,
    duration_ms: int = 120,
) -> None:
    if not optimizer.pattern_history:
        return

    frames = [
        frame_builder(optimizer, snapshot, iteration)
        for snapshot, iteration in zip(optimizer.pattern_history, optimizer.history_iterations, strict=False)
    ]
    if frames:
        iio.imwrite(path, frames, duration=duration_ms, loop=0)


def generate_pattern_evolution_gif(optimizer: OpticalSGDOptimizer) -> None:
    if optimizer.config.output_dir is None:
        return
    _write_gif(
        optimizer,
        Path(optimizer.config.output_dir) / "pattern_evolution.gif",
        _render_pattern_frame,
    )


def generate_spectrum_evolution_gif(optimizer: OpticalSGDOptimizer) -> None:
    if optimizer.config.output_dir is None:
        return
    _write_gif(
        optimizer,
        Path(optimizer.config.output_dir) / "spectrum_evolution.gif",
        _render_spectrum_frame,
    )
