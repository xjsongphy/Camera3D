from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from math import ceil, sqrt
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lab2.decoder import ZNCCDecoder, ZNCCNNDecoder, soft_correspondence_loss
from lab2.scene_genertor import SCENE_PRESETS, create_standard_renderer


@dataclass
class GradientComparisonResult:
    method_name: str
    time: float
    memory_mb: float
    gradient: torch.Tensor
    loss_value: float


@dataclass
class SampleComparison:
    sample_index: int
    pattern_seed: int
    loss_autodiff: float
    loss_fd: float
    relative_l2_error: float
    signed_relative_error: torch.Tensor
    autodiff_gradient: torch.Tensor
    fd_gradient: torch.Tensor
    autodiff_time: float
    fd_time: float


def auto_detect_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def create_output_dir(base_dir: str, scene: str, decoder: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / "gradient_comparison" / f"{timestamp}_{scene}_{decoder}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_decoder(
    decoder_name: str,
    num_patterns: int,
    neighborhood: int,
    device: torch.device,
) -> torch.nn.Module:
    if decoder_name == "zncc":
        decoder = ZNCCDecoder(neighborhood_size=neighborhood)
    elif decoder_name == "zncc_nn":
        decoder = ZNCCNNDecoder(
            num_patterns=num_patterns,
            neighborhood_size=neighborhood,
        )
    else:
        raise ValueError(f"Unknown decoder: {decoder_name}")
    decoder.to(device)
    return decoder


def sample_random_patterns(
    num_patterns: int,
    projector_width: int,
    dtype: torch.dtype,
    device: torch.device,
    sample_seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sample_seed)
    patterns = torch.rand((num_patterns, projector_width), generator=generator, dtype=dtype)
    return patterns.to(device=device, dtype=dtype)


def compute_autodiff_gradient(
    renderer,
    patterns: torch.Tensor,
    decoder: torch.nn.Module,
    tau: float,
    penalty: str,
) -> GradientComparisonResult:
    device = renderer.device
    patterns_param = torch.nn.Parameter(patterns.clone())

    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0

    images = renderer.render_images_autodiff(patterns_param)
    gt_corr = renderer.gt_corr
    scores = decoder(images, patterns_param)
    loss = soft_correspondence_loss(scores, gt_corr, tau=tau, penalty=penalty)
    loss.backward()

    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0

    return GradientComparisonResult(
        method_name="autodiff",
        time=end_time - start_time,
        memory_mb=end_memory - start_memory,
        gradient=patterns_param.grad.detach().clone(),
        loss_value=loss.item(),
    )


def compute_finite_difference_gradient(
    renderer,
    patterns: torch.Tensor,
    decoder: torch.nn.Module,
    epsilon: float,
    tau: float,
    penalty: str,
) -> GradientComparisonResult:
    device = renderer.device
    num_patterns, projector_width = patterns.shape

    with torch.no_grad():
        images = renderer.render_images(patterns)
        gt_corr = renderer.gt_corr
        scores = decoder(images, patterns)
        base_loss = soft_correspondence_loss(scores, gt_corr, tau=tau, penalty=penalty)

    gradient = torch.zeros_like(patterns)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0

    for pattern_idx in range(num_patterns):
        for column_idx in range(projector_width):
            perturbed_patterns = patterns.clone()
            perturbed_patterns[pattern_idx, column_idx] += epsilon
            with torch.no_grad():
                images_perturbed = renderer.render_images(perturbed_patterns)
                scores_perturbed = decoder(images_perturbed, perturbed_patterns)
                perturbed_loss = soft_correspondence_loss(
                    scores_perturbed,
                    gt_corr,
                    tau=tau,
                    penalty=penalty,
                )
            gradient[pattern_idx, column_idx] = (perturbed_loss - base_loss) / epsilon

    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0

    return GradientComparisonResult(
        method_name="finite_difference",
        time=end_time - start_time,
        memory_mb=end_memory - start_memory,
        gradient=gradient,
        loss_value=base_loss.item(),
    )


def compare_single_sample(
    renderer,
    decoder: torch.nn.Module,
    patterns: torch.Tensor,
    sample_index: int,
    pattern_seed: int,
    epsilon: float,
    tau: float,
    penalty: str,
    relative_eps: float,
) -> SampleComparison:
    autodiff_result = compute_autodiff_gradient(
        renderer=renderer,
        patterns=patterns,
        decoder=decoder,
        tau=tau,
        penalty=penalty,
    )
    fd_result = compute_finite_difference_gradient(
        renderer=renderer,
        patterns=patterns,
        decoder=decoder,
        epsilon=epsilon,
        tau=tau,
        penalty=penalty,
    )

    signed_relative_error = (
        (autodiff_result.gradient - fd_result.gradient)
        / (autodiff_result.gradient.abs() + relative_eps)
    )
    relative_l2_error = (
        (autodiff_result.gradient - fd_result.gradient).norm()
        / (autodiff_result.gradient.norm() + relative_eps)
    ).item()

    return SampleComparison(
        sample_index=sample_index,
        pattern_seed=pattern_seed,
        loss_autodiff=autodiff_result.loss_value,
        loss_fd=fd_result.loss_value,
        relative_l2_error=relative_l2_error,
        signed_relative_error=signed_relative_error.detach().cpu(),
        autodiff_gradient=autodiff_result.gradient.detach().cpu(),
        fd_gradient=fd_result.gradient.detach().cpu(),
        autodiff_time=autodiff_result.time,
        fd_time=fd_result.time,
    )


def plot_signed_relative_error_heatmaps(
    comparisons: list[SampleComparison],
    output_path: Path,
) -> None:
    num_samples = len(comparisons)
    num_cols = max(1, ceil(sqrt(num_samples)))
    num_rows = ceil(num_samples / num_cols)

    all_values = np.concatenate([
        comparison.signed_relative_error.numpy().reshape(-1)
        for comparison in comparisons
    ])
    vmax = float(np.max(np.abs(all_values)))
    if vmax == 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(5 * num_cols, 3.5 * num_rows),
        squeeze=False,
    )

    last_im = None
    for ax, comparison in zip(axes.flat, comparisons):
        heatmap = comparison.signed_relative_error.numpy()
        last_im = ax.imshow(
            heatmap,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(
            f"Sample {comparison.sample_index + 1}\n"
            f"seed={comparison.pattern_seed}, relL2={comparison.relative_l2_error:.3e}"
        )
        ax.set_xlabel("Projector Column")
        ax.set_ylabel("Pattern Index")

    for ax in axes.flat[num_samples:]:
        ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.92, label="Signed Relative Error")

    fig.suptitle("(dL/dc_auto - dL/dc_fd) / (|dL/dc_auto| + eps)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_relative_l2_histogram(
    comparisons: list[SampleComparison],
    output_path: Path,
) -> None:
    values = np.array([comparison.relative_l2_error for comparison in comparisons], dtype=np.float64)
    bins = min(20, max(5, len(values)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_xlabel(r"$||g_{auto} - g_{fd}|| / ||g_{auto}||$")
    ax.set_ylabel("Count")
    ax.set_title("Relative L2 Error Histogram Across Random Pattern Samples")
    ax.grid(True, alpha=0.3)

    mean_value = float(values.mean())
    ax.axvline(mean_value, color="crimson", linestyle="--", linewidth=2, label=f"mean={mean_value:.3e}")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(
    comparisons: list[SampleComparison],
    output_dir: Path,
    config_snapshot: dict[str, Any],
) -> None:
    relative_l2_values = np.array([comparison.relative_l2_error for comparison in comparisons], dtype=np.float64)
    summary = {
        "config": config_snapshot,
        "num_samples": len(comparisons),
        "relative_l2_error": {
            "mean": float(relative_l2_values.mean()),
            "std": float(relative_l2_values.std()),
            "min": float(relative_l2_values.min()),
            "max": float(relative_l2_values.max()),
        },
        "samples": [
            {
                "sample_index": comparison.sample_index,
                "pattern_seed": comparison.pattern_seed,
                "loss_autodiff": comparison.loss_autodiff,
                "loss_fd": comparison.loss_fd,
                "relative_l2_error": comparison.relative_l2_error,
                "autodiff_time": comparison.autodiff_time,
                "fd_time": comparison.fd_time,
            }
            for comparison in comparisons
        ],
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    torch.save(
        {
            "signed_relative_errors": torch.stack([c.signed_relative_error for c in comparisons], dim=0),
            "autodiff_gradients": torch.stack([c.autodiff_gradient for c in comparisons], dim=0),
            "fd_gradients": torch.stack([c.fd_gradient for c in comparisons], dim=0),
            "relative_l2_errors": torch.tensor(relative_l2_values, dtype=torch.float32),
            "pattern_seeds": torch.tensor([c.pattern_seed for c in comparisons], dtype=torch.long),
        },
        output_dir / "gradient_tensors.pt",
    )


def compare_for_decoder(
    cfg: dict[str, Any],
    decoder_name: str,
    num_samples: int,
    epsilon: float,
    relative_eps: float,
    output_root: Path,
) -> None:
    scene_name = cfg["scene"]
    training = cfg["training"]
    rendering = cfg["rendering"]
    seed = int(training.get("seed", 0))
    device = auto_detect_device(rendering.get("device", "auto"))
    spp = int(rendering.get("spp", 64))
    num_patterns = int(training.get("num_patterns", 4))
    neighborhood = int(training.get("neighborhood", 1))
    tau = float(training.get("tau", 50.0))
    penalty = str(training.get("penalty", "l1"))

    print("\n" + "=" * 60)
    print(f"Decoder: {decoder_name}")
    print(f"Scene: {scene_name}")
    print(f"Samples: {num_samples}")
    print(f"Patterns per sample: {num_patterns} x projector_width")
    print(f"Device: {device}")
    print("=" * 60)

    set_random_seed(seed)
    renderer = create_standard_renderer(scene_name, device=str(device), spp=spp)
    renderer._depth = None
    renderer._gt_corr = None
    renderer.render_depth_for_visualization()

    set_random_seed(seed)
    decoder = build_decoder(
        decoder_name=decoder_name,
        num_patterns=num_patterns,
        neighborhood=neighborhood,
        device=device,
    )
    for param in decoder.parameters():
        param.requires_grad_(False)
    decoder.eval()

    projector_width = int(renderer.projector.width)
    comparisons: list[SampleComparison] = []

    for sample_index in range(num_samples):
        pattern_seed = seed + sample_index
        patterns = sample_random_patterns(
            num_patterns=num_patterns,
            projector_width=projector_width,
            dtype=renderer.dtype,
            device=device,
            sample_seed=pattern_seed,
        )
        print(
            f"  sample {sample_index + 1}/{num_samples}: "
            f"pattern_seed={pattern_seed}, shape={tuple(patterns.shape)}"
        )
        comparison = compare_single_sample(
            renderer=renderer,
            decoder=decoder,
            patterns=patterns,
            sample_index=sample_index,
            pattern_seed=pattern_seed,
            epsilon=epsilon,
            tau=tau,
            penalty=penalty,
            relative_eps=relative_eps,
        )
        comparisons.append(comparison)
        print(
            f"    relL2={comparison.relative_l2_error:.3e}, "
            f"loss_auto={comparison.loss_autodiff:.6f}, "
            f"loss_fd={comparison.loss_fd:.6f}"
        )

    decoder_output_dir = output_root / decoder_name
    decoder_output_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "scene": scene_name,
        "decoder": decoder_name,
        "num_samples": num_samples,
        "epsilon": epsilon,
        "relative_eps": relative_eps,
        "seed": seed,
        "training": training,
        "rendering": rendering,
        "projector_width": projector_width,
    }
    with open(decoder_output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    plot_signed_relative_error_heatmaps(
        comparisons=comparisons,
        output_path=decoder_output_dir / "signed_relative_error_heatmaps.png",
    )
    plot_relative_l2_histogram(
        comparisons=comparisons,
        output_path=decoder_output_dir / "relative_l2_error_histogram.png",
    )
    save_summary(
        comparisons=comparisons,
        output_dir=decoder_output_dir,
        config_snapshot=config_snapshot,
    )

    print(f"Saved decoder results to: {decoder_output_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare finite-difference and autodiff gradients.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--decoder", type=str, choices=["zncc", "zncc_nn", "both"], default=None)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--num-samples", type=int, default=4, help="Number of random pattern tensors")
    parser.add_argument("--epsilon", type=float, default=1e-2, help="Finite-difference perturbation")
    parser.add_argument("--relative-eps", type=float, default=1e-8, help="Denominator stabilizer for signed relative error")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output base directory")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")

    cfg = load_config(args.config)

    if args.decoder is not None:
        cfg["decoder"] = args.decoder
    if args.device is not None:
        cfg.setdefault("rendering", {})["device"] = args.device
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed

    scene_name = cfg["scene"]
    if scene_name not in SCENE_PRESETS:
        raise ValueError(f"Unknown scene '{scene_name}'. Available: {list(SCENE_PRESETS.keys())}")

    output_cfg = cfg.setdefault("output", {})
    decoder_mode = cfg.get("decoder", "zncc")
    output_base_dir = args.output_dir or output_cfg.get("base_dir", "outputs/lab2")
    output_root = create_output_dir(output_base_dir, scene_name, decoder_mode)

    run_config = {
        "config_path": args.config,
        "scene": scene_name,
        "decoder": decoder_mode,
        "num_samples": args.num_samples,
        "epsilon": args.epsilon,
        "relative_eps": args.relative_eps,
        "device": cfg.setdefault("rendering", {}).get("device", "auto"),
        "seed": cfg.setdefault("training", {}).get("seed"),
    }
    with open(output_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Gradient Comparison: Finite Difference vs Autodiff")
    print("=" * 60)
    print(f"Output directory: {output_root}")

    if decoder_mode == "both":
        for decoder_name in ("zncc", "zncc_nn"):
            compare_for_decoder(
                cfg=cfg,
                decoder_name=decoder_name,
                num_samples=args.num_samples,
                epsilon=args.epsilon,
                relative_eps=args.relative_eps,
                output_root=output_root,
            )
    else:
        compare_for_decoder(
            cfg=cfg,
            decoder_name=decoder_mode,
            num_samples=args.num_samples,
            epsilon=args.epsilon,
            relative_eps=args.relative_eps,
            output_root=output_root,
        )

    print(f"\nResults saved to: {output_root}")


if __name__ == "__main__":
    main()
