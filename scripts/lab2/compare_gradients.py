"""
Compare finite difference vs automatic differentiation gradients.

This script compares two methods for computing ∂loss/∂patterns:
1. Finite difference: Perturb patterns, render, compute loss difference
2. Automatic differentiation: Backprop through differentiable renderer

Metrics compared:
- Gradient accuracy (directional cosine similarity)
- Computation time
- Memory usage
- Convergence behavior
- Sensitivity to hyperparameters (epsilon, noise, etc.)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from src.lab2.decoder import ZNCCDecoder, soft_correspondence_loss, hard_decode, correspondence_metrics
from src.lab2.OpticalSGD import DecoderType, OptimizerConfig, OpticalSGDOptimizer
from src.lab2.shader import StructuredLightRenderer


@dataclass
class GradientComparisonResult:
    """Results from comparing two gradient computation methods."""
    method_name: str
    time: float
    memory_mb: float
    gradient: torch.Tensor
    loss_value: float


def compute_autodiff_gradient(
    renderer: StructuredLightRenderer,
    patterns: torch.Tensor,
    decoder: torch.nn.Module,
    tau: float = 50.0,
    penalty: str = "l1",
) -> GradientComparisonResult:
    """
    Compute gradient using automatic differentiation.

    Args:
        renderer: StructuredLightRenderer
        patterns: Current patterns [K, Wp]
        decoder: ZNCC or ZNCC-NN decoder
        tau: Softmax temperature
        penalty: Penalty function

    Returns:
        Gradient comparison result
    """
    device = renderer.device
    dtype = renderer.dtype

    # Make patterns learnable
    patterns_param = torch.nn.Parameter(patterns.clone())

    # Reset gradients
    if patterns_param.grad is not None:
        patterns_param.grad.zero_()

    # Start timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0

    # Forward pass
    images = renderer.render_images_autodiff(patterns_param)
    gt_corr = renderer.gt_corr

    scores = decoder(images, patterns_param)
    loss = soft_correspondence_loss(scores, gt_corr, tau=tau, penalty=penalty)

    # Backward pass
    loss.backward()

    # End timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0

    return GradientComparisonResult(
        method_name="autodiff",
        time=end_time - start_time,
        memory_mb=end_memory - start_memory,
        gradient=patterns_param.grad.clone(),
        loss_value=loss.item(),
    )


def compute_finite_difference_gradient(
    renderer: StructuredLightRenderer,
    patterns: torch.Tensor,
    decoder: torch.nn.Module,
    epsilon: float = 0.01,
    tau: float = 50.0,
    penalty: str = "l1",
) -> GradientComparisonResult:
    """
    Compute gradient using finite differences.

    For each pattern element c_i:
        ∂loss/∂c_i ≈ (loss(c + ε·e_i) - loss(c)) / ε

    Args:
        renderer: StructuredLightRenderer
        patterns: Current patterns [K, Wp]
        decoder: ZNCC or ZNCC-NN decoder
        epsilon: Perturbation magnitude
        tau: Softmax temperature
        penalty: Penalty function

    Returns:
        Gradient comparison result
    """
    device = renderer.device
    dtype = renderer.dtype
    K, Wp = patterns.shape

    # Compute base loss
    with torch.no_grad():
        images = renderer.render_images(patterns)
        gt_corr = renderer.gt_corr
        scores = decoder(images, patterns)
        base_loss = soft_correspondence_loss(scores, gt_corr, tau=tau, penalty=penalty)

    # Allocate gradient tensor
    gradient = torch.zeros_like(patterns)

    # Start timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0

    # Compute finite difference for each element
    print(f"Computing finite difference gradient for {K}x{Wp} pattern elements...")

    for k in range(K):
        for i in range(Wp):
            # Create perturbed patterns
            perturbed_patterns = patterns.clone()
            perturbed_patterns[k, i] += epsilon

            # Compute perturbed loss
            with torch.no_grad():
                images_perturbed = renderer.render_images(perturbed_patterns)
                scores_perturbed = decoder(images_perturbed, perturbed_patterns)
                perturbed_loss = soft_correspondence_loss(
                    scores_perturbed, gt_corr, tau=tau, penalty=penalty
                )

            # Finite difference
            gradient[k, i] = (perturbed_loss - base_loss) / epsilon

        if (k + 1) % 1 == 0:
            print(f"  Progress: {k + 1}/{K} pattern(s)")

    # End timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0

    return GradientComparisonResult(
        method_name="finite_difference",
        time=end_time - start_time,
        memory_mb=end_memory - start_memory,
        gradient=gradient,
        loss_value=base_loss.item(),
    )


def compare_gradients(
    result_autodiff: GradientComparisonResult,
    result_fd: GradientComparisonResult,
) -> dict[str, Any]:
    """
    Compare two gradient computation results.

    Args:
        result_autodiff: Automatic differentiation result
        result_fd: Finite difference result

    Returns:
        Dictionary of comparison metrics
    """
    grad_autodiff = result_autodiff.gradient.flatten()
    grad_fd = result_fd.gradient.flatten()

    # Normalize for comparison
    grad_autodiff_norm = grad_autodiff / (grad_autodiff.norm() + 1e-8)
    grad_fd_norm = grad_fd / (grad_fd.norm() + 1e-8)

    # Directional cosine similarity
    cosine_sim = (grad_autodiff_norm * grad_fd_norm).sum().item()

    # Relative error
    relative_error = (grad_autodiff - grad_fd).norm() / (grad_autodiff.norm() + 1e-8)
    relative_error = relative_error.item()

    # Absolute difference
    abs_diff = (grad_autodiff - grad_fd).abs().mean().item()

    # Speedup
    speedup = result_fd.time / result_autodiff.time if result_autodiff.time > 0 else float('inf')

    return {
        "cosine_similarity": cosine_sim,
        "relative_error": relative_error,
        "absolute_difference": abs_diff,
        "autodiff_time": result_autodiff.time,
        "fd_time": result_fd.time,
        "speedup": speedup,
        "autodiff_memory_mb": result_autodiff.memory_mb,
        "fd_memory_mb": result_fd.memory_mb,
    }


def plot_gradient_comparison(
    result_autodiff: GradientComparisonResult,
    result_fd: GradientComparisonResult,
    comparison: dict[str, Any],
    patterns: torch.Tensor,
) -> Figure:
    """Plot gradient comparison visualization."""
    K, Wp = patterns.shape

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. Gradient magnitudes
    grad_auto_flat = result_autodiff.gradient.abs().mean(dim=0).cpu().numpy()
    grad_fd_flat = result_fd.gradient.abs().mean(dim=0).cpu().numpy()

    axes[0, 0].plot(grad_auto_flat, label='Autodiff', linewidth=2)
    axes[0, 0].plot(grad_fd_flat, label='Finite Diff', linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel("Projector Column")
    axes[0, 0].set_ylabel("Mean |∂loss/∂c|")
    axes[0, 0].set_title("Gradient Magnitudes")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Gradient difference
    grad_diff = (result_autodiff.gradient - result_fd.gradient).abs().mean(dim=0).cpu().numpy()
    axes[0, 1].plot(grad_diff, linewidth=2, color='red')
    axes[0, 1].set_xlabel("Projector Column")
    axes[0, 1].set_ylabel("|∂auto/∂c - ∂FD/∂c|")
    axes[0, 1].set_title(f"Absolute Difference (mean={grad_diff.mean():.6f})")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Scatter plot: autodiff vs FD
    grad_auto_flat = result_autodiff.gradient.flatten().cpu().numpy()
    grad_fd_flat = result_fd.gradient.flatten().cpu().numpy()

    sample_idx = np.linspace(0, len(grad_auto_flat) - 1, min(1000, len(grad_auto_flat)), dtype=int)
    axes[1, 0].scatter(grad_fd_flat[sample_idx], grad_auto_flat[sample_idx], alpha=0.3, s=1)
    axes[1, 0].plot([grad_fd_flat.min(), grad_fd_flat.max()],
                    [grad_fd_flat.min(), grad_fd_flat.max()], 'r--', label='y=x')
    axes[1, 0].set_xlabel("Finite Difference Gradient")
    axes[1, 0].set_ylabel("Autodiff Gradient")
    axes[1, 0].set_title(f"Gradient Correlation (cos={comparison['cosine_similarity']:.4f})")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Performance comparison
    methods = ['Autodiff', 'Finite Diff']
    times = [result_autodiff.time, result_fd.time]

    bars = axes[1, 1].bar(methods, times, color=['blue', 'orange'])
    axes[1, 1].set_ylabel("Time (seconds)")
    axes[1, 1].set_title(f"Computation Time ({comparison['speedup']:.1f}x speedup)")
    axes[1, 1].bar_label(bars, [f"{t:.2f}s" for t in times])

    # 5. Memory usage
    memories = [result_autodiff.memory_mb, result_fd.memory_mb]
    bars = axes[2, 0].bar(methods, memories, color=['blue', 'orange'])
    axes[2, 0].set_ylabel("Memory (MB)")
    axes[2, 0].set_title("Memory Usage")
    axes[2, 0].bar_label(bars, [f"{m:.1f} MB" for m in memories])

    # 6. Summary metrics
    axes[2, 1].axis('off')
    summary_text = f"""
    Gradient Comparison Summary

    Cosine Similarity: {comparison['cosine_similarity']:.4f}
    Relative Error: {comparison['relative_error']:.4f}
    Absolute Difference: {comparison['absolute_difference']:.6f}

    Autodiff: {comparison['autodiff_time']:.3f}s, {comparison['autodiff_memory_mb']:.1f} MB
    Finite Diff: {comparison['fd_time']:.3f}s, {comparison['fd_memory_mb']:.1f} MB
    Speedup: {comparison['speedup']:.1f}x
    """
    axes[2, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    fontfamily='monospace')

    plt.suptitle("Finite Difference vs Automatic Differentiation", fontsize=16)
    plt.tight_layout()
    return fig


def run_convergence_comparison(
    renderer: StructuredLightRenderer,
    patterns_init: torch.Tensor,
    decoder_type: DecoderType,
    num_iterations: int = 50,
) -> dict[str, Any]:
    """
    Compare convergence behavior of autodiff vs finite difference.

    Note: This is simplified - proper comparison would implement
    finite difference SGD loop.
    """
    # For now, just run autodiff optimization
    config = OptimizerConfig(
        num_iterations=num_iterations,
        learning_rate=0.01,
        decoder_type=decoder_type,
        neighborhood_size=1,
        penalty="l1",
        tau=50.0,
        log_interval=10,
        save_interval=num_iterations,  # Don't save intermediate
    )

    optimizer = OpticalSGDOptimizer(renderer, config)
    optimizer.initialize_patterns(
        patterns_init.shape[0],
        patterns_init.shape[1],
        init_mode="given",
    )

    # Set initial patterns
    optimizer.patterns.data = patterns_init.clone()

    # Run optimization
    start_time = time.time()
    loss_history = optimizer.optimize(num_iterations)
    end_time = time.time()

    return {
        "loss_history": loss_history,
        "total_time": end_time - start_time,
        "final_loss": loss_history[-1],
    }


@click.command()
@click.option('--scene', type=click.Choice(['sl_plane_diffuse', 'sl_marble_objects', 'sl_wood_glass']), default='sl_plane_diffuse',
              help='Scene preset to use')
@click.option('--output-dir', type=str, default='results/lab2/gradient_comparison',
              help='Base output directory')
@click.option('--num-patterns', type=int, default=2,
              help='Number of patterns (keep small for FD)')
@click.option('--projector-width', type=int, default=64,
              help='Projector width (keep small for FD)')
@click.option('--epsilon', type=float, default=0.01,
              help='Finite difference perturbation')
@click.option('--tau', type=float, default=50.0,
              help='Softmax temperature')
@click.option('--penalty', type=str, default='l1',
              type=click.Choice(['l1', 'zero_tolerance', 'one_tolerance']),
              help='Penalty function')
@click.option('--run-convergence', is_flag=True,
              help='Run convergence comparison (slow)')
@click.option('--convergence-iters', type=int, default=50,
              help='Iterations for convergence test')
@click.option('--device', type=str, default='cpu',
              help='Device to use')
def main(
    scene: str,
    output_dir: str,
    num_patterns: int,
    projector_width: int,
    epsilon: float,
    tau: float,
    penalty: str,
    run_convergence: bool,
    convergence_iters: int,
    device: str,
) -> None:
    """Compare finite difference vs automatic differentiation gradients."""

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{timestamp}_gradient_comparison"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = {
        "scene": scene,
        "num_patterns": num_patterns,
        "projector_width": projector_width,
        "epsilon": epsilon,
        "tau": tau,
        "penalty": penalty,
        "device": device,
        "run_convergence": run_convergence,
        "convergence_iters": convergence_iters if run_convergence else None,
    }
    config_file = output_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 60)
    print("Gradient Comparison: Finite Difference vs Autodiff")
    print("=" * 60)
    print(f"Output directory: {output_path}")

    # Load scene path from preset
    scene_dir = Path("assets/scenes") / scene
    scene_path = scene_dir / "scene.xml"
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene XML not found: {scene_path}")

    # Setup renderer
    print(f"\nScene: {scene} ({scene_path})")
    renderer = StructuredLightRenderer(device=device, dtype=torch.float32, spp=32)

    projector_config = {
        "width": projector_width,
        "height": 1,
        "fx": 200.0,
        "fy": 200.0,
        "cx": projector_width / 2,
        "cy": 0.5,
        "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "t": [0, 0, 0],
    }
    renderer.set_projector(projector_config)
    renderer.load_scene(scene_path)

    # Create initial patterns
    patterns = torch.rand(num_patterns, projector_width, device=device)

    # Create decoder
    decoder = ZNCCDecoder(neighborhood_size=1)
    decoder.to(device)

    print(f"\nConfiguration:")
    print(f"  Patterns: {num_patterns} x {projector_width}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Tau: {tau}")
    print(f"  Penalty: {penalty}")

    # Compute autodiff gradient
    print("\n" + "=" * 60)
    print("Computing AUTOMATIC DIFFERENTIATION gradient...")
    print("=" * 60)
    result_autodiff = compute_autodiff_gradient(
        renderer, patterns, decoder, tau=tau, penalty=penalty
    )
    print(f"  Time: {result_autodiff.time:.3f}s")
    print(f"  Memory: {result_autodiff.memory_mb:.1f} MB")
    print(f"  Loss: {result_autodiff.loss_value:.6f}")
    print(f"  Gradient norm: {result_autodiff.gradient.norm():.6f}")

    # Compute finite difference gradient
    print("\n" + "=" * 60)
    print("Computing FINITE DIFFERENCE gradient...")
    print("=" * 60)
    result_fd = compute_finite_difference_gradient(
        renderer, patterns, decoder, epsilon=epsilon, tau=tau, penalty=penalty
    )
    print(f"  Time: {result_fd.time:.3f}s")
    print(f"  Memory: {result_fd.memory_mb:.1f} MB")
    print(f"  Loss: {result_fd.loss_value:.6f}")
    print(f"  Gradient norm: {result_fd.gradient.norm():.6f}")

    # Compare
    comparison = compare_gradients(result_autodiff, result_fd)

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"  Cosine Similarity: {comparison['cosine_similarity']:.4f}")
    print(f"  Relative Error: {comparison['relative_error']:.4f}")
    print(f"  Absolute Difference: {comparison['absolute_difference']:.6f}")
    print(f"  Speedup: {comparison['speedup']:.1f}x")

    # Plot
    print("\nCreating visualizations...")
    fig = plot_gradient_comparison(result_autodiff, result_fd, comparison, patterns)
    fig.savefig(output_path / "gradient_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path / 'gradient_comparison.png'}")

    # Optional: convergence comparison
    if run_convergence:
        print("\n" + "=" * 60)
        print("Running convergence comparison...")
        print("=" * 60)
        result_conv = run_convergence_comparison(
            renderer, patterns, DecoderType.ZNCC, convergence_iters
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(result_conv['loss_history'], linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Autodiff Convergence")
        ax.grid(True, alpha=0.3)
        fig.savefig(output_path / "convergence.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_path / 'convergence.png'}")

    print(f"\n✅ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
