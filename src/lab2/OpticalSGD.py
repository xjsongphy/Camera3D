"""
OpticalSGD optimizer for structured light pattern optimization.

This module implements the Optical SGD optimizer from Chen et al., CVPR 2020.
Core insight: use softmax to approximate argmax, making the entire pipeline differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .shader import StructuredLightRenderer


@dataclass
class OptimizerConfig:
    """Configuration for OpticalSGD optimizer."""

    # Optimization parameters
    num_iterations: int = 100
    learning_rate: float = 0.01
    tau: float = 50.0  # Softmax temperature for correspondence

    # Pattern constraints
    min_value: float = 0.0
    max_value: float = 1.0
    max_frequency: float | None = None  # Nyquist frequency constraint

    # Logging and visualization
    log_interval: int = 10
    save_interval: int = 50
    output_dir: str | Path | None = None


def zncc_scores(images: torch.Tensor, patterns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute vectorized ZNCC scores between observed images and pattern references.

    For each camera pixel m and projector column n, compute:
        z[m, n] = ZNCC(observed_signal_at_m, pattern_reference_at_n)

    Args:
        images: Observed images [K, H, W]
        patterns: Pattern references [K, Wp]
        eps: Small constant for numerical stability

    Returns:
        ZNCC scores [H, W, Wp]
    """
    K, H, W = images.shape
    _, Wp = patterns.shape
    device = images.device
    dtype = images.dtype

    # Reshape images: [K, H, W] -> [K, H*W]
    obs = images.reshape(K, H * W)

    # Normalize observed signals (zero-normalized cross-correlation)
    obs_mean = obs.mean(dim=0, keepdim=True)  # [1, M]
    obs_centered = obs - obs_mean  # [K, M]
    obs_norm = obs_centered / (obs_centered.norm(dim=0, keepdim=True) + eps)  # [K, M]

    # Normalize pattern references
    ref_mean = patterns.mean(dim=0, keepdim=True)  # [1, Wp]
    ref_centered = patterns - ref_mean  # [K, Wp]
    ref_norm = ref_centered / (ref_centered.norm(dim=0, keepdim=True) + eps)  # [K, Wp]

    # Compute ZNCC scores: [M, Wp] where M = H*W
    scores = obs_norm.T @ ref_norm  # [H*W, Wp]

    return scores.reshape(H, W, Wp)


def soft_correspondence_loss(
    scores: torch.Tensor,
    gt_corr: torch.Tensor,
    tau: float = 50.0,
) -> torch.Tensor:
    """
    Compute soft correspondence loss using softmax approximation.

    Instead of hard argmax: pred[m] = argmax_n z[m, n]
    We use soft expected penalty: E[|n - gt[m]|] ~ sum_n softmax(tau * z[m, n]) * |n - gt[m]|

    This is fully differentiable and matches the Optical SGD paper's approach.

    Args:
        scores: ZNCC scores [H, W, Wp]
        gt_corr: Ground truth correspondence [H, W]
        tau: Softmax temperature (higher = sharper approximation)

    Returns:
        Scalar loss value
    """
    H, W, Wp = scores.shape
    device = scores.device
    dtype = scores.dtype

    # Valid mask (where gt_corr is finite)
    valid = torch.isfinite(gt_corr)

    if not valid.any():
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Projector column indices [0, 1, ..., Wp-1]
    index = torch.arange(Wp, device=device, dtype=dtype)

    # Penalty matrix: |n - gt_corr[m]| for all m, n
    # Shape: [H, W, Wp]
    penalty = torch.abs(index[None, None, :] - gt_corr[..., None])

    # Softmax over projector columns (temperature-scaled)
    # prob[m, n] = softmax(tau * z[m, n]) over n
    prob = torch.softmax(tau * scores, dim=-1)  # [H, W, Wp]

    # Expected penalty at each pixel
    # loss_map[m] = sum_n prob[m, n] * penalty[m, n]
    loss_map = (prob * penalty).sum(dim=-1)  # [H, W]

    # Mean loss over valid pixels
    return loss_map[valid].mean()


def frequency_penalty(patterns: torch.Tensor, max_frequency: float) -> torch.Tensor:
    """
    Compute frequency constraint penalty.

    Penalizes energy above Nyquist frequency to ensure patterns can be
    reliably displayed on real hardware.

    Args:
        patterns: Current patterns [K, Wp]
        max_frequency: Maximum allowed frequency (Nyquist limit)

    Returns:
        Frequency penalty value (scalar)
    """
    K, Wp = patterns.shape
    device = patterns.device
    dtype = patterns.dtype

    # Compute FFT for each pattern
    patterns_fft = torch.fft.rfft(patterns, dim=-1, norm='ortho')

    # Frequency bins
    freq_bins = torch.fft.rfftfreq(Wp, d=1.0 / Wp)

    # Mask for frequencies above Nyquist limit
    high_freq_mask = freq_bins > max_frequency

    # Penalty: L2 energy of high-frequency components
    high_freq_energy = patterns_fft[:, high_freq_mask].abs().pow(2).sum()
    return high_freq_energy


def apply_frequency_constraints(patterns: torch.Tensor, max_frequency: float) -> torch.Tensor:
    """
    Apply hard frequency constraints via low-pass filtering.

    Args:
        patterns: Patterns to constrain [K, Wp]
        max_frequency: Maximum allowed frequency

    Returns:
        Frequency-constrained patterns [K, Wp]
    """
    K, Wp = patterns.shape

    # FFT
    patterns_fft = torch.fft.rfft(patterns, dim=-1)

    # Create low-pass filter mask
    freq_bins = torch.fft.rfftfreq(Wp, d=1.0 / Wp)
    freq_mask = freq_bins <= max_frequency

    # Apply filter
    patterns_fft_filtered = patterns_fft * freq_mask.to(device=patterns.device)

    # Inverse FFT
    patterns_filtered = torch.fft.irfft(patterns_fft_filtered, n=Wp, dim=-1)

    return patterns_filtered


def initialize_patterns(
    num_patterns: int,
    projector_width: int,
    device: torch.device,
    dtype: torch.dtype,
    init_mode: str = "random",
) -> torch.Tensor:
    """
    Initialize projection patterns.

    Args:
        num_patterns: Number of patterns K
        projector_width: Projector width Wp
        device: Torch device
        dtype: Torch dtype
        init_mode: Initialization mode ("random", "sine", "constant")

    Returns:
        Initialized patterns [K, Wp]
    """
    if init_mode == "random":
        # Random uniform initialization
        patterns = torch.rand(num_patterns, projector_width, device=device, dtype=dtype)

    elif init_mode == "constant":
        # Constant pattern (baseline)
        patterns = torch.ones(num_patterns, projector_width, device=device, dtype=dtype) * 0.5

    elif init_mode == "sine":
        # Multi-frequency sine waves (different frequency per pattern)
        patterns = torch.zeros(num_patterns, projector_width, device=device, dtype=dtype)
        frequencies = torch.linspace(1, 8, num_patterns, device=device, dtype=dtype)
        for k in range(num_patterns):
            x = torch.linspace(0, 2 * np.pi * frequencies[k], projector_width, device=device, dtype=dtype)
            patterns[k] = 0.5 + 0.5 * torch.sin(x)

    elif init_mode == "stripe":
        # Phase-shifted stripes
        patterns = torch.zeros(num_patterns, projector_width, device=device, dtype=dtype)
        for k in range(num_patterns):
            phase_shift = 2 * np.pi * k / num_patterns
            x = torch.linspace(0, 4 * np.pi, projector_width, device=device, dtype=dtype)
            patterns[k] = 0.5 + 0.5 * torch.sin(x + phase_shift)

    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    return patterns


class OpticalSGDOptimizer:
    """
    Optical SGD optimizer with proper differentiable pipeline.

    Uses softmax-approximated correspondence loss for end-to-end optimization.
    """

    def __init__(
        self,
        renderer: StructuredLightRenderer,
        config: OptimizerConfig | None = None,
    ) -> None:
        """
        Initialize the optimizer.

        Args:
            renderer: StructuredLightRenderer for rendering
            config: Optimizer configuration
        """
        self.renderer = renderer
        self.config = config or OptimizerConfig()

        # Training state
        self.patterns: torch.nn.Parameter | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.iteration = 0
        self.loss_history: list[float] = []

        # Ensure output directory exists
        if self.config.output_dir is not None:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def initialize_patterns(
        self,
        num_patterns: int,
        projector_width: int,
        init_mode: str = "random",
    ) -> torch.nn.Parameter:
        """
        Initialize patterns as learnable parameters.

        Args:
            num_patterns: Number of patterns K
            projector_width: Projector width Wp
            init_mode: Initialization mode

        Returns:
            Pattern parameter
        """
        patterns = initialize_patterns(
            num_patterns,
            projector_width,
            self.renderer.device,
            self.renderer.dtype,
            init_mode,
        )

        # Make patterns learnable
        self.patterns = torch.nn.Parameter(patterns)

        # Setup optimizer
        self.optimizer = torch.optim.Adam([self.patterns], lr=self.config.learning_rate)

        # Sync to renderer
        self.renderer.update_patterns(self.patterns.detach())

        return self.patterns

    def compute_loss(self, images: torch.Tensor, gt_corr: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss including correspondence and frequency penalty.

        Args:
            images: Rendered images [K, H, W]
            gt_corr: Ground truth correspondence [H, W]

        Returns:
            Total loss (scalar)
        """
        # ZNCC scores
        scores = zncc_scores(images, self.patterns)  # [H, W, Wp]

        # Soft correspondence loss
        corr_loss = soft_correspondence_loss(scores, gt_corr, self.config.tau)

        # Frequency penalty (if configured)
        if self.config.max_frequency is not None:
            freq_penalty = frequency_penalty(self.patterns, self.config.max_frequency)
            return corr_loss + freq_penalty

        return corr_loss

    def step(self) -> float:
        """
        Perform one optimization step.

        Returns:
            Loss value for this step
        """
        if self.patterns is None or self.optimizer is None:
            raise RuntimeError("Patterns must be initialized before optimization")

        # Forward pass: render with differentiable renderer
        images = self.renderer.render_images_autodiff(self.patterns)
        gt_corr = self.renderer.gt_corr

        # Compute loss
        loss = self.compute_loss(images, gt_corr)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Apply constraints (value clamping, frequency filtering)
        with torch.no_grad():
            self.patterns.data = self.patterns.data.clamp(self.config.min_value, self.config.max_value)

            if self.config.max_frequency is not None:
                self.patterns.data = apply_frequency_constraints(
                    self.patterns.data, self.config.max_frequency
                )
                self.patterns.data = self.patterns.data.clamp(
                    self.config.min_value, self.config.max_value
                )

        # Sync to renderer (for Mitsuba rendering if needed)
        self.renderer.update_patterns(self.patterns.detach())

        # Update iteration
        self.iteration += 1
        loss_value = loss.item()
        self.loss_history.append(loss_value)

        # Logging
        if self.config.log_interval > 0 and self.iteration % self.config.log_interval == 0:
            self._log_progress(loss_value)

        # Visualization
        if self.config.save_interval > 0 and self.iteration % self.config.save_interval == 0:
            self._save_visualization()

        return loss_value

    def optimize(self, num_iterations: int | None = None) -> list[float]:
        """
        Run optimization for specified iterations.

        Args:
            num_iterations: Number of iterations (default: config.num_iterations)

        Returns:
            Loss history
        """
        num_iterations = num_iterations or self.config.num_iterations

        for _ in range(num_iterations):
            self.step()

        return self.loss_history

    def _log_progress(self, loss: float) -> None:
        """Log optimization progress."""
        print(f"Iteration {self.iteration}: Loss = {loss:.6f}")

    def _save_visualization(self) -> None:
        """Save visualization of current state."""
        if self.config.output_dir is None or self.patterns is None:
            return

        output_path = Path(self.config.output_dir)

        # Save patterns
        fig = self.plot_patterns()
        fig.savefig(output_path / f"patterns_iter_{self.iteration}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Save frequency spectrum
        fig = self.plot_frequency_spectrum()
        fig.savefig(output_path / f"spectrum_iter_{self.iteration}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Save loss curve
        fig = self.plot_loss_curve()
        fig.savefig(output_path / f"loss_iter_{self.iteration}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_patterns(self) -> Figure:
        """Plot current patterns."""
        if self.patterns is None:
            raise RuntimeError("No patterns to plot")

        K, Wp = self.patterns.shape
        patterns_np = self.patterns.detach().cpu().numpy()

        fig, axes = plt.subplots(K, 1, figsize=(10, 2 * K))
        if K == 1:
            axes = [axes]

        for k in range(K):
            axes[k].plot(patterns_np[k], linewidth=1.5)
            axes[k].set_ylim(self.config.min_value, self.config.max_value)
            axes[k].set_ylabel("Intensity")
            axes[k].set_title(f"Pattern {k + 1}")
            axes[k].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Projector Column")
        plt.tight_layout()
        return fig

    def plot_frequency_spectrum(self) -> Figure:
        """Plot frequency spectrum of patterns."""
        if self.patterns is None:
            raise RuntimeError("No patterns to analyze")

        K = self.patterns.shape[0]
        patterns_np = self.patterns.detach().cpu().numpy()

        fig, axes = plt.subplots(K, 1, figsize=(10, 2 * K))
        if K == 1:
            axes = [axes]

        freq_bins = np.fft.rfftfreq(self.patterns.shape[1], d=1.0 / self.patterns.shape[1])

        for k in range(K):
            pattern_fft = np.abs(np.fft.rfft(patterns_np[k]))
            axes[k].plot(freq_bins, pattern_fft, linewidth=1.5)
            axes[k].set_xlabel("Frequency")
            axes[k].set_ylabel("Magnitude")
            axes[k].set_title(f"Pattern {k + 1} Spectrum")
            axes[k].grid(True, alpha=0.3)

            if self.config.max_frequency is not None:
                axes[k].axvline(self.config.max_frequency, color='r', linestyle='--',
                               label='Nyquist Limit')
                axes[k].legend()

        plt.tight_layout()
        return fig

    def plot_loss_curve(self) -> Figure:
        """Plot training loss curve."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(self.loss_history, linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


def finite_difference_gradient(
    renderer: StructuredLightRenderer,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    patterns: torch.Tensor,
    direction: torch.Tensor,
    epsilon: float = 1e-3,
) -> float:
    """
    Compute directional derivative via finite difference.

    This implements the "optical domain" gradient estimation from the paper.

    Args:
        renderer: StructuredLightRenderer
        loss_fn: Loss function that takes patterns and returns scalar loss
        patterns: Current patterns [K, Wp]
        direction: Direction vector [K, Wp]
        epsilon: Perturbation magnitude

    Returns:
        Directional derivative (dL/dpatterns) · direction
    """
    # Loss at current point
    loss0 = loss_fn(patterns)

    # Loss at perturbed point
    loss1 = loss_fn(patterns + epsilon * direction)

    # Finite difference approximation
    return (loss1 - loss0).item() / epsilon


def verify_gradient(
    renderer: StructuredLightRenderer,
    patterns: torch.nn.Parameter,
    tau: float = 50.0,
    num_directions: int = 5,
    epsilon: float = 1e-3,
) -> dict[str, list[float]]:
    """
    Verify autodiff gradient against finite difference.

    Args:
        renderer: StructuredLightRenderer
        patterns: Learnable pattern parameters
        tau: Softmax temperature
        num_directions: Number of random directions to test
        epsilon: Finite difference step size

    Returns:
        Dictionary with autodiff and finite difference values
    """
    device = patterns.device
    dtype = patterns.dtype
    K, Wp = patterns.shape

    # Define loss function
    def loss_fn(p: torch.Tensor) -> torch.Tensor:
        images = renderer.render_images_autodiff(p)
        gt_corr = renderer.gt_corr
        scores = zncc_scores(images, p)
        return soft_correspondence_loss(scores, gt_corr, tau=tau)

    # Compute autodiff gradient
    loss = loss_fn(patterns)
    loss.backward()

    autodiff_grads = []
    fd_grads = []

    for _ in range(num_directions):
        # Random direction
        direction = torch.randn(K, Wp, device=device, dtype=dtype)
        direction = direction / direction.norm()  # Normalize

        # Autodiff directional derivative
        autodiff_dir = (patterns.grad * direction).sum().item()

        # Finite difference directional derivative
        fd_dir = finite_difference_gradient(renderer, loss_fn, patterns.detach(), direction, epsilon)

        autodiff_grads.append(autodiff_dir)
        fd_grads.append(fd_dir)

    return {
        "autodiff": autodiff_grads,
        "finite_difference": fd_grads,
    }


def create_optimizer(
    renderer: StructuredLightRenderer,
    **kwargs,
) -> OpticalSGDOptimizer:
    """
    Convenience function to create an optimizer.

    Args:
        renderer: StructuredLightRenderer
        **kwargs: Arguments for OptimizerConfig

    Returns:
        Initialized optimizer
    """
    config = OptimizerConfig(**kwargs)
    return OpticalSGDOptimizer(renderer, config)
