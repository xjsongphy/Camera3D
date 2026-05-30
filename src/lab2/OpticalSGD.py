"""
OpticalSGD optimizer for structured light pattern optimization.

This module implements the Optical SGD optimizer from Chen et al., CVPR 2020.
Core insight: use softmax to approximate argmax, making the entire pipeline differentiable.

Key architectural change:
- Decoders (ZNCCDecoder, ZNCCNNDecoder) are separate from the optimizer
- Optimizer only handles: forward, backward, optimizer.step(), constraints
- Loss computation is delegated to decoder.soft_correspondence_loss()
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .shader import StructuredLightRenderer
from . import decoder
from . import losses


class DecoderType(Enum):
    """Decoder type enumeration."""
    ZNCC = "zncc"
    ZNCC_NN = "zncc_nn"


@dataclass
class OptimizerConfig:
    """Configuration for OpticalSGD optimizer."""

    # Optimization parameters
    num_iterations: int = 100
    learning_rate: float = 0.01
    decoder_learning_rate: float = 0.01  # Learning rate for decoder parameters
    tau: float = 50.0  # Softmax temperature for correspondence

    # Decoder configuration
    decoder_type: DecoderType = DecoderType.ZNCC
    neighborhood_size: int = 1  # p=1, 3, or 5

    # Loss function
    penalty: str = "l1"  # "l1", "zero_tolerance", "one_tolerance"

    # Pattern constraints
    min_value: float = 0.0
    max_value: float = 1.0
    max_frequency: float | None = None  # Nyquist frequency constraint

    # Logging and visualization
    log_interval: int = 10
    save_interval: int = 50
    output_dir: str | Path | None = None
    optimizer_name: Literal["adam", "rmsprop"] = "adam"
    lr_decay_step: int = 0
    lr_decay_gamma: float = 1.0
    projector_height: int = 1080  # For 2D pattern visualization


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
        init_mode: Initialization mode ("random", "sine", "constant", "stripe")

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
    Optical SGD optimizer with pluggable decoders.

    Uses softmax-approximated correspondence loss for end-to-end optimization.
    Supports both ZNCC and ZNCC-NN decoders.
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
        self.decoder: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
        self.iteration = 0
        self.loss_history: list[float] = []
        self.pattern_snapshots: list[torch.Tensor] = []  # Store pattern snapshots for GIF

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
        Initialize patterns and decoder.

        Args:
            num_patterns: Number of patterns K
            projector_width: Projector width Wp
            init_mode: Initialization mode

        Returns:
            Pattern parameter
        """
        # Initialize patterns
        patterns = initialize_patterns(
            num_patterns,
            projector_width,
            self.renderer.device,
            self.renderer.dtype,
            init_mode,
        )

        # Make patterns learnable
        self.patterns = torch.nn.Parameter(patterns)

        # Create decoder based on configuration
        if self.config.decoder_type == DecoderType.ZNCC:
            self.decoder = decoder.ZNCCDecoder(
                neighborhood_size=self.config.neighborhood_size,
            )
        elif self.config.decoder_type == DecoderType.ZNCC_NN:
            self.decoder = decoder.ZNCCNNDecoder(
                num_patterns=num_patterns,
                neighborhood_size=self.config.neighborhood_size,
            )
        else:
            raise ValueError(f"Unknown decoder type: {self.config.decoder_type}")

        # Move decoder to same device as renderer
        self.decoder.to(self.renderer.device)

        # Setup optimizer with separate learning rates
        # Patterns have two gradient paths:
        # 1. patterns -> renderer -> images -> decoder -> loss
        # 2. patterns -> projector feature/codebook -> decoder -> loss
        if self.config.optimizer_name == "rmsprop":
            opt_ctor = torch.optim.RMSprop
        else:
            opt_ctor = torch.optim.Adam

        if self.config.decoder_type == DecoderType.ZNCC_NN:
            self.optimizer = opt_ctor([
                {"params": [self.patterns], "lr": self.config.learning_rate},
                {"params": self.decoder.parameters(), "lr": self.config.decoder_learning_rate},
            ])
        else:
            # ZNCC decoder has no learnable parameters
            self.optimizer = opt_ctor(
                [self.patterns], lr=self.config.learning_rate
            )

        if self.config.lr_decay_step > 0 and self.config.lr_decay_gamma != 1.0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_decay_step,
                gamma=self.config.lr_decay_gamma,
            )
        else:
            self.scheduler = None

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
        if self.decoder is None:
            raise RuntimeError("Decoder must be initialized before computing loss")

        # Compute ZNCC/ZNCC-NN scores
        scores = self.decoder(images, self.patterns)  # [H, W, Wp]

        # Soft correspondence loss
        corr_loss = losses.soft_correspondence_loss(
            scores, gt_corr,
            tau=self.config.tau,
            penalty=self.config.penalty,
        )

        # Frequency penalty (if configured)
        if self.config.max_frequency is not None:
            freq_penalty = losses.frequency_penalty(self.patterns, self.config.max_frequency)
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
        if self.scheduler is not None:
            self.scheduler.step()

        # Apply constraints (value clamping, frequency filtering)
        with torch.no_grad():
            self.patterns.data = self.patterns.data.clamp(
                self.config.min_value, self.config.max_value
            )

            if self.config.max_frequency is not None:
                self.patterns.data = losses.apply_frequency_constraints(
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
        decoder_name = self.config.decoder_type.value if self.config.decoder_type else "none"
        print(f"Iteration {self.iteration}: Loss = {loss:.6f} ({decoder_name})")

    def _save_visualization(self) -> None:
        """Save visualization of current state."""
        if self.config.output_dir is None or self.patterns is None:
            return

        output_path = Path(self.config.output_dir)

        # Save pattern snapshot for GIF
        self.pattern_snapshots.append(self.patterns.detach().cpu().clone())

        # Save patterns as 2D visualization
        fig = self.plot_patterns()
        fig.savefig(output_path / f"patterns_iter_{self.iteration}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Save frequency spectrum
        fig = self.plot_frequency_spectrum()
        fig.savefig(output_path / f"spectrum_iter_{self.iteration}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_patterns(self) -> Figure:
        """Plot current patterns as 2D matrices (QR-code style)."""
        if self.patterns is None:
            raise RuntimeError("No patterns to plot")

        K, Wp = self.patterns.shape
        Hp = self.config.projector_height
        patterns_np = self.patterns.detach().cpu().numpy()

        # Reshape 1D patterns to 2D by tiling vertically
        patterns_2d = np.tile(patterns_np[:, None, :], (1, Hp, 1))  # [K, Hp, Wp]

        fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
        if K == 1:
            axes = [axes]

        for k in range(K):
            im = axes[k].imshow(patterns_2d[k], cmap='gray', vmin=self.config.min_value, vmax=self.config.max_value)
            axes[k].set_title(f"Pattern {k + 1}")
            axes[k].axis('off')
            plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

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
        decoder_name = self.config.decoder_type.value if self.config.decoder_type else "none"
        ax.set_title(f"Training Loss ({decoder_name})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def generate_pattern_evolution_gif(self) -> None:
        """Generate GIF from pattern evolution snapshots."""
        if not self.pattern_snapshots or self.config.output_dir is None:
            return

        try:
            import imageio.v3 as iio
        except ImportError:
            print("  Warning: imageio not available, skipping GIF generation")
            return

        output_path = Path(self.config.output_dir)
        K, Wp = self.pattern_snapshots[0].shape
        Hp = self.config.projector_height

        # Generate frames for GIF
        frames = []
        for i, pattern_snapshot in enumerate(self.pattern_snapshots):
            patterns_np = pattern_snapshot.numpy()
            patterns_2d = np.tile(patterns_np[:, None, :], (1, Hp, 1))  # [K, Hp, Wp]

            # Create combined visualization of all patterns
            n_cols = min(K, 5)
            n_rows = (K + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
            if K == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            for k in range(K):
                row, col = k // n_cols, k % n_cols
                axes[row, col].imshow(patterns_2d[k], cmap='gray', vmin=0.0, vmax=1.0)
                axes[row, col].set_title(f"P{k + 1}", fontsize=10)
                axes[row, col].axis('off')

            # Hide empty subplots
            for k in range(K, n_rows * n_cols):
                row, col = k // n_cols, k % n_cols
                axes[row, col].axis('off')

            fig.suptitle(f"Iteration {i + 1}", fontsize=12)
            plt.tight_layout()

            # Save frame to memory
            from io import BytesIO
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            frame = iio.imread(buf)
            frames.append(frame)
            plt.close(fig)
            buf.close()

        # Save as GIF
        if frames:
            iio.imwrite(output_path / "pattern_evolution.gif", frames, duration=500, loop=0)
            print(f"  Saved pattern evolution GIF: {output_path / 'pattern_evolution.gif'}")


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
