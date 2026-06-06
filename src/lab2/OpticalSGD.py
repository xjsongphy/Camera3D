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

from .shader import StructuredLightRenderer
from . import decoder
from . import losses
from .common import prepare_decoder_images


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
    gradient_mode: Literal["autodiff", "finite_difference"] = "autodiff"
    fd_num_coords: int = 32
    fd_epsilon: float = 1e-2
    fd_seed_base: int = 0

    # Decoder configuration
    decoder_type: DecoderType = DecoderType.ZNCC
    neighborhood_size: int = 1  # p=1, 3, or 5
    use_projector_response_curve: bool = False

    # Loss function
    penalty: str = "l1"  # "l1", "zero_tolerance", "one_tolerance"

    # Pattern constraints
    min_value: float = 0.0
    max_value: float = 1.0
    max_frequency: float | None = None  # Nyquist frequency constraint
    frequency_weight: float = 1.0

    # Logging
    log_interval: int = 10
    save_interval: int = 50
    output_dir: str | Path | None = None
    optimizer_name: Literal["adam", "rmsprop"] = "adam"
    lr_decay_step: int = 0
    lr_decay_gamma: float = 1.0
    projector_height: int = 1080


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
        self.history_iterations: list[int] = []
        self.pattern_history: list[torch.Tensor] = []
        self.decoder_param_history: dict[str, list[torch.Tensor]] = {}

        # Ensure output directory exists
        if self.config.output_dir is not None:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        if self.config.gradient_mode == "finite_difference" and self.config.decoder_type != DecoderType.ZNCC:
            raise ValueError("Finite-difference training is currently supported only for the ZNCC decoder.")

    def initialize_patterns(
        self,
        num_patterns: int,
        projector_width: int,
        init_mode: str = "random",
        initial_patterns: torch.Tensor | None = None,
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
        if initial_patterns is None:
            patterns = initialize_patterns(
                num_patterns,
                projector_width,
                self.renderer.device,
                self.renderer.dtype,
                init_mode,
            )
        else:
            expected_shape = (num_patterns, projector_width)
            if tuple(initial_patterns.shape) != expected_shape:
                raise ValueError(
                    f"Initial patterns shape mismatch: expected {expected_shape}, "
                    f"got {tuple(initial_patterns.shape)}"
                )
            patterns = initial_patterns.to(
                device=self.renderer.device,
                dtype=self.renderer.dtype,
            ).clone()

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
                use_projector_response_curve=self.config.use_projector_response_curve,
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
        self._record_history()

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

        images = prepare_decoder_images(images)

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
            freq_penalty = self.config.frequency_weight * losses.frequency_penalty(
                self.patterns,
                self.config.max_frequency,
            )
            return corr_loss + freq_penalty

        return corr_loss

    def _compute_freq_penalty_grad(self) -> tuple[torch.Tensor | None, float]:
        if self.patterns is None or self.config.max_frequency is None:
            return None, 0.0

        patterns = self.patterns.detach().clone().requires_grad_(True)
        freq_penalty = self.config.frequency_weight * losses.frequency_penalty(
            patterns,
            self.config.max_frequency,
        )
        (grad,) = torch.autograd.grad(freq_penalty, patterns, retain_graph=False, create_graph=False)
        return grad.detach(), float(freq_penalty.item())

    def _compute_fd_corr_loss(self, patterns: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("Decoder must be initialized before computing loss")
        with torch.no_grad():
            images = self.renderer.render_images(patterns)
            images = images.mean(dim=-1) if images.ndim == 4 and images.shape[-1] == 3 else images
            scores = self.decoder(images, patterns)
            return losses.soft_correspondence_loss(
                scores,
                self.renderer.gt_corr,
                tau=self.config.tau,
                penalty=self.config.penalty,
            )

    def _compute_finite_difference_pattern_grad(self) -> tuple[torch.Tensor, float]:
        if self.patterns is None or self.decoder is None:
            raise RuntimeError("Patterns and decoder must be initialized before optimization")

        patterns = self.patterns.detach()
        device = self.renderer.device
        num_patterns, projector_width = patterns.shape
        total_coords = num_patterns * projector_width
        sample_count = min(max(1, int(self.config.fd_num_coords)), total_coords)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.fd_seed_base + self.iteration))
        flat_indices = torch.randperm(total_coords, generator=generator)[:sample_count].tolist()

        direction = torch.zeros_like(patterns)
        for flat_idx in flat_indices:
            pattern_idx = flat_idx // projector_width
            column_idx = flat_idx % projector_width
            direction[pattern_idx, column_idx] = 1.0 if torch.rand((), generator=generator).item() > 0.5 else -1.0

        epsilon = float(self.config.fd_epsilon)
        patterns_pos = (patterns + epsilon * direction).clamp(self.config.min_value, self.config.max_value)
        patterns_neg = (patterns - epsilon * direction).clamp(self.config.min_value, self.config.max_value)
        loss_pos = self._compute_fd_corr_loss(patterns_pos)
        loss_neg = self._compute_fd_corr_loss(patterns_neg)

        gradient = direction * ((loss_pos - loss_neg) / (2.0 * epsilon)) * (total_coords / sample_count)

        freq_grad, freq_penalty = self._compute_freq_penalty_grad()
        if freq_grad is not None:
            gradient = gradient + freq_grad

        total_loss = float((0.5 * (loss_pos + loss_neg)).item()) + freq_penalty
        return gradient.to(device=device, dtype=self.renderer.dtype), total_loss

    def step(self) -> float:
        """
        Perform one optimization step.

        Returns:
            Loss value for this step
        """
        if self.patterns is None or self.optimizer is None:
            raise RuntimeError("Patterns must be initialized before optimization")

        self.optimizer.zero_grad()
        if self.config.gradient_mode == "autodiff":
            images = self.renderer.render_images_autodiff(self.patterns)
            gt_corr = self.renderer.gt_corr
            loss = self.compute_loss(images, gt_corr)
            loss.backward()
            with torch.no_grad():
                loss_value = float(loss.item())
        else:
            grad, loss_value = self._compute_finite_difference_pattern_grad()
            self.patterns.grad = grad
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
        self.loss_history.append(loss_value)
        self._record_history()

        # Logging
        if self.config.log_interval > 0 and self.iteration % self.config.log_interval == 0:
            self._log_progress(loss_value)

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
        if self.config.decoder_type == DecoderType.ZNCC_NN and self.config.use_projector_response_curve:
            decoder_name = f"{decoder_name}_response"
        print(f"Iteration {self.iteration}: Loss = {loss:.6f} ({decoder_name})")

    def _record_history(self) -> None:
        """Record full optimization history for downstream analysis."""
        if self.patterns is None:
            return

        self.history_iterations.append(self.iteration)
        self.pattern_history.append(self.patterns.detach().cpu().clone())

        if self.decoder is None:
            return

        for name, param in self.decoder.named_parameters():
            self.decoder_param_history.setdefault(name, []).append(
                param.detach().cpu().clone()
            )


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
