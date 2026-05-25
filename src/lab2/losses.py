"""
Loss functions for structured light optimization.

This module contains all loss-related functions:
- Correspondence loss (soft/hard)
- Frequency constraint penalty
- Pattern constraints
"""

from __future__ import annotations

import torch


def soft_correspondence_loss(
    scores: torch.Tensor,
    gt_corr: torch.Tensor,
    tau: float = 100.0,
    penalty: str = "l1",
) -> torch.Tensor:
    """
    Compute soft correspondence loss using softmax approximation.

    Instead of using non-differentiable argmax:
        pred_corr[m] = argmax_n scores[m, n]
    We use soft expected penalty:
        loss = E_n[softmax(tau * scores[m, n]) * penalty(n - gt_corr[m])]

    This makes the entire pipeline end-to-end differentiable.

    Args:
        scores: ZNCC scores [H, W, Wp]
        gt_corr: Ground truth correspondence [H, W], invalid pixels marked as NaN
        tau: Softmax temperature (higher = sharper approximation to argmax)
        penalty: Penalty function type
            - "l1": L1 penalty |n - gt[m]|
            - "zero_tolerance": 0 if |n - gt[m]| <= 0.5, else 1
            - "one_tolerance": 0 if |n - gt[m]| <= 1.0, else 1

    Returns:
        Scalar loss value (mean over valid pixels)
    """
    H, W, Wp = scores.shape
    device = scores.device
    dtype = scores.dtype

    # Valid pixel mask
    valid = torch.isfinite(gt_corr)

    if not valid.any():
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Projector column indices [0, 1, ..., Wp-1]
    index = torch.arange(Wp, device=device, dtype=dtype)

    # Compute penalty matrix: |n - gt_corr[m]| for all m, n
    # Shape: [H, W, Wp]
    diff = index[None, None, :] - gt_corr[..., None]

    if penalty == "l1":
        err = diff.abs()
    elif penalty == "zero_tolerance":
        err = (diff.abs() > 0.5).to(dtype)
    elif penalty == "one_tolerance":
        err = (diff.abs() > 1.0).to(dtype)
    else:
        raise ValueError(f"Unknown penalty: {penalty}")

    # Softmax over projector columns (temperature-scaled)
    # prob[m, n] = softmax(tau * scores[m, n]) over n
    prob = torch.softmax(tau * scores, dim=-1)  # [H, W, Wp]

    # Expected penalty at each pixel
    # loss_map[m] = sum_n prob[m, n] * err[m, n]
    loss_map = (prob * err).sum(dim=-1)  # [H, W]

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


def apply_value_constraints(
    patterns: torch.Tensor,
    min_value: float,
    max_value: float,
) -> torch.Tensor:
    """
    Apply value constraints by clamping.

    Args:
        patterns: Patterns to constrain [K, Wp]
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Value-constrained patterns [K, Wp]
    """
    return patterns.clamp(min_value, max_value)
