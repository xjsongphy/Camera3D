"""
Structured Light Decoders: ZNCC and ZNCC-NN.

This module implements differentiable decoders for structured light correspondence.
Following the Optical SGD paper (Chen et al., CVPR 2020).

Key design principle:
- Decoder outputs scores [H, W, Wp], NOT hard correspondences
- Training uses soft correspondence loss with softmax approximation (see losses.py)
- Evaluation uses hard_decode (argmax) on scores
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    """
    Residual MLP with two fully-connected layers and ReLU activation.

    Architecture: x -> Linear -> ReLU -> Linear -> + x
    """

    def __init__(self, dim: int) -> None:
        """
        Initialize residual MLP.

        Args:
            dim: Feature dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual MLP.

        Args:
            x: Input tensor [N, dim]

        Returns:
            Output tensor [N, dim] with residual connection
        """
        return x + self.net(x)


def extract_camera_features(images: torch.Tensor, p: int) -> torch.Tensor:
    """
    Extract neighborhood features from camera images.

    For each pixel, collect its (1 x p) horizontal neighborhood across all K images.
    Uses toroidal (wrap-around) boundary conditions via torch.roll.

    Args:
        images: Camera images [K, H, W]
        p: Neighborhood size (odd number: 1, 3, 5, ...)

    Returns:
        Camera features [H, W, p*K]
        Each pixel's feature is a concatenation of its neighborhood values across all K images
    """
    K, H, W = images.shape
    r = p // 2

    # [K, H, W] -> [H, W, K] for easier roll operations
    img = images.permute(1, 2, 0)  # [H, W, K]

    feats = []
    for dx in range(-r, r + 1):
        # Shift horizontally by dx pixels (wrap-around)
        shifted = torch.roll(img, shifts=dx, dims=1)  # [H, W, K]
        feats.append(shifted)

    # Concatenate along feature dimension: [H, W, p*K]
    return torch.cat(feats, dim=-1)


def extract_projector_features(patterns: torch.Tensor, p: int) -> torch.Tensor:
    """
    Extract neighborhood features from projector patterns.

    For each column, collect its p-column neighborhood across all K patterns.
    Uses toroidal (wrap-around) boundary conditions via torch.roll.

    Args:
        patterns: Projection patterns [K, Wp]
        p: Neighborhood size (odd number: 1, 3, 5, ...)

    Returns:
        Projector features [Wp, p*K]
        Each column's feature is a concatenation of its neighborhood values across all K patterns
    """
    K, Wp = patterns.shape
    r = p // 2

    # [K, Wp] -> [Wp, K] for easier roll operations
    pat = patterns.T  # [Wp, K]

    feats = []
    for dx in range(-r, r + 1):
        # Shift horizontally by dx columns (wrap-around)
        shifted = torch.roll(pat, shifts=dx, dims=0)  # [Wp, K]
        feats.append(shifted)

    # Concatenate along feature dimension: [Wp, p*K]
    return torch.cat(feats, dim=-1)


def zncc_matrix(obs_feat: torch.Tensor, ref_feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute ZNCC (Zero-mean Normalized Cross-Correlation) matrix.

    For each observation vector m and reference vector n:
        ZNCC[m, n] = (obs_m - mean(obs_m)) · (ref_n - mean(ref_n))
                     / (||obs_m - mean(obs_m)|| * ||ref_n - mean(ref_n)||)

    Args:
        obs_feat: Observation features [M, D] (M = H*W camera pixels)
        ref_feat: Reference features [Wp, D] (Wp projector columns)
        eps: Small constant for numerical stability

    Returns:
        ZNCC score matrix [M, Wp]
    """
    # Normalize observations: zero-mean and unit-norm
    obs = obs_feat - obs_feat.mean(dim=-1, keepdim=True)
    obs = obs / (obs.norm(dim=-1, keepdim=True) + eps)

    # Normalize references: zero-mean and unit-norm
    ref = ref_feat - ref_feat.mean(dim=-1, keepdim=True)
    ref = ref / (ref.norm(dim=-1, keepdim=True) + eps)

    # Compute ZNCC matrix via matrix multiplication
    return obs @ ref.T


class ZNCCDecoder(nn.Module):
    """
    ZNCC-based decoder for structured light correspondence.

    Supports neighborhood-based matching:
    - p=1: Standard ZNCC on single pixels
    - p=3: ZNCC on 3-pixel horizontal neighborhood
    - p=5: ZNCC on 5-pixel horizontal neighborhood

    The decoder outputs matching scores, not hard correspondences.
    Use soft_correspondence_loss for training and hard_decode for evaluation.
    """

    def __init__(self, neighborhood_size: int = 1, eps: float = 1e-6) -> None:
        """
        Initialize ZNCC decoder.

        Args:
            neighborhood_size: Size of horizontal neighborhood (1, 3, or 5)
            eps: Small constant for numerical stability in ZNCC computation
        """
        super().__init__()
        self.p = neighborhood_size
        self.eps = eps

    def forward(self, images: torch.Tensor, patterns: torch.Tensor) -> torch.Tensor:
        """
        Compute ZNCC scores between all camera pixels and projector columns.

        Args:
            images: Camera images [K, H, W]
            patterns: Projection patterns [K, Wp]

        Returns:
            ZNCC scores [H, W, Wp]
            scores[u, v, x] represents the similarity between camera pixel (u,v)
            and projector column x
        """
        K, H, W = images.shape
        _, Wp = patterns.shape

        # Extract neighborhood features
        cam_feat = extract_camera_features(images, self.p)       # [H, W, p*K]
        proj_feat = extract_projector_features(patterns, self.p) # [Wp, p*K]

        # Reshape camera features for matrix computation
        cam_feat = cam_feat.reshape(H * W, self.p * K)  # [H*W, p*K]

        # Compute ZNCC matrix
        scores = zncc_matrix(cam_feat, proj_feat, eps=self.eps)  # [H*W, Wp]

        # Reshape back to spatial dimensions
        return scores.reshape(H, W, Wp)


class ZNCCNNDecoder(nn.Module):
    """
    ZNCC-NN decoder with learnable residual MLPs.

    Extends ZNCCDecoder by adding learnable transformations on both
    camera and projector features before computing ZNCC.

    Architecture:
        camera_feat: extract_camera_features()
        projector_feat: extract_projector_features()
        cam_feat' = camera_net(camera_feat)
        proj_feat' = projector_net(projector_feat)
        scores = ZNCC(cam_feat', proj_feat')

    This allows the decoder to learn robust feature representations
    that are invariant to noise, system non-linearities, and material properties.
    """

    def __init__(
        self,
        num_patterns: int,
        neighborhood_size: int = 3,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize ZNCC-NN decoder.

        Args:
            num_patterns: Number of projection patterns K
            neighborhood_size: Size of horizontal neighborhood (1, 3, or 5)
            eps: Small constant for numerical stability in ZNCC computation
        """
        super().__init__()
        self.K = num_patterns
        self.p = neighborhood_size
        self.D = num_patterns * neighborhood_size
        self.eps = eps

        # Learnable residual networks
        self.camera_net = ResidualMLP(self.D)
        self.projector_net = ResidualMLP(self.D)

    def forward(self, images: torch.Tensor, patterns: torch.Tensor) -> torch.Tensor:
        """
        Compute ZNCC-NN scores with learnable feature transformations.

        Args:
            images: Camera images [K, H, W]
            patterns: Projection patterns [K, Wp]

        Returns:
            ZNCC-NN scores [H, W, Wp]
        """
        K, H, W = images.shape
        _, Wp = patterns.shape

        # Extract neighborhood features
        cam_feat = extract_camera_features(images, self.p)       # [H, W, D]
        proj_feat = extract_projector_features(patterns, self.p) # [Wp, D]

        # Reshape for MLP processing
        cam_feat = cam_feat.reshape(H * W, self.D)  # [H*W, D]
        proj_feat = proj_feat.reshape(Wp, self.D)   # [Wp, D]

        # Apply learnable transformations
        cam_feat = self.camera_net(cam_feat)
        proj_feat = self.projector_net(proj_feat)

        # Compute ZNCC matrix on transformed features
        scores = zncc_matrix(cam_feat, proj_feat, eps=self.eps)  # [H*W, Wp]

        # Reshape back to spatial dimensions
        return scores.reshape(H, W, Wp)


def hard_decode(scores: torch.Tensor) -> torch.Tensor:
    """
    Decode correspondence by taking argmax of scores.

    Use this for evaluation and visualization only (not for training).

    Args:
        scores: ZNCC scores [H, W, Wp]

    Returns:
        Predicted correspondence map [H, W]
        pred_corr[u, v] = argmax_x scores[u, v, x]
    """
    return scores.argmax(dim=-1).to(scores.dtype)


def correspondence_metrics(
    pred_corr: torch.Tensor,
    gt_corr: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute correspondence error metrics.

    Args:
        pred_corr: Predicted correspondence [H, W]
        gt_corr: Ground truth correspondence [H, W], invalid pixels marked as NaN

    Returns:
        Dictionary of metrics:
            - "mae": Mean Absolute Error
            - "rmse": Root Mean Squared Error
            - "acc_0": Percentage of pixels with error <= 0.5
            - "acc_1": Percentage of pixels with error <= 1.0
            - "acc_2": Percentage of pixels with error <= 2.0
    """
    valid = torch.isfinite(gt_corr)
    err = (pred_corr - gt_corr).abs()

    return {
        "mae": err[valid].mean(),
        "rmse": torch.sqrt((err[valid] ** 2).mean()),
        "acc_0": (err[valid] <= 0.5).float().mean(),
        "acc_1": (err[valid] <= 1.0).float().mean(),
        "acc_2": (err[valid] <= 2.0).float().mean(),
    }
