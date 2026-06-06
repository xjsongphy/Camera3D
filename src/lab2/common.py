from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .shader import CameraConfig, ProjectorConfig


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


def create_timestamped_output_dir(base_dir: str, subdir: str, name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / subdir / f"{timestamp}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def prepare_decoder_images(images: torch.Tensor) -> torch.Tensor:
    """
    Convert renderer outputs to decoder input shape [K, H, W].

    Mitsuba rendering may return RGB images [K, H, W, 3], while decoders expect
    grayscale observations [K, H, W]. This helper keeps both Mitsuba forward
    rendering and Mitsuba autodiff rendering on the same decoder interface.
    """
    if images.ndim == 3:
        return images
    if images.ndim == 4 and images.shape[-1] == 3:
        return images.mean(dim=-1)
    raise ValueError(f"Unsupported image tensor shape for decoder: {tuple(images.shape)}")


def correspondence_to_depth(
    corr: torch.Tensor,
    camera: CameraConfig,
    projector: ProjectorConfig,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Triangulate depth from structured-light correspondence.

    For each camera pixel (u, v), the predicted correspondence ``corr[u,v]``
    specifies which projector column observes the same 3D point.  This function
    computes the ray-plane intersection between the camera ray at (u, v) and
    the projector plane defined by column ``corr[u,v]``, and returns the
    depth (Z in camera coordinates) of the resulting 3D point.

    Args:
        corr: Correspondence map [H, W] — projector column per camera pixel.
        camera: Camera configuration with intrinsics ``(fx, fy, cx, cy)`` and
            extrinsics ``R`` (world → camera) and ``t`` (world → camera).
        projector: Projector configuration with the same intrinsic/extrinsic
            convention as the camera.
        valid_mask: Optional bool mask [H, W].  If provided, only pixels where
            ``valid_mask`` is True produce valid depth; all others are set to
            NaN.  If omitted, the mask is ``isfinite(corr)``.

    Returns:
        Depth map [H, W] in camera Z-coordinate units.  Invalid pixels are NaN.
    """
    H, W = corr.shape
    device = corr.device
    dtype = corr.dtype

    # ---- pixel grid -------------------------------------------------------
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )

    # ---- camera ray in world frame ----------------------------------------
    # Camera-frame ray: X = (u - cx) / fx, Y = -(v - cy) / fy, Z = 1
    dirs_cam = torch.stack([
        (u - camera.cx) / camera.fx,
        -(v - camera.cy) / camera.fy,
        torch.ones_like(u),
    ], dim=-1)
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)

    R_cw = camera.R.to(device=device, dtype=dtype)       # world → camera
    t_cw = camera.t.to(device=device, dtype=dtype)
    cam_origin = -(R_cw.T @ t_cw)                         # camera origin in world
    dirs_world = dirs_cam @ R_cw                          # [H, W, 3]

    # ---- projector origin in world ---------------------------------------
    R_pw = projector.R.to(device=device, dtype=dtype)
    t_pw = projector.t.to(device=device, dtype=dtype)
    proj_origin = -(R_pw.T @ t_pw)

    # ---- projector plane from correspondence ------------------------------
    # A point projects to column "col" in the projector image when
    #   fx_p * X_proj - (col - cx_p) * Z_proj = 0.
    # The plane normal in projector frame is (fx_p, 0, -(col - cx_p)).
    col = corr
    n_proj = torch.stack([
        torch.full_like(col, projector.fx),
        torch.zeros_like(col),
        -(col - projector.cx),
    ], dim=-1)                                             # [H, W, 3]

    # Transform plane normal to world frame: n_world = R_pw^T @ n_proj
    n_world = n_proj @ R_pw                                # [H, W, 3]

    # ---- ray-plane intersection -------------------------------------------
    #   n_world · (cam_origin + t * dirs_world - proj_origin) = 0
    #   t = - n_world · (cam_origin - proj_origin) / (n_world · dirs_world)
    oc = cam_origin.view(1, 1, 3) - proj_origin.view(1, 1, 3)
    n_dot_oc = (n_world * oc).sum(dim=-1)                  # [H, W]
    n_dot_dir = (n_world * dirs_world).sum(dim=-1)         # [H, W]

    t = -n_dot_oc / torch.where(n_dot_dir.abs() > 1e-8, n_dot_dir, torch.full_like(n_dot_dir, 1e-8))

    # ---- 3D point and Z-depth in camera frame -----------------------------
    pts = cam_origin.view(1, 1, 3) + t[..., None] * dirs_world  # [H, W, 3]
    pts_cam = pts @ R_cw.T + t_cw.view(1, 1, 3)           # world → camera
    depth = pts_cam[..., 2]                                # [H, W]

    # ---- validity ---------------------------------------------------------
    if valid_mask is None:
        valid_mask = torch.isfinite(corr)
    valid_mask = valid_mask & (t > 0) & (depth > 0)

    depth = depth.where(valid_mask, torch.full_like(depth, float("nan")))
    return depth
