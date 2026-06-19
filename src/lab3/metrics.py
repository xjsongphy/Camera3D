"""Unified image-quality metrics for Lab 3 novel-view evaluation.

PSNR and SSIM are implemented in pure NumPy so every method is scored with the
*same* metric on the *same* held-out views (assignment §5.1). LPIPS is loaded
lazily: if the optional ``lpips`` package is unavailable we return ``None`` and
log why — the assignment explicitly allows submitting PSNR/SSIM alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

_LPIPS_WARNED = False
_LPIPS_MODEL: Callable | None = None
_LPIPS_AVAILABLE: bool | None = None

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
DATA_RANGE = 1.0


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB for images in [0, 1]; ``inf`` if equal."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if gt.shape != pred.shape:
        raise ValueError(f"PSNR shape mismatch: {gt.shape} vs {pred.shape}")
    mse = float(np.mean((gt - pred) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((DATA_RANGE * DATA_RANGE) / mse))


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    """Mean structural similarity over channels for images in [0, 1]."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if gt.shape != pred.shape:
        raise ValueError(f"SSIM shape mismatch: {gt.shape} vs {pred.shape}")
    c1 = (0.01 * DATA_RANGE) ** 2
    c2 = (0.03 * DATA_RANGE) ** 2
    if gt.ndim == 2:
        return _ssim_single(gt, pred, c1, c2)
    if gt.ndim == 3:
        return float(np.mean([_ssim_single(gt[..., c], pred[..., c], c1, c2) for c in range(gt.shape[2])]))
    raise ValueError(f"SSIM expects 2D or 3D arrays, got {gt.ndim}D")


def _ssim_single(x: np.ndarray, y: np.ndarray, c1: float, c2: float) -> float:
    size = min(SSIM_WINDOW, x.shape[0], x.shape[1])
    if size % 2 == 0:
        size -= 1  # keep an odd, centered window
    size = max(size, 1)
    kernel = _gaussian_kernel1d(size, SSIM_SIGMA)
    mu_x = _filter_valid(x, kernel)
    mu_y = _filter_valid(y, kernel)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = _filter_valid(x * x, kernel) - mu_x2
    sigma_y2 = _filter_valid(y * y, kernel) - mu_y2
    sigma_xy = _filter_valid(x * y, kernel) - mu_xy
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(numerator / denominator))


def _filter_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable 'valid' Gaussian filtering of a 2D image."""
    windows = np.lib.stride_tricks.sliding_window_view(image, kernel.shape[0], axis=0)
    filtered = np.tensordot(windows, kernel, axes=([-1], [0]))
    windows = np.lib.stride_tricks.sliding_window_view(filtered, kernel.shape[0], axis=1)
    return np.tensordot(windows, kernel, axes=([-1], [0]))


def _gaussian_kernel1d(size: int, sigma: float) -> np.ndarray:
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-(coords**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def lpips_score(gt: np.ndarray, pred: np.ndarray) -> float | None:
    """Perceptual LPIPS distance; ``None`` if the package/model is unavailable."""
    model = _get_lpips_model()
    if model is None:
        return None
    import torch  # local import: only needed when lpips is available

    device = next(model.parameters()).device
    gt_t = _to_lpips_tensor(gt, device)
    pred_t = _to_lpips_tensor(pred, device)
    try:
        with torch.no_grad():
            return float(model(gt_t, pred_t).item())
    except Exception as exc:  # e.g. image too small for VGG, device mismatch
        global _LPIPS_AVAILABLE, _LPIPS_WARNED
        _LPIPS_AVAILABLE = False
        if not _LPIPS_WARNED:
            print(f"[lab3.metrics] lpips forward failed ({exc}); reporting PSNR/SSIM only.")
            _LPIPS_WARNED = True
        return None


def compute_image_metrics(
    gt_paths: Iterable[Path],
    pred_paths: Iterable[Path],
    eval_size: tuple[int, int] | None = None,
    lpips_enabled: bool = True,
) -> dict[str, float | int | None]:
    """Compute mean PSNR/SSIM/(LPIPS) over matched GT/prediction image pairs.

    ``eval_size`` (height, width) resamples every image to a common resolution so
    that methods rendered at different native sizes are compared fairly.
    """
    gt_paths = list(gt_paths)
    pred_paths = list(pred_paths)
    if len(gt_paths) != len(pred_paths):
        raise ValueError(f"GT/prediction count mismatch: {len(gt_paths)} vs {len(pred_paths)}")

    psnrs: list[float] = []
    ssims: list[float] = []
    lpipss: list[float] = []
    for gt_path, pred_path in zip(gt_paths, pred_paths):
        gt_img = load_image(gt_path, eval_size)
        pred_img = load_image(pred_path, eval_size)
        psnrs.append(psnr(gt_img, pred_img))
        ssims.append(ssim(gt_img, pred_img))
        score = lpips_score(gt_img, pred_img) if lpips_enabled else None
        if score is not None:
            lpipss.append(score)

    return {
        "psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
        "ssim": float(np.mean(ssims)) if ssims else float("nan"),
        "lpips": float(np.mean(lpipss)) if lpipss else None,
        "n": len(psnrs),
    }


def load_image(path: Path | str, eval_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an image as float64 RGB in [0, 1], optionally resized to ``eval_size``."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if eval_size is not None:
            image = image.resize((eval_size[1], eval_size[0]), Image.BICUBIC)
        return np.asarray(image, dtype=np.float64) / 255.0


def _to_lpips_tensor(image: np.ndarray, device=None):
    import torch

    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _get_lpips_model() -> Callable | None:
    """Lazily build (and cache) an LPIPS model; ``None`` if unavailable.

    Covers both the missing-package case and the offline case where the package
    is installed but cannot download its pretrained VGG/linear weights.
    """
    global _LPIPS_AVAILABLE, _LPIPS_MODEL, _LPIPS_WARNED
    if _LPIPS_AVAILABLE is False:
        return None
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL
    try:
        import lpips  # type: ignore
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = lpips.LPIPS(net="vgg").to(device).eval()
    except Exception as exc:  # pragma: no cover - environment dependent
        _LPIPS_AVAILABLE = False
        if not _LPIPS_WARNED:
            print(f"[lab3.metrics] lpips unavailable ({exc}); reporting PSNR/SSIM only.")
            _LPIPS_WARNED = True
        return None
    _LPIPS_AVAILABLE = True
    _LPIPS_MODEL = model
    return model
