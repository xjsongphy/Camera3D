from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lab3.metrics import compute_image_metrics, psnr, ssim


def test_psnr_identical_images_is_infinite() -> None:
    img = np.zeros((8, 8, 3), dtype=np.float64)
    assert psnr(img, img) == np.inf


def test_psnr_known_mse_matches_formula() -> None:
    # gt zeros, pred = 0.1 everywhere -> MSE = 0.01 -> PSNR = 10*log10(1/0.01) = 20 dB
    gt = np.zeros((8, 8, 3), dtype=np.float64)
    pred = np.full((8, 8, 3), 0.1, dtype=np.float64)
    assert psnr(gt, pred) == pytest.approx(20.0, abs=1e-6)


def test_psnr_max_difference_is_zero_db() -> None:
    gt = np.zeros((8, 8, 3), dtype=np.float64)
    pred = np.ones((8, 8, 3), dtype=np.float64)
    assert psnr(gt, pred) == pytest.approx(0.0, abs=1e-6)


def test_ssim_identical_images_is_one() -> None:
    rng = np.random.default_rng(0)
    img = rng.random((32, 32, 3)).astype(np.float64)
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_ssym_is_symmetric() -> None:
    rng = np.random.default_rng(1)
    a = rng.random((32, 32)).astype(np.float64)
    b = rng.random((32, 32)).astype(np.float64)
    assert ssim(a, b) == pytest.approx(ssim(b, a), abs=1e-6)


def test_ssim_decreases_with_more_noise() -> None:
    rng = np.random.default_rng(2)
    gt = rng.random((32, 32)).astype(np.float64)
    mild = np.clip(gt + rng.normal(0, 0.05, gt.shape), 0, 1)
    heavy = np.clip(gt + rng.normal(0, 0.4, gt.shape), 0, 1)
    assert ssim(gt, mild) > ssim(gt, heavy)


def test_ssim_stays_in_unit_range() -> None:
    rng = np.random.default_rng(3)
    a = rng.random((32, 32)).astype(np.float64)
    b = rng.random((32, 32)).astype(np.float64)
    value = ssim(a, b)
    assert -1.0 <= value <= 1.0


def _write_gradient(path: Path, size: int = 64, offset: float = 0.0) -> None:
    base = np.linspace(0, 1, size, dtype=np.float32)
    grad = np.tile(base[None, :, None], (size, 1, 3))
    grad = np.clip(grad + offset, 0, 1)
    Image.fromarray((grad * 255).astype(np.uint8)).save(path)


def test_compute_image_metrics_returns_expected_keys(tmp_path: Path) -> None:
    gt = tmp_path / "gt.png"
    pred = tmp_path / "pred.png"
    _write_gradient(gt, offset=0.0)
    _write_gradient(pred, offset=0.1)

    result = compute_image_metrics([gt], [pred])

    assert set(result) >= {"psnr", "ssim", "lpips", "n"}
    assert result["n"] == 1
    assert np.isfinite(result["psnr"])
    assert result["psnr"] >= 0.0
    assert 0.0 <= result["ssim"] <= 1.0
    # lpips is optional: either a non-negative float or None when unavailable
    assert result["lpips"] is None or result["lpips"] >= 0.0


def test_compute_image_metrics_identical_pair_is_perfect(tmp_path: Path) -> None:
    gt = tmp_path / "gt.png"
    pred = tmp_path / "pred.png"
    _write_gradient(gt)
    _write_gradient(pred)

    result = compute_image_metrics([gt], [pred])

    assert result["psnr"] == np.inf
    assert result["ssim"] == pytest.approx(1.0, abs=1e-3)


def test_compute_image_metrics_resizes_to_eval_size(tmp_path: Path) -> None:
    gt = tmp_path / "gt.png"
    pred = tmp_path / "pred.png"
    _write_gradient(gt, size=16, offset=0.0)
    _write_gradient(pred, size=16, offset=0.2)

    # Different eval size must still succeed without raising
    result = compute_image_metrics([gt], [pred], eval_size=(8, 8))
    assert np.isfinite(result["psnr"])
