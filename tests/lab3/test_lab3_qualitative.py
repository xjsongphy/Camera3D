from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from lab3.qualitative import build_comparison_figure, error_map


def test_error_map_is_zero_for_identical_images() -> None:
    gt = np.random.default_rng(0).random((16, 16, 3))
    assert error_map(gt, gt).max() == 0.0


def test_error_map_has_image_shape_and_unit_range() -> None:
    gt = np.zeros((16, 16, 3))
    pred = np.ones((16, 16, 3))
    err = error_map(gt, pred)
    assert err.shape == (16, 16)
    assert err.min() >= 0.0 and err.max() <= 1.0
    assert err.max() == 1.0  # fully off -> normalized to 1


def test_build_comparison_figure_creates_axes_for_gt_methods_and_errors() -> None:
    gt = np.random.default_rng(1).random((12, 12, 3))
    methods = {
        "3dgs": np.clip(gt + 0.1, 0, 1),
        "nerf": np.clip(gt + 0.2, 0, 1),
    }

    fig = build_comparison_figure("img_000009", gt, methods)

    # 1 GT + 2 methods + 2 error maps = 5 axes
    assert len(fig.axes) == 5
    # GT axis should be the first
    titles = [ax.get_title() for ax in fig.axes]
    assert "GT" in titles[0]
