from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from PIL import Image

from lab3.qualitative import (
    _is_positional_render_bundle,
    _list_render_images,
    _resolve_render,
    build_comparison_figure,
    error_map,
)


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


def test_resolve_render_prefers_positional_bundle_over_stem_match(tmp_path) -> None:
    render_dir = tmp_path / "ours_40000" / "renders"
    gt_dir = tmp_path / "ours_40000" / "gt"
    render_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    for name, value in (("00000.png", 10), ("00001.png", 20), ("00030.png", 30)):
        Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(render_dir / name)
        Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(gt_dir / name)

    ordered = _list_render_images(render_dir)
    assert _is_positional_render_bundle(render_dir, ordered)

    # Canonical name ends with 000030, but for positional bundles index=1 should
    # still resolve to the second render (00001.png), not the misleading 00030.
    resolved = _resolve_render(render_dir, "vid_001_dormitory_000030.jpg", 1, ordered)
    assert resolved is not None
    assert resolved.name == "00001.png"
