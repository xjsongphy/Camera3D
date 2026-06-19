from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lab3.reconstruction.neus import NeuSConfig, build_sdfstudio_dataset


def test_neus_defaults_to_neus_facto_and_mesh_export() -> None:
    cfg = NeuSConfig()

    assert cfg.method == "neus-facto"
    assert cfg.export_mesh is True
    assert cfg.mesh_resolution == 512


def test_build_sdfstudio_dataset_converts_colmap_text_model(tmp_path: Path) -> None:
    model = tmp_path / "sparse" / "0"
    images = tmp_path / "images"
    model.mkdir(parents=True)
    images.mkdir()
    (images / "frame.jpg").write_bytes(b"image")
    (model / "cameras.txt").write_text("1 PINHOLE 640 480 500 510 320 240\n")
    (model / "images.txt").write_text(
        "1 1 0 0 0 0 0 2 1 frame.jpg\n0 0 -1\n", encoding="utf-8"
    )
    (model / "points3D.txt").write_text(
        "1 0 0 0 255 255 255 0.1 1 0\n2 1 0 0 255 255 255 0.1 1 0\n",
        encoding="utf-8",
    )

    path = build_sdfstudio_dataset(model, images, tmp_path / "processed")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["has_mono_prior"] is False
    assert data["width"] == 640 and data["height"] == 480
    assert data["frames"][0]["rgb_path"] == str(images / "frame.jpg")
    assert np.asarray(data["frames"][0]["intrinsics"])[0, 0] == 500
    assert np.asarray(data["frames"][0]["camtoworld"]).shape == (4, 4)
