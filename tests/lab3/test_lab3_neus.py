from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from types import SimpleNamespace

from lab3.reconstruction.neus import NeuSConfig, build_sdfstudio_dataset, write_neus_eval_config


def test_neus_defaults_to_neus_facto_and_mesh_export() -> None:
    cfg = NeuSConfig()

    assert cfg.method == "neus-facto"
    assert cfg.export_mesh is True
    assert cfg.mesh_resolution == 512
    assert cfg.train_num_rays_per_batch is None
    assert cfg.train_num_images_to_sample_from == -1
    assert cfg.train_num_times_to_repeat_images == -1


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


def test_build_sdfstudio_dataset_filters_canonical_split(tmp_path: Path) -> None:
    model = tmp_path / "sparse" / "0"
    images = tmp_path / "images"
    model.mkdir(parents=True)
    images.mkdir()
    for name in ("train.jpg", "test.jpg"):
        (images / name).write_bytes(b"image")
    (model / "cameras.txt").write_text("1 PINHOLE 640 480 500 500 320 240\n")
    (model / "images.txt").write_text(
        "1 1 0 0 0 0 0 2 1 train.jpg\n\n"
        "2 1 0 0 0 1 0 2 1 test.jpg\n\n",
        encoding="utf-8",
    )
    (model / "points3D.txt").write_text("1 0 0 0 255 255 255 0.1 1 0\n")

    path = build_sdfstudio_dataset(
        model, images, tmp_path / "processed", image_names=("test.jpg",)
    )
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"]

    assert [Path(frame["rgb_path"]).name for frame in frames] == ["test.jpg"]


def test_write_neus_eval_config_retargets_only_data(tmp_path: Path) -> None:
    config = SimpleNamespace(
        data=Path("train"),
        pipeline=SimpleNamespace(
            datamanager=SimpleNamespace(
                data=Path("train"),
                dataparser=SimpleNamespace(data=Path("train"), auto_orient=True),
            )
        ),
    )
    source = tmp_path / "config.yml"
    source.write_text(yaml.dump(config), encoding="utf-8")

    output = write_neus_eval_config(source, tmp_path / "test-data", tmp_path / "eval.yml")
    loaded = yaml.load(output.read_text(encoding="utf-8"), Loader=yaml.Loader)

    assert loaded.data == (tmp_path / "test-data").resolve()
    assert loaded.pipeline.datamanager.dataparser.auto_orient is False
