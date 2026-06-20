from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab3.common import Lab3Error
from lab3.reconstruction.nerf import apply_nerfstudio_split


def test_apply_nerfstudio_split_writes_explicit_filename_lists(tmp_path: Path) -> None:
    path = tmp_path / "transforms.json"
    path.write_text(
        json.dumps(
            {
                "frames": [
                    {"file_path": "images/a.jpg"},
                    {"file_path": "images/b.jpg"},
                ]
            }
        ),
        encoding="utf-8",
    )

    apply_nerfstudio_split(path, ("a.jpg",), ("b.jpg",))
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["train_filenames"] == ["images/a.jpg"]
    assert data["val_filenames"] == ["images/b.jpg"]
    assert data["test_filenames"] == ["images/b.jpg"]


def test_apply_nerfstudio_split_rejects_unknown_image(tmp_path: Path) -> None:
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps({"frames": [{"file_path": "images/a.jpg"}]}))

    with pytest.raises(Lab3Error, match="absent"):
        apply_nerfstudio_split(path, ("a.jpg",), ("missing.jpg",))


def test_apply_nerfstudio_split_maps_renamed_images_and_filters_unregistered(
    tmp_path: Path,
) -> None:
    images = tmp_path / "source"
    images.mkdir()
    for name in ("original_001.jpg", "original_003.jpg", "original_005.jpg"):
        (images / name).touch()
    path = tmp_path / "transforms.json"
    path.write_text(
        json.dumps(
            {
                "frames": [
                    {"file_path": "images/frame_00001.jpg"},
                    {"file_path": "images/frame_00003.jpg"},
                ]
            }
        )
    )

    counts = apply_nerfstudio_split(
        path,
        ("original_001.jpg", "original_003.jpg"),
        ("original_005.jpg",),
        source_images_dir=images,
    )
    data = json.loads(path.read_text())

    assert data["train_filenames"] == ["images/frame_00001.jpg"]
    assert data["test_filenames"] == ["images/frame_00003.jpg"]
    assert counts == {"train": 1, "test": 1, "dropped_train": 1, "dropped_test": 0}
