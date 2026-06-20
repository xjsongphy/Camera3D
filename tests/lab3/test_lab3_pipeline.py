from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from lab3.extract import ExtractionConfig, discover_inputs, prepare_dataset, split_train_test
from lab3.pipeline import config_from_dict, normalize_method


def test_discover_inputs_separates_images_and_videos(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"image")
    (tmp_path / "b.MP4").write_bytes(b"video")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    images, videos = discover_inputs(tmp_path)

    assert [path.name for path in images] == ["a.jpg"]
    assert [path.name for path in videos] == ["b.MP4"]


def test_split_train_test_uses_regular_holdout() -> None:
    names = [f"{idx:03d}.jpg" for idx in range(20)]

    train, test = split_train_test(names, 0.1)

    assert test == ["009.jpg", "019.jpg"]
    assert len(train) == 18
    assert not (set(train) & set(test))


def test_split_train_test_keeps_one_holdout_for_small_dataset() -> None:
    train, test = split_train_test(["a.jpg", "b.jpg", "c.jpg"], 0.1)

    assert train == ["a.jpg", "b.jpg"]
    assert test == ["c.jpg"]


def test_config_from_dict_normalizes_reconstructors() -> None:
    cfg = config_from_dict(
        {
            "input_dir": "data/scene",
            "scene_name": "desk",
            "methods": ["dgs", "nerf"],
            "reconstruction": {
                "3dgs": {
                    "repo_dir": "external/gaussian-splatting",
                    "iterations": 100,
                    "save_iterations": [25, 50, 100],
                },
                "nerf": {
                    "max_num_iterations": 200,
                    "save_iterations": [50, 100, 200],
                    "train_num_rays_per_batch": 8192,
                },
                "neus": {
                    "train_num_rays_per_batch": 4096,
                },
            },
        }
    )

    assert cfg.methods == ("3dgs", "nerf")
    assert cfg.dgs.repo_dir == Path("external/gaussian-splatting")
    assert cfg.dgs.iterations == 100
    assert cfg.dgs.save_iterations == (25, 50, 100)
    assert cfg.dgs.data_device == "cpu"
    assert cfg.dgs.camera_cache_size == 4
    assert cfg.nerf.max_num_iterations == 200
    assert cfg.nerf.save_iterations == (50, 100, 200)
    assert cfg.nerf.train_num_rays_per_batch == 8192
    assert cfg.neus.train_num_rays_per_batch == 4096


def test_3dgs_camera_cache_exhausts_chunk_before_switching() -> None:
    module_path = (
        Path(__file__).parents[2]
        / "gaussian-splatting"
        / "utils"
        / "camera_cache.py"
    )
    spec = importlib.util.spec_from_file_location("lab3_camera_cache", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    events: list[tuple[str, int]] = []

    class FakeCamera:
        def __init__(self, index: int) -> None:
            self.index = index

        def load_image_data(self, device: str) -> None:
            events.append(("load", self.index))

        def release_image_data(self) -> None:
            events.append(("release", self.index))

    sampler = module.CameraChunkSampler([FakeCamera(i) for i in range(7)], 3)
    first_chunk = [sampler.next().index for _ in range(3)]
    assert len(set(first_chunk)) == 3
    assert [event[0] for event in events] == ["load"] * 3

    sampler.next()
    assert [event[0] for event in events] == ["load"] * 3 + ["release"] * 3 + ["load"] * 3


def test_3dgs_camera_cache_resume_does_not_reload_consumed_camera() -> None:
    module_path = (
        Path(__file__).parents[2]
        / "gaussian-splatting"
        / "utils"
        / "camera_cache.py"
    )
    spec = importlib.util.spec_from_file_location("lab3_camera_cache_resume", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeCamera:
        def __init__(self, index: int) -> None:
            self.index = index
            self.loads = 0

        def load_image_data(self, device: str) -> None:
            self.loads += 1

        def release_image_data(self) -> None:
            pass

    cameras = [FakeCamera(i) for i in range(4)]
    sampler = module.CameraChunkSampler(cameras, 4)
    consumed = sampler.next()
    sampler.release()
    sampler.next()

    assert consumed.loads == 1
    assert sorted(camera.loads for camera in cameras) == [1, 2, 2, 2]


def test_config_from_dict_defaults_dgs_repo_to_relative_checkout() -> None:
    cfg = config_from_dict(
        {
            "input_dir": "data/scene",
            "scene_name": "desk",
            "methods": ["3dgs"],
        }
    )

    assert cfg.dgs.repo_dir == Path("gaussian-splatting")


def test_config_from_dict_reads_blur_threshold() -> None:
    cfg = config_from_dict(
        {
            "input_dir": "data/scene",
            "scene_name": "desk",
            "blur_threshold": 120.5,
        }
    )

    assert cfg.blur_threshold == 120.5


def test_config_from_dict_reads_crop_and_image_size() -> None:
    cfg = config_from_dict(
        {
            "input_dir": "data/scene",
            "crop_ratio": 0.8,
            "image_size": [900, 1600],
        }
    )

    assert cfg.crop_ratio == 0.8
    assert cfg.image_size == (900, 1600)


def test_prepare_dataset_center_crops_then_resizes(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "prepared"
    input_dir.mkdir()
    pixels = np.zeros((100, 200, 3), dtype=np.uint8)
    pixels[:, :] = (255, 0, 0)
    pixels[25:75, 50:150] = (0, 255, 0)
    Image.fromarray(pixels).save(input_dir / "wide.png")

    prepared = prepare_dataset(
        ExtractionConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            crop_ratio=0.5,
            image_size=(20, 40),
            test_ratio=0.0,
        )
    )

    with Image.open(prepared.images_dir / "img_000001.png") as image:
        result = np.asarray(image)
    assert image.size == (40, 20)
    assert np.all(result == (0, 255, 0))


def test_prepare_dataset_filters_blurry_images(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "prepared"
    input_dir.mkdir()

    sharp = np.zeros((32, 32), dtype=np.uint8)
    sharp[:, ::2] = 255
    blurry = np.full((32, 32), 127, dtype=np.uint8)

    Image.fromarray(sharp, mode="L").save(input_dir / "sharp.png")
    Image.fromarray(blurry, mode="L").save(input_dir / "blurry.png")

    prepared = prepare_dataset(
        ExtractionConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            blur_threshold=100.0,
            test_ratio=0.0,
        )
    )

    assert prepared.image_count == 1
    assert prepared.blurry_rejected_count == 1
    assert [path.name for path in prepared.images_dir.iterdir()] == ["img_000002.png"]

    with prepared.manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["source_path"].endswith("sharp.png")
    assert rows[0]["blur_score"] != ""


def test_normalize_method_rejects_unknown() -> None:
    try:
        normalize_method("mesh")
    except Exception as exc:
        assert "Unsupported method" in str(exc)
    else:
        raise AssertionError("normalize_method should reject unsupported methods")


def test_config_from_dict_parses_neus() -> None:
    cfg = config_from_dict(
        {
            "input_dir": "data/scene",
            "methods": ["sfm", "neus"],
            "reconstruction": {
                "neus": {
                    "method": "neus",
                    "max_num_iterations": 1234,
                    "mesh_resolution": 1024,
                }
            },
        }
    )

    assert cfg.methods == ("sfm", "neus")
    assert cfg.neus.method == "neus"
    assert cfg.neus.max_num_iterations == 1234
    assert cfg.neus.mesh_resolution == 1024
