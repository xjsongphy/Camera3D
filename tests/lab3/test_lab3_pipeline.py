from __future__ import annotations

from pathlib import Path

from lab3.extract import discover_inputs, split_train_test
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
                    "save_every": 25,
                },
                "nerf": {
                    "max_num_iterations": 200,
                    "save_every": 50,
                    "save_only_latest_checkpoint": False,
                },
            },
        }
    )

    assert cfg.methods == ("3dgs", "nerf")
    assert cfg.dgs.repo_dir == Path("external/gaussian-splatting")
    assert cfg.dgs.iterations == 100
    assert cfg.dgs.save_every == 25
    assert cfg.nerf.max_num_iterations == 200
    assert cfg.nerf.save_every == 50
    assert cfg.nerf.save_only_latest_checkpoint is False


def test_config_from_dict_defaults_dgs_repo_to_relative_checkout() -> None:
    cfg = config_from_dict(
        {
            "input_dir": "data/scene",
            "scene_name": "desk",
            "methods": ["3dgs"],
        }
    )

    assert cfg.dgs.repo_dir == Path("gaussian-splatting")


def test_normalize_method_rejects_unknown() -> None:
    try:
        normalize_method("mesh")
    except Exception as exc:
        assert "Unsupported method" in str(exc)
    else:
        raise AssertionError("normalize_method should reject unsupported methods")
