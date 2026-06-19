from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from lab3.common import IMAGE_SUFFIXES, VIDEO_SUFFIXES, Lab3Error, copy_file, require_tool, run_cmd, slugify


@dataclass(frozen=True)
class ExtractionConfig:
    input_dir: Path
    output_dir: Path
    fps: float = 2.0
    image_limit: int | None = None
    blur_threshold: float | None = None
    test_ratio: float = 0.1
    ffmpeg_bin: str = "ffmpeg"
    force: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class PreparedDataset:
    root: Path
    images_dir: Path
    train_list: Path
    test_list: Path
    manifest_path: Path
    image_count: int
    video_count: int
    train_count: int
    test_count: int
    blurry_rejected_count: int = 0


def prepare_dataset(cfg: ExtractionConfig) -> PreparedDataset:
    if cfg.fps <= 0:
        raise Lab3Error(f"fps must be positive, got {cfg.fps}")
    if cfg.blur_threshold is not None and cfg.blur_threshold < 0:
        raise Lab3Error(f"blur_threshold must be non-negative, got {cfg.blur_threshold}")
    if not 0.0 <= cfg.test_ratio < 1.0:
        raise Lab3Error(f"test_ratio must be in [0, 1), got {cfg.test_ratio}")
    if not cfg.input_dir.exists():
        raise Lab3Error(f"Input directory not found: {cfg.input_dir}")
    if not cfg.input_dir.is_dir():
        raise Lab3Error(f"Input path must be a directory: {cfg.input_dir}")

    images_dir = cfg.output_dir / "images"
    manifest_path = cfg.output_dir / "manifest.csv"
    train_list = cfg.output_dir / "train.txt"
    test_list = cfg.output_dir / "test.txt"

    if cfg.force and cfg.output_dir.exists() and not cfg.dry_run:
        shutil.rmtree(cfg.output_dir)
    if not cfg.dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)

    images, videos = discover_inputs(cfg.input_dir)
    if not images and not videos:
        raise Lab3Error(f"No supported images or videos found under {cfg.input_dir}")
    if videos and not cfg.dry_run:
        require_tool(cfg.ffmpeg_bin)

    manifest_rows: list[dict[str, str]] = []
    copied_names: list[str] = []
    blurry_rejected_count = 0

    for index, image_path in enumerate(images, start=1):
        if cfg.image_limit is not None and len(copied_names) >= cfg.image_limit:
            break
        blur_score = compute_blur_score(image_path) if cfg.blur_threshold is not None else None
        if blur_score is not None and blur_score < cfg.blur_threshold:
            blurry_rejected_count += 1
            continue
        image_name = f"img_{index:06d}{image_path.suffix.lower()}"
        if not cfg.dry_run:
            copy_file(image_path, images_dir / image_name, overwrite=cfg.force)
        copied_names.append(image_name)
        manifest_rows.append(
            {
                "image_name": image_name,
                "source_type": "image",
                "source_path": str(image_path),
                "source_time_sec": "",
                "blur_score": "" if blur_score is None else f"{blur_score:.6f}",
            }
        )

    for video_index, video_path in enumerate(videos, start=1):
        frame_prefix = f"vid_{video_index:03d}_{slugify(video_path.stem)}"
        pattern = images_dir / f"{frame_prefix}_%06d.jpg"
        run_cmd(
            [
                cfg.ffmpeg_bin,
                "-y" if cfg.force else "-n",
                "-i",
                str(video_path),
                "-vf",
                f"fps={cfg.fps}",
                "-q:v",
                "2",
                str(pattern),
            ],
            dry_run=cfg.dry_run,
        )
        if cfg.dry_run:
            continue
        extracted = sorted(images_dir.glob(f"{frame_prefix}_*.jpg"))
        for frame_index, frame_path in enumerate(extracted):
            blur_score = compute_blur_score(frame_path) if cfg.blur_threshold is not None else None
            if blur_score is not None and blur_score < cfg.blur_threshold:
                blurry_rejected_count += 1
                frame_path.unlink(missing_ok=True)
                continue
            if cfg.image_limit is not None and len(copied_names) >= cfg.image_limit:
                frame_path.unlink(missing_ok=True)
                continue
            copied_names.append(frame_path.name)
            manifest_rows.append(
                {
                    "image_name": frame_path.name,
                    "source_type": "video",
                    "source_path": str(video_path),
                    "source_time_sec": f"{frame_index / cfg.fps:.9f}",
                    "blur_score": "" if blur_score is None else f"{blur_score:.6f}",
                }
            )

    if not cfg.dry_run:
        if not copied_names:
            raise Lab3Error(f"No images prepared under {images_dir}")
        train_names, test_names = split_train_test(copied_names, cfg.test_ratio)
        _write_manifest(manifest_path, manifest_rows)
        _write_lines(train_list, train_names)
        _write_lines(test_list, test_names)
    else:
        train_names, test_names = [], []

    return PreparedDataset(
        root=cfg.output_dir,
        images_dir=images_dir,
        train_list=train_list,
        test_list=test_list,
        manifest_path=manifest_path,
        image_count=len(copied_names),
        video_count=len(videos),
        train_count=len(train_names),
        test_count=len(test_names),
        blurry_rejected_count=blurry_rejected_count,
    )


def discover_inputs(input_dir: Path) -> tuple[list[Path], list[Path]]:
    files = sorted(path for path in input_dir.rglob("*") if path.is_file())
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    videos = [path for path in files if path.suffix.lower() in VIDEO_SUFFIXES]
    return images, videos


def split_train_test(image_names: list[str], test_ratio: float) -> tuple[list[str], list[str]]:
    if not image_names:
        return [], []
    if test_ratio <= 0 or len(image_names) < 2:
        return list(image_names), []
    stride = max(round(1.0 / test_ratio), 2)
    test = [name for idx, name in enumerate(image_names) if idx % stride == stride - 1]
    if not test and len(image_names) >= 10:
        test = [image_names[-1]]
    test_set = set(test)
    train = [name for name in image_names if name not in test_set]
    return train, test


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["image_name", "source_type", "source_path", "source_time_sec", "blur_score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{value}\n")


def compute_blur_score(path: Path) -> float:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    return float(np.var(laplacian))
