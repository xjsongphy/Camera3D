from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


S1_VIDEOS = ["S1-1.mp4", "S1-2.mp4", "S1-3.mp4"]


@dataclass
class Task1Config:
    lab1_root: Path
    output_root: Path
    fps: float
    colmap_bin: str
    ffmpeg_bin: str
    force: bool
    dry_run: bool
    videos: list[str] | None = None
    stage: str = "all"


class Task1Error(RuntimeError):
    pass


def _run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=120)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        tail.append(line.rstrip("\n"))
    return_code = proc.wait()
    if return_code != 0:
        log_tail = "\n".join(tail)
        raise Task1Error(
            f"Command failed with exit code {return_code}: {' '.join(cmd)}\n"
            f"Last output lines:\n{log_tail}"
        )


def _require_tool(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise Task1Error(f"Required tool not found in PATH: {tool_name}")


def _extract_frames(video_path: Path, images_dir: Path, fps: float, ffmpeg_bin: str, force: bool, dry_run: bool) -> None:
    if images_dir.exists() and force and not dry_run:
        shutil.rmtree(images_dir)
    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(images_dir / "%06d.jpg")
    cmd = [
        ffmpeg_bin,
        "-y" if force else "-n",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        output_pattern,
    ]
    _run_cmd(cmd, dry_run=dry_run)


def _has_any_frames(images_dir: Path) -> bool:
    return images_dir.exists() and any(images_dir.glob("*.jpg"))


def _run_colmap(images_dir: Path, sparse_root: Path, db_path: Path, colmap_bin: str, force: bool, dry_run: bool) -> Path:
    if db_path.exists() and force and not dry_run:
        db_path.unlink()
    if sparse_root.exists() and force and not dry_run:
        shutil.rmtree(sparse_root)
    if not dry_run:
        sparse_root.mkdir(parents=True, exist_ok=True)

    _run_cmd(
        [
            colmap_bin,
            "feature_extractor",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_dir),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "PINHOLE",
        ],
        dry_run=dry_run,
    )

    _run_cmd(
        [
            colmap_bin,
            "sequential_matcher",
            "--database_path",
            str(db_path),
        ],
        dry_run=dry_run,
    )

    _run_cmd(
        [
            colmap_bin,
            "mapper",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_root),
        ],
        dry_run=dry_run,
    )

    model_dir = sparse_root / "0"
    if not dry_run and not model_dir.exists():
        raise Task1Error(f"COLMAP mapper produced no model under: {model_dir}")

    _run_cmd(
        [
            colmap_bin,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(model_dir),
            "--output_type",
            "TXT",
        ],
        dry_run=dry_run,
    )

    return model_dir


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise Task1Error("Encountered near-zero quaternion norm while parsing COLMAP poses.")
    q /= norm
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def _parse_image_centers(images_txt: Path) -> tuple[np.ndarray, list[str]]:
    centers: list[np.ndarray] = []
    names: list[str] = []
    with images_txt.open("r", encoding="utf-8") as f:
        while True:
            pose_line = f.readline()
            if not pose_line:
                break
            pose_line = pose_line.strip()
            if not pose_line or pose_line.startswith("#"):
                continue
            parts = pose_line.split()
            if len(parts) < 10:
                raise Task1Error(f"Malformed COLMAP images.txt pose line: {pose_line}")
            image_id = parts[0]
            if not image_id.isdigit():
                raise Task1Error(f"Expected image id at pose line start, got: {pose_line}")

            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = parts[9]

            r = _quat_to_rot(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=float)
            c = -r.T @ t

            centers.append(c)
            names.append(name)
            # COLMAP images.txt stores a second line of POINTS2D per image entry.
            # Consume it explicitly to avoid accidental mis-parsing.
            _ = f.readline()

    if not centers:
        raise Task1Error(f"No image poses parsed from {images_txt}")

    order = np.argsort(np.array(names))
    centers_arr = np.array(centers)[order]
    names_sorted = [names[i] for i in order]
    return centers_arr, names_sorted


def _plot_trajectory(centers: np.ndarray, out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], marker="o", markersize=2)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _format_float_tag(value: float) -> str:
    s = f"{value:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _build_param_tag(cfg: Task1Config) -> str:
    # Keep parameterized outputs separate to avoid mixing results from different runs.
    return f"fps{_format_float_tag(cfg.fps)}"


def _has_completed_outputs(case_root: Path) -> bool:
    images_dir = case_root / "images"
    sparse_dir = case_root / "sparse" / "0"
    trajectory = case_root / "trajectory.png"
    return (
        images_dir.exists()
        and any(images_dir.glob("*.jpg"))
        and (sparse_dir / "images.txt").exists()
        and (sparse_dir / "cameras.txt").exists()
        and (sparse_dir / "points3D.txt").exists()
        and trajectory.exists()
    )


def _has_completed_sfm(case_root: Path) -> bool:
    sparse_dir = case_root / "sparse" / "0"
    trajectory = case_root / "trajectory.png"
    return (
        (sparse_dir / "images.txt").exists()
        and (sparse_dir / "cameras.txt").exists()
        and (sparse_dir / "points3D.txt").exists()
        and trajectory.exists()
    )


def _normalize_video_name(name: str) -> str:
    value = name.strip()
    if not value:
        raise Task1Error("Empty video name provided")
    if not value.endswith(".mp4"):
        value = f"{value}.mp4"
    return value


def run_task1(cfg: Task1Config) -> int:
    videos_dir = cfg.lab1_root / "assets" / "videos"
    selected_videos = S1_VIDEOS if not cfg.videos else [_normalize_video_name(v) for v in cfg.videos]
    param_tag = _build_param_tag(cfg)

    invalid = [v for v in selected_videos if v not in S1_VIDEOS]
    if invalid:
        raise Task1Error(
            f"Unsupported video(s): {invalid}. Supported: {S1_VIDEOS} "
            "(or short names S1-1/S1-2/S1-3)"
        )

    if cfg.stage not in {"all", "extract", "sfm"}:
        raise Task1Error(f"Unsupported stage: {cfg.stage}. Choose from all|extract|sfm")
    if cfg.fps <= 0:
        raise Task1Error(f"fps must be positive, got {cfg.fps}")

    if not cfg.dry_run:
        if cfg.stage in {"all", "extract"}:
            _require_tool(cfg.ffmpeg_bin)
        if cfg.stage in {"all", "sfm"}:
            _require_tool(cfg.colmap_bin)

    for video_name in selected_videos:
        video_path = videos_dir / video_name
        if not video_path.exists():
            raise Task1Error(f"Video not found: {video_path}")

        case_name = video_path.stem
        case_root = cfg.output_root / f"{case_name}_{param_tag}"
        images_dir = case_root / "images"
        sparse_root = case_root / "sparse"
        db_path = case_root / "database.db"

        print(f"\n=== Task1 / {case_name} / {param_tag} ===")
        if cfg.stage == "all" and not cfg.force and _has_completed_outputs(case_root):
            print(f"Reuse existing outputs (same parameters): {case_root}")
            continue

        if not cfg.dry_run:
            case_root.mkdir(parents=True, exist_ok=True)

        if cfg.stage in {"all", "extract"}:
            if not cfg.force and _has_any_frames(images_dir):
                print(f"Reuse extracted frames: {images_dir}")
            else:
                _extract_frames(
                    video_path=video_path,
                    images_dir=images_dir,
                    fps=cfg.fps,
                    ffmpeg_bin=cfg.ffmpeg_bin,
                    force=cfg.force,
                    dry_run=cfg.dry_run,
                )
        elif cfg.stage == "sfm":
            if not _has_any_frames(images_dir):
                raise Task1Error(
                    f"No extracted frames found under {images_dir}. "
                    "Run extraction first (--stage extract) or use --stage all."
                )
            print(f"Reuse extracted frames: {images_dir}")

        if cfg.stage == "extract":
            print("Skip SfM for extract stage.")
            continue

        if cfg.stage == "sfm" and not cfg.force and _has_completed_sfm(case_root):
            print(f"Reuse SfM outputs: {case_root}")
            continue

        model_dir = _run_colmap(
            images_dir=images_dir,
            sparse_root=sparse_root,
            db_path=db_path,
            colmap_bin=cfg.colmap_bin,
            force=cfg.force,
            dry_run=cfg.dry_run,
        )

        if cfg.dry_run:
            continue

        images_txt = model_dir / "images.txt"
        centers, _ = _parse_image_centers(images_txt)
        _plot_trajectory(
            centers,
            case_root / "trajectory.png",
            f"{case_name} Camera Trajectory (fps={cfg.fps:g})",
        )

        print(f"Saved trajectory: {case_root / 'trajectory.png'}")
        print(f"Sparse model text files: {model_dir}")

    print("\nTask1 completed.")
    return 0
