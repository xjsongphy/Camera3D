from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lab1.colmap_utils import require_tool
from lab1.logging_utils import print_timing_summary, timed_block, write_timing_csv
from lab1.task1 import (
    FRAME_MAP_FILENAME,
    TIMING_FILENAME,
    Task1Error,
    _count_registered_images_in_model,
    _extract_frames,
    _format_float_tag,
    _has_any_frames,
    _parse_image_centers,
    _plot_trajectory,
    _run_colmap,
)

S2_VIDEOS = ["S2-1.mp4", "S2-2.mp4"]


@dataclass
class Task3Config:
    lab1_root: Path
    output_root: Path
    fps: float
    colmap_bin: str
    ffmpeg_bin: str
    force: bool
    dry_run: bool
    stage: str = "all"
    videos: list[str] | None = None


def _normalize_video_name(name: str) -> str:
    value = name.strip()
    if not value:
        raise Task1Error("Empty video name provided")
    if not value.endswith(".mp4"):
        value = f"{value}.mp4"
    return value


def _count_points3d(points3d_txt: Path) -> int:
    count = 0
    with points3d_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            count += 1
    return count


def _compute_jump_stats(centers: np.ndarray) -> tuple[float, float, float]:
    if centers.shape[0] < 3:
        return 0.0, 0.0, 0.0
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    median_step = float(np.median(step))
    max_step = float(np.max(step))
    if median_step <= 1e-12:
        jump_ratio = 0.0
    else:
        jump_ratio = float(np.mean(step > (3.0 * median_step)))
    return median_step, max_step, jump_ratio


def _write_analysis(
    *,
    case_root: Path,
    case_name: str,
    fps: float,
    total_frames: int,
    registered_frames: int,
    points3d_count: int,
    median_step: float,
    max_step: float,
    jump_ratio: float,
) -> None:
    registration_ratio = (registered_frames / total_frames) if total_frames > 0 else 0.0
    analysis_txt = case_root / "analysis.txt"
    analysis_csv = case_root / "analysis.csv"

    lines = [
        f"video={case_name}",
        f"fps={fps:g}",
        f"total_frames={total_frames}",
        f"registered_frames={registered_frames}",
        f"registration_ratio={registration_ratio:.8f}",
        f"points3d={points3d_count}",
        f"trajectory_step_median={median_step:.8f}",
        f"trajectory_step_max={max_step:.8f}",
        f"trajectory_jump_ratio={jump_ratio:.8f}",
    ]
    analysis_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with analysis_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "fps",
                "total_frames",
                "registered_frames",
                "registration_ratio",
                "points3d",
                "trajectory_step_median",
                "trajectory_step_max",
                "trajectory_jump_ratio",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "video": case_name,
                "fps": f"{fps:g}",
                "total_frames": total_frames,
                "registered_frames": registered_frames,
                "registration_ratio": f"{registration_ratio:.8f}",
                "points3d": points3d_count,
                "trajectory_step_median": f"{median_step:.8f}",
                "trajectory_step_max": f"{max_step:.8f}",
                "trajectory_jump_ratio": f"{jump_ratio:.8f}",
            }
        )


def run_task3(cfg: Task3Config) -> int:
    if cfg.fps <= 0:
        raise Task1Error(f"fps must be positive, got {cfg.fps}")
    if cfg.stage not in {"all", "extract", "sfm", "analyze"}:
        raise Task1Error(f"Unsupported stage: {cfg.stage}. Choose from all|extract|sfm|analyze")

    videos_dir = cfg.lab1_root / "assets" / "videos"
    selected_videos = S2_VIDEOS if not cfg.videos else [_normalize_video_name(v) for v in cfg.videos]
    invalid = [v for v in selected_videos if v not in S2_VIDEOS]
    if invalid:
        raise Task1Error(f"Unsupported video(s): {invalid}. Supported: {S2_VIDEOS} (or short names S2-1/S2-2)")

    if not cfg.dry_run:
        if cfg.stage in {"all", "extract"}:
            require_tool(cfg.ffmpeg_bin, error_cls=Task1Error)
        if cfg.stage in {"all", "sfm"}:
            require_tool(cfg.colmap_bin, error_cls=Task1Error)

    param_tag = f"fps{_format_float_tag(cfg.fps)}"

    for video_name in selected_videos:
        timings: dict[str, float] = {}
        video_path = videos_dir / video_name
        if not video_path.exists():
            raise Task1Error(f"Video not found: {video_path}")

        case_name = video_path.stem
        case_root = cfg.output_root / f"{case_name}_{param_tag}"
        images_dir = case_root / "images"
        frame_map_path = case_root / FRAME_MAP_FILENAME
        sparse_root = case_root / "sparse"
        db_path = case_root / "database.db"
        timing_path = case_root / TIMING_FILENAME
        model_dir = sparse_root / "0"

        print(f"\n=== Task3 / {case_name} / {param_tag} ===")
        if not cfg.dry_run:
            case_root.mkdir(parents=True, exist_ok=True)

        if cfg.stage in {"all", "extract"}:
            with timed_block("extract", timings):
                _extract_frames(
                    video_path=video_path,
                    images_dir=images_dir,
                    frame_map_path=frame_map_path,
                    fps=cfg.fps,
                    ffmpeg_bin=cfg.ffmpeg_bin,
                    force=cfg.force,
                    dry_run=cfg.dry_run,
                )

        if cfg.stage == "extract":
            if not cfg.dry_run:
                write_timing_csv(timing_path, timings)
                print(f"Saved timing summary: {timing_path}")
            print_timing_summary(f"Timing / {case_name} / {param_tag}", timings)
            continue

        if cfg.stage in {"all", "sfm"}:
            if not _has_any_frames(images_dir):
                raise Task1Error(
                    f"No extracted frames found under {images_dir}. "
                    "Run extraction first (--stage extract) or use --stage all."
                )
            with timed_block("sfm_total", timings):
                model_dir = _run_colmap(
                    images_dir=images_dir,
                    sparse_root=sparse_root,
                    db_path=db_path,
                    colmap_bin=cfg.colmap_bin,
                    force=cfg.force,
                    dry_run=cfg.dry_run,
                    timings=timings,
                )
            if not cfg.dry_run:
                with timed_block("plot", timings):
                    centers, _ = _parse_image_centers(model_dir / "images.txt")
                    _plot_trajectory(
                        centers,
                        case_root / "trajectory_raw.png",
                        f"{case_name} Raw SfM Trajectory (fps={cfg.fps:g})",
                    )

        if cfg.stage in {"all", "analyze"}:
            if cfg.dry_run:
                print("Dry run: skip analysis file generation.")
            else:
                with timed_block("analyze", timings):
                    images_txt = model_dir / "images.txt"
                    points3d_txt = model_dir / "points3D.txt"
                    if not images_txt.exists() or not points3d_txt.exists():
                        raise Task1Error(
                            f"Missing SfM outputs under {model_dir}. "
                            "Run with --stage sfm or --stage all first."
                        )

                    centers, _ = _parse_image_centers(images_txt)
                    registered_frames = _count_registered_images_in_model(images_txt)
                    total_frames = len(list(images_dir.glob("*.jpg")))
                    points3d_count = _count_points3d(points3d_txt)
                    median_step, max_step, jump_ratio = _compute_jump_stats(centers)

                    _write_analysis(
                        case_root=case_root,
                        case_name=case_name,
                        fps=cfg.fps,
                        total_frames=total_frames,
                        registered_frames=registered_frames,
                        points3d_count=points3d_count,
                        median_step=median_step,
                        max_step=max_step,
                        jump_ratio=jump_ratio,
                    )
                print(f"Saved analysis: {case_root / 'analysis.txt'}")
                print(f"Saved analysis: {case_root / 'analysis.csv'}")

        if not cfg.dry_run:
            write_timing_csv(timing_path, timings)
            print(f"Saved timing summary: {timing_path}")
        print_timing_summary(f"Timing / {case_name} / {param_tag}", timings)

    print("\nTask3 completed.")
    return 0
