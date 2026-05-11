from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from lab1.colmap_utils import (
    require_tool,
    run_cmd,
    run_feature_extractor,
    run_model_converter,
    run_sequential_matcher,
)
from lab1.geometry_utils import (
    apply_sim3,
    parse_image_centers_sorted,
    parse_image_poses_sorted,
    quat_to_rot,
    umeyama_sim3,
)
from lab1.logging_utils import print_timing_summary, timed_block, write_timing_csv


S1_VIDEOS = ["S1-1.mp4", "S1-2.mp4", "S1-3.mp4"]
DEFAULT_FPS_LIST = [4.0, 8.0, 16.0, 30.0]
ALL_FPS_SENTINEL = -1.0  # Special value to indicate "all fps"



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
    mode: str = "run"
    direction_arrows: int = 12
    max_points_plot: int = 12000


class Task1Error(RuntimeError):
    pass


FRAME_MAP_FILENAME = "frame_map.csv"
TIMING_FILENAME = "timing.csv"


def _probe_video_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rate_text = result.stdout.strip()
    if not rate_text:
        raise Task1Error(f"Failed to probe video fps for {video_path}")
    if "/" in rate_text:
        num_text, den_text = rate_text.split("/", maxsplit=1)
        value = float(num_text) / float(den_text)
    else:
        value = float(rate_text)
    if value <= 0:
        raise Task1Error(f"Probed non-positive source fps {value} for {video_path}")
    return value


def _write_frame_map(frame_map_path: Path, image_names: list[str], source_fps: float, sample_fps: float) -> None:
    with frame_map_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "sample_index", "source_time_sec", "source_frame_index"])
        for sample_idx, image_name in enumerate(image_names):
            source_time = sample_idx / sample_fps
            source_frame_index = int(round(source_time * source_fps))
            writer.writerow([image_name, sample_idx, f"{source_time:.9f}", source_frame_index])


def _load_frame_map(frame_map_path: Path) -> dict[str, int]:
    if not frame_map_path.exists():
        raise Task1Error(f"Frame map not found: {frame_map_path}")
    mapping: dict[str, int] = {}
    with frame_map_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["image_name"]
            source_frame_index = int(row["source_frame_index"])
            mapping[image_name] = source_frame_index
    if not mapping:
        raise Task1Error(f"Frame map is empty: {frame_map_path}")
    return mapping


def _extract_frames(
    video_path: Path,
    images_dir: Path,
    frame_map_path: Path,
    fps: float,
    ffmpeg_bin: str,
    force: bool,
    dry_run: bool,
) -> None:
    if images_dir.exists() and force and not dry_run:
        shutil.rmtree(images_dir)
    if frame_map_path.exists() and force and not dry_run:
        frame_map_path.unlink()
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
    run_cmd(cmd, dry_run=dry_run, error_cls=Task1Error)
    if dry_run:
        return

    image_names = sorted(p.name for p in images_dir.glob("*.jpg"))
    if not image_names:
        raise Task1Error(f"No frames extracted under {images_dir}")
    source_fps = _probe_video_fps(video_path)
    _write_frame_map(frame_map_path, image_names, source_fps=source_fps, sample_fps=fps)


def _has_any_frames(images_dir: Path) -> bool:
    return images_dir.exists() and any(images_dir.glob("*.jpg"))


def _run_colmap(
    images_dir: Path,
    sparse_root: Path,
    db_path: Path,
    colmap_bin: str,
    feature_mask_path: Path | None,
    camera_mask_path: Path | None,
    force: bool,
    dry_run: bool,
    timings: dict[str, float] | None = None,
) -> Path:
    if db_path.exists() and force and not dry_run:
        db_path.unlink()
    if sparse_root.exists() and force and not dry_run:
        shutil.rmtree(sparse_root)
    if not dry_run:
        sparse_root.mkdir(parents=True, exist_ok=True)

    timings = {} if timings is None else timings
    with timed_block("feature_extractor", timings):
        run_feature_extractor(
            colmap_bin=colmap_bin,
            db_path=db_path,
            images_dir=images_dir,
            mask_path=feature_mask_path,
            camera_mask_path=camera_mask_path,
            dry_run=dry_run,
            error_cls=Task1Error,
        )

    with timed_block("sequential_matcher", timings):
        run_sequential_matcher(
            colmap_bin=colmap_bin,
            db_path=db_path,
            dry_run=dry_run,
            error_cls=Task1Error,
        )

    with timed_block("hierarchical_mapper", timings):
        run_cmd(
            [
                colmap_bin,
                "hierarchical_mapper",
                "--database_path",
                str(db_path),
                "--image_path",
                str(images_dir),
                "--output_path",
                str(sparse_root),
            ],
            dry_run=dry_run,
            error_cls=Task1Error,
        )

    if dry_run:
        return sparse_root / "0"

    model_dir = _select_best_model_dir(sparse_root)
    if model_dir is None:
        raise Task1Error(f"COLMAP hierarchical_mapper produced no model under: {sparse_root}")

    with timed_block("model_converter", timings):
        run_model_converter(
            colmap_bin=colmap_bin,
            model_dir=model_dir,
            dry_run=dry_run,
            error_cls=Task1Error,
        )

    model_dir = _materialize_canonical_model_dir(model_dir, sparse_root / "0")
    return model_dir


def _count_registered_images_in_model(images_txt: Path) -> int:
    count = 0
    with images_txt.open("r", encoding="utf-8") as f:
        while True:
            pose_line = f.readline()
            if not pose_line:
                break
            pose_line = pose_line.strip()
            if not pose_line or pose_line.startswith("#"):
                continue
            parts = pose_line.split()
            if len(parts) >= 10 and parts[0].isdigit():
                count += 1
            _ = f.readline()
    return count


def _select_best_model_dir(sparse_root: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    for child in sorted(sparse_root.iterdir()):
        if not child.is_dir():
            continue
        images_bin = child / "images.bin"
        images_txt = child / "images.txt"
        if not images_bin.exists() and not images_txt.exists():
            continue
        count = _count_registered_images_in_model(images_txt) if images_txt.exists() else 0
        if best is None or count > best[0]:
            best = (count, child)
    return None if best is None else best[1]


def _materialize_canonical_model_dir(model_dir: Path, canonical_dir: Path) -> Path:
    if model_dir == canonical_dir:
        return model_dir
    if canonical_dir.exists():
        shutil.rmtree(canonical_dir)
    shutil.copytree(model_dir, canonical_dir)
    return canonical_dir


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return quat_to_rot(qw, qx, qy, qz, error_cls=Task1Error)


def _parse_image_poses(images_txt: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    return parse_image_poses_sorted(images_txt, error_cls=Task1Error)


def _parse_image_centers(images_txt: Path) -> tuple[np.ndarray, list[str]]:
    return parse_image_centers_sorted(images_txt, error_cls=Task1Error)


def _select_direction_indices(num_points: int, arrow_count: int) -> np.ndarray:
    if num_points <= 0:
        raise Task1Error("Cannot select direction arrows from an empty trajectory.")
    if arrow_count <= 0:
        return np.array([], dtype=int)
    count = min(num_points, arrow_count)
    return np.unique(np.round(np.linspace(0, num_points - 1, count)).astype(int))


def _plot_trajectory(
    centers: np.ndarray,
    out_path: Path,
    title: str,
    forward_dirs: np.ndarray | None = None,
    direction_arrows: int = 0,
) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], marker="o", markersize=2)
    if forward_dirs is not None and direction_arrows > 0:
        indices = _select_direction_indices(len(centers), direction_arrows)
        sampled_centers = centers[indices]
        sampled_dirs = forward_dirs[indices]
        axis_span = np.ptp(centers, axis=0)
        arrow_length = max(float(np.max(axis_span)) * 0.08, 1.0)
        ax.quiver(
            sampled_centers[:, 0],
            sampled_centers[:, 1],
            sampled_centers[:, 2],
            sampled_dirs[:, 0],
            sampled_dirs[:, 1],
            sampled_dirs[:, 2],
            length=arrow_length,
            normalize=True,
            color="tab:red",
            linewidth=1.2,
        )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _parse_points3d(points3d_txt: Path) -> tuple[np.ndarray, np.ndarray]:
    xyzs: list[list[float]] = []
    rgbs: list[list[float]] = []
    with points3d_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            xyzs.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgbs.append([float(parts[4]) / 255.0, float(parts[5]) / 255.0, float(parts[6]) / 255.0])
    if not xyzs:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)
    return np.array(xyzs, dtype=float), np.array(rgbs, dtype=float)


def _plot_sparse_point_cloud(
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    centers: np.ndarray,
    out_path: Path,
    title: str,
    max_points: int,
    crop_percentile: float | None = 1.0,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    pts = np.empty((0, 3), dtype=float)
    colors = np.empty((0, 3), dtype=float)
    if len(points_xyz) > 0:
        pts_all = points_xyz
        colors_all = points_rgb
        if crop_percentile is not None and 0.0 < crop_percentile < 50.0 and len(points_xyz) >= 32:
            lo = np.percentile(points_xyz, crop_percentile, axis=0)
            hi = np.percentile(points_xyz, 100.0 - crop_percentile, axis=0)
            keep = np.all((points_xyz >= lo) & (points_xyz <= hi), axis=1)
            if np.any(keep):
                pts_all = points_xyz[keep]
                colors_all = points_rgb[keep]

        if len(pts_all) > max_points:
            idx = np.linspace(0, len(pts_all) - 1, max_points, dtype=int)
            pts = pts_all[idx]
            colors = colors_all[idx]
        else:
            pts = pts_all
            colors = colors_all
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=1.2, alpha=0.65, linewidths=0)

    if len(centers) > 0:
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="tab:red", linewidth=1.4, label="camera path")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)
    if len(centers) > 0:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _cleanup_legacy_point_cloud_plots(case_root: Path) -> None:
    path = case_root / "sparse_points_full.png"
    if path.exists():
        path.unlink()


def _plot_merged_trajectories(
    trajectories: list[tuple[float, str, np.ndarray]],
    out_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.get_cmap("tab10", max(len(trajectories), 1))
    for idx, (fps, label, centers) in enumerate(trajectories):
        ax.plot(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            marker="o",
            markersize=2,
            linewidth=1.5,
            alpha=0.9,
            color=colors(idx),
            label=f"{label} ({len(centers)} poses)",
        )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    return umeyama_sim3(src, dst, error_cls=Task1Error)


def _apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return apply_sim3(points, scale, rot, trans)


def _format_float_tag(value: float) -> str:
    s = f"{value:.3f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _build_param_tag(cfg: Task1Config) -> str:
    # Keep parameterized outputs separate to avoid mixing results from different runs.
    return f"fps{_format_float_tag(cfg.fps)}"


def _parse_param_tag_value(param_tag: str) -> float:
    if not param_tag.startswith("fps"):
        raise Task1Error(f"Unsupported parameter tag: {param_tag}")
    value = param_tag[3:].replace("p", ".")
    try:
        return float(value)
    except ValueError as exc:
        raise Task1Error(f"Failed to parse fps from tag: {param_tag}") from exc


def _has_completed_outputs(case_root: Path) -> bool:
    images_dir = case_root / "images"
    sparse_dir = case_root / "sparse" / "0"
    trajectory = case_root / "trajectory.png"
    frame_map = case_root / FRAME_MAP_FILENAME
    return (
        images_dir.exists()
        and any(images_dir.glob("*.jpg"))
        and frame_map.exists()
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


def _parse_registered_pose_map(images_txt: Path) -> dict[str, np.ndarray]:
    centers, names = _parse_image_centers(images_txt)
    return {name: center for name, center in zip(names, centers, strict=True)}


def _plot_existing_case(
    case_root: Path,
    case_name: str,
    fps: float,
    direction_arrows: int,
    force: bool,
    dry_run: bool,
    timings: dict[str, float],
) -> None:
    images_txt = case_root / "sparse" / "0" / "images.txt"
    cameras_txt = case_root / "sparse" / "0" / "cameras.txt"
    points3d_txt = case_root / "sparse" / "0" / "points3D.txt"
    if not images_txt.exists() or not cameras_txt.exists() or not points3d_txt.exists():
        raise Task1Error(f"Missing sparse model text files under {case_root / 'sparse' / '0'}")

    trajectory_path = case_root / "trajectory.png"
    direction_path = case_root / "trajectory_with_directions.png"
    if dry_run:
        print(f"Would redraw trajectory plots from: {images_txt}")
        return
    if trajectory_path.exists() and direction_path.exists() and not force:
        print(f"Reuse trajectory plots: {case_root}")
        return

    with timed_block("plot", timings):
        centers, forward_dirs, _names = _parse_image_poses(images_txt)
        _plot_trajectory(
            centers,
            trajectory_path,
            f"{case_name} Camera Trajectory (fps={fps:g})",
        )
        _plot_trajectory(
            centers,
            direction_path,
            f"{case_name} Camera Trajectory + Viewing Directions (fps={fps:g})",
            forward_dirs=forward_dirs,
            direction_arrows=direction_arrows,
        )
    print(f"Saved trajectory: {trajectory_path}")
    print(f"Saved trajectory with directions: {direction_path}")


def _collect_existing_case_trajectories(
    output_root: Path,
    case_name: str,
) -> list[tuple[float, str, dict[str, np.ndarray], dict[str, int], Path]]:
    pattern = re.compile(rf"^{re.escape(case_name)}_(fps[0-9p]+)$")
    found: list[tuple[float, str, dict[str, np.ndarray], dict[str, int], Path]] = []
    if not output_root.exists():
        return found

    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        param_tag = match.group(1)
        images_txt = child / "sparse" / "0" / "images.txt"
        frame_map_path = child / FRAME_MAP_FILENAME
        if not images_txt.exists() or not frame_map_path.exists():
            continue
        pose_map = _parse_registered_pose_map(images_txt)
        frame_map = _load_frame_map(frame_map_path)
        found.append((_parse_param_tag_value(param_tag), param_tag, pose_map, frame_map, child))

    return sorted(found, key=lambda item: (item[0], item[1]))


def _align_trajectories_to_reference(
    trajectories: list[tuple[float, str, dict[str, np.ndarray], dict[str, int], Path]],
) -> tuple[list[tuple[float, str, np.ndarray]], list[dict[str, str | int | float]]]:
    ref_fps, ref_tag, ref_pose_map, ref_frame_map, _ref_case_root = trajectories[0]
    ref_pairs = {
        source_frame_idx: ref_pose_map[image_name]
        for image_name, source_frame_idx in ref_frame_map.items()
        if image_name in ref_pose_map
    }
    if len(ref_pairs) < 3:
        raise Task1Error(f"Reference trajectory {ref_tag} has fewer than 3 registered mapped frames.")

    ref_names_sorted = sorted(
        (source_idx, image_name) for image_name, source_idx in ref_frame_map.items() if image_name in ref_pose_map
    )
    ref_full = np.array([ref_pose_map[image_name] for _source_idx, image_name in ref_names_sorted])
    aligned = [(ref_fps, ref_tag, ref_full)]
    summary: list[dict[str, str | int | float]] = [
        {"param_tag": ref_tag, "fps": ref_fps, "common_frames": len(ref_pairs), "scale": 1.0}
    ]

    for fps, param_tag, pose_map, frame_map, _case_root in trajectories[1:]:
        cur_pairs = {
            source_frame_idx: pose_map[image_name]
            for image_name, source_frame_idx in frame_map.items()
            if image_name in pose_map
        }
        common_frame_indices = sorted(set(ref_pairs) & set(cur_pairs))
        if len(common_frame_indices) < 3:
            raise Task1Error(
                f"Need at least 3 common registered source frames to align {param_tag} to {ref_tag}, "
                f"found {len(common_frame_indices)}"
            )
        ref_common = np.array([ref_pairs[idx] for idx in common_frame_indices])
        cur_common = np.array([cur_pairs[idx] for idx in common_frame_indices])
        scale, rot, trans = _umeyama_sim3(cur_common, ref_common)
        cur_names_sorted = sorted(
            (source_idx, image_name) for image_name, source_idx in frame_map.items() if image_name in pose_map
        )
        cur_full = np.array([pose_map[image_name] for _source_idx, image_name in cur_names_sorted])
        cur_aligned = _apply_sim3(cur_full, scale, rot, trans)
        aligned.append((fps, param_tag, cur_aligned))
        summary.append(
            {
                "param_tag": param_tag,
                "fps": fps,
                "common_frames": len(common_frame_indices),
                "scale": scale,
            }
        )
    return aligned, summary


def _run_task1_merge(cfg: Task1Config, selected_videos: list[str], strict: bool) -> int:
    if cfg.stage != "all":
        raise Task1Error("task1 merge does not support --stage; it only reads existing outputs.")

    merged_root = cfg.output_root / "merged"
    if not cfg.dry_run:
        merged_root.mkdir(parents=True, exist_ok=True)

    for video_name in selected_videos:
        case_name = Path(video_name).stem
        timings: dict[str, float] = {}
        print(f"\n=== Task1 Merge / {case_name} ===")
        with timed_block("load_results", timings):
            trajectories = _collect_existing_case_trajectories(cfg.output_root, case_name)
        if len(trajectories) < 2:
            message = (
                f"Need at least 2 existing task1 results to merge for {case_name}, "
                f"found {len(trajectories)} under {cfg.output_root}"
            )
            if strict:
                raise Task1Error(message)
            print(f"Skip merge: {message}")
            continue

        merge_dir = merged_root / case_name
        out_path = merge_dir / "trajectory_overlay.png"
        summary_path = merge_dir / "alignment_summary.csv"
        timing_path = merge_dir / TIMING_FILENAME
        if cfg.dry_run:
            print(f"Would save merged trajectory: {out_path}")
            for fps, param_tag, _pose_map, _frame_map, case_root in trajectories:
                print(f"  source: {case_root} ({param_tag}, fps={fps:g})")
            print_timing_summary(f"Timing / {case_name} / merge", timings)
            continue

        merge_dir.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not cfg.force:
            print(f"Reuse merged trajectory: {out_path}")
            print_timing_summary(f"Timing / {case_name} / merge", timings)
            continue

        for fps, param_tag, _pose_map, _frame_map, case_root in trajectories:
            print(f"Source trajectory: {case_root} ({param_tag}, fps={fps:g})")
        with timed_block("align", timings):
            aligned_trajectories, alignment_summary = _align_trajectories_to_reference(trajectories)
        with timed_block("plot", timings):
            _plot_merged_trajectories(
                aligned_trajectories,
                out_path,
                f"{case_name} Camera Trajectory Overlay (aligned across fps)",
            )
        with timed_block("write_summary", timings):
            with summary_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["param_tag", "fps", "common_frames", "scale"])
                writer.writeheader()
                writer.writerows(alignment_summary)
        write_timing_csv(timing_path, timings)
        print(f"Saved merged trajectory: {out_path}")
        print(f"Saved alignment summary: {summary_path}")
        print(f"Saved timing summary: {timing_path}")
        print_timing_summary(f"Timing / {case_name} / merge", timings)

    print("\nTask1 merge completed.")
    return 0


def _run_task1_plot(cfg: Task1Config, selected_videos: list[str]) -> int:
    if cfg.stage != "all":
        raise Task1Error("task1 plot does not support --stage; it only reads existing outputs.")

    # Determine fps values to process
    if cfg.fps == ALL_FPS_SENTINEL:
        # Auto-discover existing fps for each video
        fps_dict: dict[str, list[float]] = {}
        for video_name in selected_videos:
            case_name = Path(video_name).stem
            fps_dict[case_name] = _discover_existing_fps(cfg.output_root, case_name)
            print(f"Discovered fps for {case_name}: {fps_dict[case_name]}")
    elif cfg.fps <= 0:
        raise Task1Error(f"fps must be positive or {ALL_FPS_SENTINEL} for all, got {cfg.fps}")
    else:
        # Single fps specified
        fps_dict = {Path(v).stem: [cfg.fps] for v in selected_videos}

    for video_name in selected_videos:
        case_name = Path(video_name).stem
        fps_list = fps_dict.get(case_name, [])

        if not fps_list:
            print(f"Skipping {case_name}: no existing results found")
            continue

        for fps in fps_list:
            param_tag = f"fps{_format_float_tag(fps)}"
            case_root = cfg.output_root / f"{case_name}_{param_tag}"
            timings: dict[str, float] = {}
            print(f"\n=== Task1 Plot / {case_name} / {param_tag} ===")
            if not case_root.exists():
                print(f"Skipping {case_root}: not found")
                continue
            _plot_existing_case(
                case_root=case_root,
                case_name=case_name,
                fps=fps,
                direction_arrows=cfg.direction_arrows,
                force=cfg.force,
                dry_run=cfg.dry_run,
                timings=timings,
            )
            if not cfg.dry_run:
                write_timing_csv(case_root / TIMING_FILENAME, timings)
                print(f"Saved timing summary: {case_root / TIMING_FILENAME}")
            print_timing_summary(f"Timing / {case_name} / {param_tag} / plot", timings)

    print("\nTask1 plot completed.")
    return 0


def _discover_existing_fps(output_root: Path, case_name: str) -> list[float]:
    """Discover all existing fps values for a given case."""
    fps_values: list[float] = []
    case_prefix = f"{case_name}_fps"
    for entry in output_root.iterdir():
        if entry.is_dir() and entry.name.startswith(case_prefix):
            # Extract fps from directory name like "S1-1_fps30"
            fps_str = entry.name[len(case_prefix):]
            try:
                fps_values.append(float(fps_str))
            except ValueError:
                continue
    return sorted(fps_values)


def _run_task1_cloud(cfg: Task1Config, selected_videos: list[str]) -> int:
    if cfg.stage != "all":
        raise Task1Error("task1 cloud does not support --stage; it only reads existing outputs.")

    # Determine fps values to process
    if cfg.fps == ALL_FPS_SENTINEL:
        # Auto-discover existing fps for each video
        fps_dict: dict[str, list[float]] = {}
        for video_name in selected_videos:
            case_name = Path(video_name).stem
            fps_dict[case_name] = _discover_existing_fps(cfg.output_root, case_name)
            print(f"Discovered fps for {case_name}: {fps_dict[case_name]}")
    elif cfg.fps <= 0:
        raise Task1Error(f"fps must be positive or {ALL_FPS_SENTINEL} for all, got {cfg.fps}")
    else:
        # Single fps specified
        fps_dict = {Path(v).stem: [cfg.fps] for v in selected_videos}

    for video_name in selected_videos:
        case_name = Path(video_name).stem
        fps_list = fps_dict.get(case_name, [])

        if not fps_list:
            print(f"Skipping {case_name}: no existing results found")
            continue

        for fps in fps_list:
            param_tag = f"fps{_format_float_tag(fps)}"
            case_root = cfg.output_root / f"{case_name}_{param_tag}"
            points3d_txt = case_root / "sparse" / "0" / "points3D.txt"
            images_txt = case_root / "sparse" / "0" / "images.txt"
            out_path = case_root / "sparse_points.png"
            timings: dict[str, float] = {}
            print(f"\n=== Task1 Cloud / {case_name} / {param_tag} ===")
            if not points3d_txt.exists() or not images_txt.exists():
                print(f"Skipping {case_root}: sparse model not found")
                continue
            if cfg.dry_run:
                print(f"Would generate sparse point cloud plot: {out_path}")
                continue
            if out_path.exists() and not cfg.force:
                print(f"Reuse sparse point cloud plot: {case_root}")
                continue
            with timed_block("cloud_plot", timings):
                _cleanup_legacy_point_cloud_plots(case_root)
                centers, _ = _parse_image_centers(images_txt)
                points_xyz, points_rgb = _parse_points3d(points3d_txt)
                _plot_sparse_point_cloud(
                    points_xyz,
                    points_rgb,
                    centers,
                    out_path,
                    f"{case_name} Sparse Point Cloud (fps={fps:g})",
                    max_points=cfg.max_points_plot,
                    crop_percentile=1.0,
                )
            write_timing_csv(case_root / "cloud_timing.csv", timings)
            print(f"Saved sparse point cloud plot: {out_path}")
            print_timing_summary(f"Timing / {case_name} / {param_tag} / cloud", timings)

    print("\nTask1 cloud completed.")
    return 0


def run_task1(cfg: Task1Config) -> int:
    videos_dir = cfg.lab1_root / "assets" / "videos"
    selected_videos = S1_VIDEOS if not cfg.videos else [_normalize_video_name(v) for v in cfg.videos]

    invalid = [v for v in selected_videos if v not in S1_VIDEOS]
    if invalid:
        raise Task1Error(
            f"Unsupported video(s): {invalid}. Supported: {S1_VIDEOS} "
            "(or short names S1-1/S1-2/S1-3)"
        )

    if cfg.mode == "merge":
        return _run_task1_merge(cfg, selected_videos, strict=cfg.videos is not None)
    if cfg.mode == "plot":
        return _run_task1_plot(cfg, selected_videos)
    if cfg.mode == "cloud":
        return _run_task1_cloud(cfg, selected_videos)
    if cfg.mode != "run":
        raise Task1Error(f"Unsupported task1 mode: {cfg.mode}")

    param_tag = _build_param_tag(cfg)
    if cfg.stage not in {"all", "extract", "sfm"}:
        raise Task1Error(f"Unsupported stage: {cfg.stage}. Choose from all|extract|sfm")
    if cfg.fps <= 0:
        raise Task1Error(f"fps must be positive, got {cfg.fps}")

    if not cfg.dry_run:
        if cfg.stage in {"all", "extract"}:
            require_tool(cfg.ffmpeg_bin, error_cls=Task1Error)
        if cfg.stage in {"all", "sfm"}:
            require_tool(cfg.colmap_bin, error_cls=Task1Error)

    def _finalize_case_timing(case_name: str, case_root: Path, param_tag: str, timings: dict[str, float]) -> None:
        if not cfg.dry_run:
            write_timing_csv(case_root / TIMING_FILENAME, timings)
            print(f"Saved timing summary: {case_root / TIMING_FILENAME}")
        print_timing_summary(f"Timing / {case_name} / {param_tag}", timings)

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

        print(f"\n=== Task1 / {case_name} / {param_tag} ===")
        if cfg.stage == "all" and not cfg.force and _has_completed_outputs(case_root):
            print(f"Reuse existing outputs (same parameters): {case_root}")
            _finalize_case_timing(case_name, case_root, param_tag, timings)
            continue

        if not cfg.dry_run:
            case_root.mkdir(parents=True, exist_ok=True)

        if cfg.stage in {"all", "extract"}:
            if not cfg.force and _has_any_frames(images_dir):
                print(f"Reuse extracted frames: {images_dir}")
            else:
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
        elif cfg.stage == "sfm":
            if not _has_any_frames(images_dir):
                raise Task1Error(
                    f"No extracted frames found under {images_dir}. "
                    "Run extraction first (--stage extract) or use --stage all."
                )
            if not frame_map_path.exists():
                raise Task1Error(
                    f"Frame map not found under {frame_map_path}. "
                    "Run extraction first (--stage extract) or use --stage all."
                )
            print(f"Reuse extracted frames: {images_dir}")

        if cfg.stage == "extract":
            print("Skip SfM for extract stage.")
            _finalize_case_timing(case_name, case_root, param_tag, timings)
            continue

        if cfg.stage == "sfm" and not cfg.force and _has_completed_sfm(case_root):
            print(f"Reuse SfM outputs: {case_root}")
            _finalize_case_timing(case_name, case_root, param_tag, timings)
            continue

        with timed_block("sfm_total", timings):
            model_dir = _run_colmap(
                images_dir=images_dir,
                sparse_root=sparse_root,
                db_path=db_path,
                colmap_bin=cfg.colmap_bin,
                feature_mask_path=None,
                camera_mask_path=None,
                force=cfg.force,
                dry_run=cfg.dry_run,
                timings=timings,
            )

        if cfg.dry_run:
            _finalize_case_timing(case_name, case_root, param_tag, timings)
            continue

        _plot_existing_case(
            case_root=case_root,
            case_name=case_name,
            fps=cfg.fps,
            direction_arrows=cfg.direction_arrows,
            force=True,
            dry_run=cfg.dry_run,
            timings=timings,
        )
        print(f"Sparse model text files: {model_dir}")
        _finalize_case_timing(case_name, case_root, param_tag, timings)

    print("\nTask1 completed.")
    return 0
