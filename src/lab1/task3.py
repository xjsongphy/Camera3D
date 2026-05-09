from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

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
    _normalize_video_name,
    _parse_image_centers,
    _parse_image_poses,
    _plot_trajectory,
    _run_colmap,
)

S2_VIDEOS = ["S2-1.mp4", "S2-2.mp4"]
TASK3_METHODS = ("raw", "static_roi_mask", "motion_mask", "semantic_mask")


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
    methods: tuple[str, ...] = ("raw",)
    semantic_mask_root: Path | None = None
    motion_threshold: int = 28
    motion_dilation: int = 9
    direction_arrows: int = 10
    max_points_plot: int = 12000


def _normalize_method_name(name: str) -> str:
    value = name.strip().lower()
    aliases = {
        "baseline": "raw",
        "camera_mask": "static_roi_mask",
        "mask": "static_roi_mask",
        "roi_mask": "static_roi_mask",
        "static": "static_roi_mask",
        "motion": "motion_mask",
        "semantic": "semantic_mask",
    }
    value = aliases.get(value, value)
    if value not in TASK3_METHODS:
        raise Task1Error(f"Unsupported task3 method: {name}. Supported: {TASK3_METHODS}")
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
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if len(points_xyz) > 0:
        if len(points_xyz) > max_points:
            idx = np.linspace(0, len(points_xyz) - 1, max_points, dtype=int)
            pts = points_xyz[idx]
            colors = points_rgb[idx]
        else:
            pts = points_xyz
            colors = points_rgb
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


def _write_analysis(
    *,
    method_root: Path,
    case_name: str,
    method: str,
    fps: float,
    total_frames: int,
    registered_frames: int,
    points3d_count: int,
    median_step: float,
    max_step: float,
    jump_ratio: float,
) -> dict[str, str | int | float]:
    registration_ratio = (registered_frames / total_frames) if total_frames > 0 else 0.0
    analysis_txt = method_root / "analysis.txt"
    analysis_csv = method_root / "analysis.csv"

    lines = [
        f"video={case_name}",
        f"method={method}",
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

    row = {
        "video": case_name,
        "method": method,
        "fps": f"{fps:g}",
        "total_frames": total_frames,
        "registered_frames": registered_frames,
        "registration_ratio": f"{registration_ratio:.8f}",
        "points3d": points3d_count,
        "trajectory_step_median": f"{median_step:.8f}",
        "trajectory_step_max": f"{max_step:.8f}",
        "trajectory_jump_ratio": f"{jump_ratio:.8f}",
    }
    with analysis_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return row


def _save_method_summary(base_root: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    out_path = base_root / "method_summary.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_gray_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _resize_gray(arr: np.ndarray, width: int = 160) -> np.ndarray:
    h, w = arr.shape
    if w <= width:
        return arr
    height = max(1, int(round(h * width / w)))
    img = Image.fromarray(arr).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _estimate_translation(ref: np.ndarray, src: np.ndarray, max_shift: int = 8) -> tuple[int, int]:
    best_score: float | None = None
    best = (0, 0)
    h, w = ref.shape
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y0_ref = max(0, dy)
            y1_ref = min(h, h + dy)
            x0_ref = max(0, dx)
            x1_ref = min(w, w + dx)
            y0_src = max(0, -dy)
            y1_src = min(h, h - dy)
            x0_src = max(0, -dx)
            x1_src = min(w, w - dx)
            if y1_ref - y0_ref < h // 2 or x1_ref - x0_ref < w // 2:
                continue
            diff = ref[y0_ref:y1_ref, x0_ref:x1_ref].astype(np.int16) - src[y0_src:y1_src, x0_src:x1_src].astype(np.int16)
            score = float(np.mean(np.abs(diff)))
            if best_score is None or score < best_score:
                best_score = score
                best = (dy, dx)
    return best


def _shift_image(arr: np.ndarray, dy: int, dx: int, fill: np.ndarray) -> np.ndarray:
    out = fill.copy()
    h, w = arr.shape
    y0_dst = max(0, dy)
    y1_dst = min(h, h + dy)
    x0_dst = max(0, dx)
    x1_dst = min(w, w + dx)
    y0_src = max(0, -dy)
    y1_src = min(h, h - dy)
    x0_src = max(0, -dx)
    x1_src = min(w, w - dx)
    out[y0_dst:y1_dst, x0_dst:x1_dst] = arr[y0_src:y1_src, x0_src:x1_src]
    return out


def _build_dynamic_mask(prev_img: np.ndarray | None, cur_img: np.ndarray, next_img: np.ndarray | None, threshold: int) -> np.ndarray:
    cur_small = _resize_gray(cur_img)
    masks: list[np.ndarray] = []
    for neighbor in (prev_img, next_img):
        if neighbor is None:
            continue
        neighbor_small = _resize_gray(neighbor)
        dy, dx = _estimate_translation(cur_small, neighbor_small)
        scale_y = cur_img.shape[0] / cur_small.shape[0]
        scale_x = cur_img.shape[1] / cur_small.shape[1]
        shifted = _shift_image(neighbor, int(round(dy * scale_y)), int(round(dx * scale_x)), fill=cur_img)
        masks.append(np.abs(cur_img.astype(np.int16) - shifted.astype(np.int16)) > threshold)
    if not masks:
        return np.ones_like(cur_img, dtype=np.uint8) * 255
    dynamic = masks[0] if len(masks) == 1 else np.logical_and.reduce(masks)
    # Dynamic objects are usually concentrated in the lower field of view for these videos.
    cutoff = int(round(cur_img.shape[0] * 0.30))
    dynamic[:cutoff, :] = False
    mask = np.where(dynamic, 0, 255).astype(np.uint8)
    return mask


def _write_motion_masks(images_dir: Path, mask_dir: Path, threshold: int, dilation: int) -> None:
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise Task1Error(f"No extracted frames found under {images_dir}")
    mask_dir.mkdir(parents=True, exist_ok=True)
    dilation = max(3, dilation)
    if dilation % 2 == 0:
        dilation += 1

    # Use sliding window to load each frame only once
    prev_img: np.ndarray | None = None
    cur_img = _load_gray_image(image_paths[0])

    for idx in range(len(image_paths)):
        next_img = _load_gray_image(image_paths[idx + 1]) if idx + 1 < len(image_paths) else None
        mask = _build_dynamic_mask(prev_img, cur_img, next_img, threshold)
        mask_img = Image.fromarray(mask, mode="L").filter(ImageFilter.MaxFilter(size=dilation))
        mask_img.save(mask_dir / f"{image_paths[idx].name}.png")

        prev_img = cur_img
        if next_img is not None:
            cur_img = next_img


def _write_static_roi_camera_mask(images_dir: Path, out_path: Path, case_name: str) -> None:
    first_image = next(iter(sorted(images_dir.glob("*.jpg"))), None)
    if first_image is None:
        raise Task1Error(f"No extracted frames found under {images_dir}")
    with Image.open(first_image) as img:
        width, height = img.size
    mask = np.zeros((height, width), dtype=np.uint8)
    if case_name == "S2-1":
        cutoff = max(1, int(round(height * 0.20)))
        mask[:cutoff, :] = 255
    elif case_name == "S2-2":
        cutoff_y = max(1, int(round(height * 0.50)))
        cutoff_x = max(1, int(round(width * 0.50)))
        mask[:cutoff_y, :cutoff_x] = 255
    else:
        raise Task1Error(f"Unsupported static ROI prior for case: {case_name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_path)


def _resolve_semantic_mask_dir(root: Path, case_name: str, param_tag: str) -> Path:
    return root / f"{case_name}_{param_tag}"


def _check_semantic_masks(mask_dir: Path, images_dir: Path) -> None:
    if not mask_dir.exists():
        raise Task1Error(
            f"Semantic mask directory not found: {mask_dir}. "
            "Generate external per-image masks first."
        )
    missing = [p.name for p in sorted(images_dir.glob('*.jpg')) if not (mask_dir / f"{p.name}.png").exists()]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise Task1Error(
            f"Semantic masks incomplete under {mask_dir}. Missing {len(missing)} file(s): {preview}{suffix}"
        )


def _prepare_method_inputs(
    *,
    method: str,
    base_root: Path,
    images_dir: Path,
    case_name: str,
    param_tag: str,
    semantic_mask_root: Path | None,
    motion_threshold: int,
    motion_dilation: int,
    force: bool,
    dry_run: bool,
) -> tuple[Path | None, Path | None]:
    method_root = base_root / method
    mask_path: Path | None = None
    camera_mask_path: Path | None = None

    if method == "raw":
        return None, None

    if method == "static_roi_mask":
        camera_mask_path = method_root / "camera_mask.png"
        if dry_run:
            print(f"Would create static ROI camera mask: {camera_mask_path}")
            return None, camera_mask_path
        if not camera_mask_path.exists() or force:
            _write_static_roi_camera_mask(images_dir, camera_mask_path, case_name)
        return None, camera_mask_path

    if method == "motion_mask":
        mask_path = method_root / "masks"
        if dry_run:
            print(f"Would create motion masks under: {mask_path}")
            return mask_path, None
        if force and mask_path.exists():
            for child in mask_path.glob("*.png"):
                child.unlink()
        if force or not any(mask_path.glob("*.png")):
            _write_motion_masks(images_dir, mask_path, motion_threshold, motion_dilation)
        return mask_path, None

    if method == "semantic_mask":
        if semantic_mask_root is None:
            raise Task1Error(
                "semantic_mask method requires --semantic-mask-root. "
                "Masks must follow COLMAP's per-image .png naming convention."
            )
        mask_path = _resolve_semantic_mask_dir(semantic_mask_root, case_name, param_tag)
        if dry_run:
            print(f"Would use semantic masks from: {mask_path}")
            return mask_path, None
        _check_semantic_masks(mask_path, images_dir)
        return mask_path, None

    raise Task1Error(f"Unsupported task3 method: {method}")


def _run_method_sfm(
    *,
    method_root: Path,
    images_dir: Path,
    colmap_bin: str,
    force: bool,
    dry_run: bool,
    timings: dict[str, float],
    feature_mask_path: Path | None,
    camera_mask_path: Path | None,
) -> Path:
    sparse_root = method_root / "sparse"
    db_path = method_root / "database.db"
    return _run_colmap(
        images_dir=images_dir,
        sparse_root=sparse_root,
        db_path=db_path,
        colmap_bin=colmap_bin,
        feature_mask_path=feature_mask_path,
        camera_mask_path=camera_mask_path,
        force=force,
        dry_run=dry_run,
        timings=timings,
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

    methods = tuple(dict.fromkeys(_normalize_method_name(m) for m in cfg.methods))

    if not cfg.dry_run:
        if cfg.stage in {"all", "extract"}:
            require_tool(cfg.ffmpeg_bin, error_cls=Task1Error)
        if cfg.stage in {"all", "sfm"}:
            require_tool(cfg.colmap_bin, error_cls=Task1Error)

    param_tag = f"fps{_format_float_tag(cfg.fps)}"

    for video_name in selected_videos:
        video_path = videos_dir / video_name
        if not video_path.exists():
            raise Task1Error(f"Video not found: {video_path}")

        case_name = video_path.stem
        base_root = cfg.output_root / f"{case_name}_{param_tag}"
        images_dir = base_root / "images"
        frame_map_path = base_root / FRAME_MAP_FILENAME
        print(f"\n=== Task3 / {case_name} / {param_tag} ===")
        if not cfg.dry_run:
            base_root.mkdir(parents=True, exist_ok=True)

        extract_timings: dict[str, float] = {}
        if cfg.stage in {"all", "extract"}:
            if not cfg.force and _has_any_frames(images_dir):
                print(f"Reuse extracted frames: {images_dir}")
            else:
                with timed_block("extract", extract_timings):
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
                write_timing_csv(base_root / TIMING_FILENAME, extract_timings)
                print(f"Saved timing summary: {base_root / TIMING_FILENAME}")
            print_timing_summary(f"Timing / {case_name} / {param_tag} / extract", extract_timings)
            continue

        if not cfg.dry_run and not _has_any_frames(images_dir):
            raise Task1Error(
                f"No extracted frames found under {images_dir}. "
                "Run extraction first (--stage extract) or use --stage all."
            )

        summary_rows: list[dict[str, str | int | float]] = []
        for method in methods:
            timings: dict[str, float] = {}
            method_root = base_root / method
            timing_path = method_root / TIMING_FILENAME
            model_dir = method_root / "sparse" / "0"
            print(f"\n--- Method: {method} ---")
            if not cfg.dry_run:
                method_root.mkdir(parents=True, exist_ok=True)

            feature_mask_path: Path | None = None
            camera_mask_path: Path | None = None
            if cfg.stage in {"all", "sfm"}:
                with timed_block("prepare_inputs", timings):
                    feature_mask_path, camera_mask_path = _prepare_method_inputs(
                        method=method,
                        base_root=base_root,
                        images_dir=images_dir,
                        case_name=case_name,
                        param_tag=param_tag,
                        semantic_mask_root=cfg.semantic_mask_root,
                        motion_threshold=cfg.motion_threshold,
                        motion_dilation=cfg.motion_dilation,
                        force=cfg.force,
                        dry_run=cfg.dry_run,
                    )
                with timed_block("sfm_total", timings):
                    model_dir = _run_method_sfm(
                        method_root=method_root,
                        images_dir=images_dir,
                        colmap_bin=cfg.colmap_bin,
                        force=cfg.force,
                        dry_run=cfg.dry_run,
                        timings=timings,
                        feature_mask_path=feature_mask_path,
                        camera_mask_path=camera_mask_path,
                    )
                if not cfg.dry_run:
                    with timed_block("plot", timings):
                        centers, forward_dirs, _ = _parse_image_poses(model_dir / "images.txt")
                        _plot_trajectory(
                            centers,
                            method_root / "trajectory_raw.png",
                            f"{case_name} {method} Trajectory (fps={cfg.fps:g})",
                        )
                        _plot_trajectory(
                            centers,
                            method_root / "trajectory_with_directions.png",
                            f"{case_name} {method} Trajectory + Directions (fps={cfg.fps:g})",
                            forward_dirs=forward_dirs,
                            direction_arrows=cfg.direction_arrows,
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
                        points_xyz, points_rgb = _parse_points3d(points3d_txt)
                        median_step, max_step, jump_ratio = _compute_jump_stats(centers)

                        summary_rows.append(
                            _write_analysis(
                                method_root=method_root,
                                case_name=case_name,
                                method=method,
                                fps=cfg.fps,
                                total_frames=total_frames,
                                registered_frames=registered_frames,
                                points3d_count=len(points_xyz),
                                median_step=median_step,
                                max_step=max_step,
                                jump_ratio=jump_ratio,
                            )
                        )
                        _plot_sparse_point_cloud(
                            points_xyz,
                            points_rgb,
                            centers,
                            method_root / "sparse_points.png",
                            f"{case_name} {method} Sparse Point Cloud",
                            max_points=cfg.max_points_plot,
                        )
                    print(f"Saved analysis: {method_root / 'analysis.txt'}")
                    print(f"Saved point cloud plot: {method_root / 'sparse_points.png'}")

            if not cfg.dry_run:
                write_timing_csv(timing_path, timings)
                print(f"Saved timing summary: {timing_path}")
            print_timing_summary(f"Timing / {case_name} / {param_tag} / {method}", timings)

        if summary_rows and not cfg.dry_run:
            _save_method_summary(base_root, summary_rows)
            print(f"Saved method summary: {base_root / 'method_summary.csv'}")

    print("\nTask3 completed.")
    return 0
