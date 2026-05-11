from __future__ import annotations

import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from lab1.colmap_utils import require_tool, run_cmd
from lab1.logging_utils import print_timing_summary, timed_block, write_timing_csv
from lab1.task1 import (
    FRAME_MAP_FILENAME,
    TIMING_FILENAME,
    Task1Error,
    _apply_sim3,
    _count_registered_images_in_model,
    _extract_frames,
    _format_float_tag,
    _has_any_frames,
    _normalize_video_name,
    _parse_image_centers,
    _parse_points3d,
    _parse_image_poses,
    _parse_registered_pose_map,
    _plot_sparse_point_cloud,
    _plot_merged_trajectories,
    _plot_trajectory,
    _run_colmap,
    _umeyama_sim3,
)

S2_VIDEOS = ["S2-1.mp4", "S2-2.mp4"]
TASK3_METHODS = ("raw", "mask")
TASK3_MASK_SOURCES = ("default", "motion", "yolo")
TASK3_YOLO_DYNAMIC_CLASS_IDS = (0, 1, 2, 3, 5, 7)
TASK3_MASK_DIRNAME = "masks"


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
    mask_source: str = "default"
    direction_arrows: int = 10
    max_points_plot: int = 12000


@dataclass
class Task3MaskConfig:
    lab1_root: Path
    output_root: Path
    fps: float
    ffmpeg_bin: str
    force: bool
    dry_run: bool
    videos: list[str] | None = None
    source: str = "default"
    motion_threshold: int = 28
    motion_dilation: int = 9
    model: str = "models/yolo11s-seg.pt"
    conf: float = 0.25
    imgsz: int = 960
    yolo_dilation: int = 7


@dataclass
class PointCloudQuality:
    xyz: np.ndarray
    rgb: np.ndarray
    reproj_error: np.ndarray
    track_length: np.ndarray


def _normalize_method_name(name: str) -> str:
    value = name.strip().lower()
    aliases = {
        "baseline": "raw",
        "masked": "mask",
    }
    value = aliases.get(value, value)
    if value not in TASK3_METHODS:
        raise Task1Error(f"Unsupported task3 method: {name}. Supported: {TASK3_METHODS}")
    return value


def _normalize_mask_source(name: str) -> str:
    value = name.strip().lower()
    aliases = {
        "roi": "default",
        "static": "default",
        "semantic": "yolo",
    }
    value = aliases.get(value, value)
    if value not in TASK3_MASK_SOURCES:
        raise Task1Error(f"Unsupported task3 mask source: {name}. Supported: {TASK3_MASK_SOURCES}")
    return value


def _default_mask_root(output_root: Path, source: str) -> Path:
    return output_root / TASK3_MASK_DIRNAME / source


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


def _parse_points3d_quality(points3d_txt: Path) -> PointCloudQuality:
    xyzs: list[list[float]] = []
    rgbs: list[list[float]] = []
    reproj_error: list[float] = []
    track_length: list[int] = []
    with points3d_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            xyzs.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgbs.append([float(parts[4]) / 255.0, float(parts[5]) / 255.0, float(parts[6]) / 255.0])
            reproj_error.append(float(parts[7]))
            track_length.append((len(parts) - 8) // 2)
    if not xyzs:
        empty_xyz = np.empty((0, 3), dtype=float)
        empty_rgb = np.empty((0, 3), dtype=float)
        empty_scalar = np.empty((0,), dtype=float)
        return PointCloudQuality(
            xyz=empty_xyz,
            rgb=empty_rgb,
            reproj_error=empty_scalar,
            track_length=empty_scalar.astype(int),
        )
    return PointCloudQuality(
        xyz=np.array(xyzs, dtype=float),
        rgb=np.array(rgbs, dtype=float),
        reproj_error=np.array(reproj_error, dtype=float),
        track_length=np.array(track_length, dtype=int),
    )


def _build_reliable_point_mask(point_cloud: PointCloudQuality) -> np.ndarray:
    if len(point_cloud.xyz) == 0:
        return np.zeros((0,), dtype=bool)

    xyz = point_cloud.xyz
    reproj_error = point_cloud.reproj_error
    track_length = point_cloud.track_length

    spatial_lo = np.percentile(xyz, 1.0, axis=0)
    spatial_hi = np.percentile(xyz, 99.0, axis=0)
    spatial_mask = np.all((xyz >= spatial_lo) & (xyz <= spatial_hi), axis=1)

    error_threshold = min(float(np.percentile(reproj_error, 90.0)), 2.0)
    track_threshold = max(4, int(np.percentile(track_length, 30.0)))
    quality_mask = (reproj_error <= error_threshold) & (track_length >= track_threshold)

    reliable = spatial_mask & quality_mask
    if np.count_nonzero(reliable) < max(128, len(xyz) // 50):
        reliable = quality_mask
    return reliable


def _summarize_point_cloud_quality(point_cloud: PointCloudQuality, reliable_mask: np.ndarray) -> dict[str, int | float]:
    if len(point_cloud.xyz) == 0:
        return {
            "points3d_reliable": 0,
            "points3d_reliable_ratio": 0.0,
            "reproj_error_median": 0.0,
            "reproj_error_p90": 0.0,
            "reproj_error_p99": 0.0,
            "track_length_median": 0.0,
            "track_length_p10": 0.0,
        }
    return {
        "points3d_reliable": int(np.count_nonzero(reliable_mask)),
        "points3d_reliable_ratio": float(np.mean(reliable_mask)),
        "reproj_error_median": float(np.median(point_cloud.reproj_error)),
        "reproj_error_p90": float(np.percentile(point_cloud.reproj_error, 90.0)),
        "reproj_error_p99": float(np.percentile(point_cloud.reproj_error, 99.0)),
        "track_length_median": float(np.median(point_cloud.track_length)),
        "track_length_p10": float(np.percentile(point_cloud.track_length, 10.0)),
    }


def _cleanup_legacy_point_cloud_plots(method_root: Path) -> None:
    for name in ("sparse_points_full.png", "sparse_points_reliable.png"):
        path = method_root / name
        if path.exists():
            path.unlink()



def _write_analysis(
    *,
    method_root: Path,
    case_name: str,
    method: str,
    mask_source: str | None,
    fps: float,
    total_frames: int,
    registered_frames: int,
    points3d_count: int,
    points3d_reliable: int,
    points3d_reliable_ratio: float,
    reproj_error_median: float,
    reproj_error_p90: float,
    reproj_error_p99: float,
    track_length_median: float,
    track_length_p10: float,
    median_step: float,
    max_step: float,
    jump_ratio: float,
) -> dict[str, str | int | float]:
    method_variant = method if method != "mask" or not mask_source else f"mask_{mask_source}"
    registration_ratio = (registered_frames / total_frames) if total_frames > 0 else 0.0
    analysis_txt = method_root / "analysis.txt"
    analysis_csv = method_root / "analysis.csv"

    lines = [
        f"video={case_name}",
        f"method={method}",
        f"method_variant={method_variant}",
        f"mask_source={mask_source or ''}",
        f"fps={fps:g}",
        f"total_frames={total_frames}",
        f"registered_frames={registered_frames}",
        f"registration_ratio={registration_ratio:.8f}",
        f"points3d={points3d_count}",
        f"points3d_reliable={points3d_reliable}",
        f"points3d_reliable_ratio={points3d_reliable_ratio:.8f}",
        f"reproj_error_median={reproj_error_median:.8f}",
        f"reproj_error_p90={reproj_error_p90:.8f}",
        f"reproj_error_p99={reproj_error_p99:.8f}",
        f"track_length_median={track_length_median:.8f}",
        f"track_length_p10={track_length_p10:.8f}",
        f"trajectory_step_median={median_step:.8f}",
        f"trajectory_step_max={max_step:.8f}",
        f"trajectory_jump_ratio={jump_ratio:.8f}",
    ]
    analysis_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    row = {
        "video": case_name,
        "method": method,
        "method_variant": method_variant,
        "mask_source": mask_source or "",
        "fps": f"{fps:g}",
        "total_frames": total_frames,
        "registered_frames": registered_frames,
        "registration_ratio": f"{registration_ratio:.8f}",
        "points3d": points3d_count,
        "points3d_reliable": points3d_reliable,
        "points3d_reliable_ratio": f"{points3d_reliable_ratio:.8f}",
        "reproj_error_median": f"{reproj_error_median:.8f}",
        "reproj_error_p90": f"{reproj_error_p90:.8f}",
        "reproj_error_p99": f"{reproj_error_p99:.8f}",
        "track_length_median": f"{track_length_median:.8f}",
        "track_length_p10": f"{track_length_p10:.8f}",
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
    existing_rows: list[dict[str, str | int | float]] = []
    out_path = base_root / "method_summary.csv"
    if out_path.exists():
        with out_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = []
            for row in reader:
                if row.get("method") == "mask" and not row.get("method_variant"):
                    continue
                existing_rows.append(row)

    merged: dict[str, dict[str, str | int | float]] = {}
    for row in existing_rows + rows:
        key = str(row.get("method_variant") or row.get("method", ""))
        merged[key] = row

    final_rows = list(merged.values())
    fieldnames: list[str] = []
    for row in final_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)


def _collect_existing_method_trajectories(
    base_root: Path,
) -> list[tuple[str, str, dict[str, np.ndarray]]]:
    candidates = [
        ("raw", "raw", base_root / "raw"),
        ("mask_default", "default", base_root / "mask_default"),
        ("mask_motion", "motion", base_root / "mask_motion"),
        ("mask_yolo", "yolo", base_root / "mask_yolo"),
    ]
    found: list[tuple[str, str, dict[str, np.ndarray]]] = []
    for method_variant, label, method_root in candidates:
        images_txt = method_root / "sparse" / "0" / "images.txt"
        if not images_txt.exists():
            continue
        pose_map = _parse_registered_pose_map(images_txt)
        if not pose_map:
            continue
        found.append((method_variant, label, pose_map))
    return found


def _align_method_trajectories_to_reference(
    trajectories: list[tuple[str, str, dict[str, np.ndarray]]],
) -> tuple[list[tuple[float, str, np.ndarray]], list[dict[str, str | int | float]]]:
    ref_variant, ref_label, ref_pose_map = trajectories[0]
    ref_names_sorted = sorted(ref_pose_map)
    ref_full = np.array([ref_pose_map[name] for name in ref_names_sorted])
    aligned: list[tuple[float, str, np.ndarray]] = [(0.0, ref_label, ref_full)]
    summary: list[dict[str, str | int | float]] = [
        {"method_variant": ref_variant, "label": ref_label, "common_frames": len(ref_names_sorted), "scale": 1.0}
    ]

    for method_variant, label, pose_map in trajectories[1:]:
        common_names = sorted(set(ref_pose_map) & set(pose_map))
        if len(common_names) < 3:
            raise Task1Error(
                f"Need at least 3 common registered frames to align {method_variant} to {ref_variant}, "
                f"found {len(common_names)}"
            )
        ref_common = np.array([ref_pose_map[name] for name in common_names])
        cur_common = np.array([pose_map[name] for name in common_names])
        scale, rot, trans = _umeyama_sim3(cur_common, ref_common)
        cur_names_sorted = sorted(pose_map)
        cur_full = np.array([pose_map[name] for name in cur_names_sorted])
        cur_aligned = _apply_sim3(cur_full, scale, rot, trans)
        aligned.append((0.0, label, cur_aligned))
        summary.append(
            {
                "method_variant": method_variant,
                "label": label,
                "common_frames": len(common_names),
                "scale": scale,
            }
        )
    return aligned, summary


def _write_merged_method_trajectory(
    *,
    base_root: Path,
    case_name: str,
    force: bool,
    dry_run: bool,
    timings: dict[str, float],
) -> None:
    trajectories = _collect_existing_method_trajectories(base_root)
    if len(trajectories) < 2:
        print(f"Skip merged trajectory: found only {len(trajectories)} method(s) under {base_root}")
        return

    out_path = base_root / "trajectory_overlay.png"
    summary_path = base_root / "trajectory_overlay_summary.csv"
    if dry_run:
        print(f"Would save merged method trajectory: {out_path}")
        return
    if out_path.exists() and not force:
        print(f"Reuse merged method trajectory: {out_path}")
        return

    with timed_block("merge_align", timings):
        aligned_trajectories, alignment_summary = _align_method_trajectories_to_reference(trajectories)
    with timed_block("merge_plot", timings):
        _plot_merged_trajectories(
            aligned_trajectories,
            out_path,
            f"{case_name} Trajectory Overlay Across Methods",
        )
    with timed_block("merge_write_summary", timings):
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method_variant", "label", "common_frames", "scale"])
            writer.writeheader()
            writer.writerows(alignment_summary)
    print(f"Saved merged trajectory: {out_path}")
    print(f"Saved merged trajectory summary: {summary_path}")


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
    print(f"Motion mask device: {_select_auto_device()}")

    # Use sliding window to load each frame only once
    prev_img: np.ndarray | None = None
    cur_img = _load_gray_image(image_paths[0])

    for idx in range(len(image_paths)):
        next_img = _load_gray_image(image_paths[idx + 1]) if idx + 1 < len(image_paths) else None
        mask = _build_dynamic_mask(prev_img, cur_img, next_img, threshold)
        dilated = _dilate_dynamic_mask(mask, dilation)
        Image.fromarray(dilated, mode="L").save(mask_dir / f"{image_paths[idx].name}.png")

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


def _resolve_mask_dir(root: Path, case_name: str, param_tag: str) -> Path:
    return root / f"{case_name}_{param_tag}"


def _resolve_default_mask_path(root: Path, case_name: str, param_tag: str) -> Path:
    return _resolve_mask_dir(root, case_name, param_tag) / "camera_mask.png"


def _check_mask_dir(mask_dir: Path, images_dir: Path, *, missing_hint: str | None = None) -> None:
    if not mask_dir.exists():
        message = f"Mask directory not found: {mask_dir}. Generate masks first."
        if missing_hint:
            message += f" {missing_hint}"
        raise Task1Error(message)
    missing = [p.name for p in sorted(images_dir.glob('*.jpg')) if not (mask_dir / f"{p.name}.png").exists()]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        message = f"Masks incomplete under {mask_dir}. Missing {len(missing)} file(s): {preview}{suffix}"
        if missing_hint:
            message += f" {missing_hint}"
        raise Task1Error(message)


def _check_default_camera_mask(mask_path: Path, *, missing_hint: str | None = None) -> None:
    if mask_path.exists():
        return
    message = f"Camera mask file not found: {mask_path}."
    if missing_hint:
        message += f" {missing_hint}"
    raise Task1Error(message)


def _is_default_camera_mask_ready(mask_path: Path) -> bool:
    return mask_path.exists()


def _is_mask_dir_complete(mask_dir: Path, images_dir: Path) -> bool:
    if not mask_dir.exists():
        return False
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        return False
    for image_path in image_paths:
        if not (mask_dir / f"{image_path.name}.png").exists():
            return False
    return True


def _build_missing_mask_hint(case_name: str, fps: float, source: str) -> str:
    return f"Run: uv run lab1 task3-mask --source {source} --videos {case_name} --fps {fps:g}"


def _is_method_run_complete(method_root: Path) -> bool:
    """Check if a method run has complete outputs."""
    images_txt = method_root / "sparse" / "0" / "images.txt"
    points3d_txt = method_root / "sparse" / "0" / "points3D.txt"
    analysis_txt = method_root / "analysis.txt"
    return images_txt.exists() and points3d_txt.exists() and analysis_txt.exists()


def _ensure_ultralytics_available() -> None:
    if importlib.util.find_spec("ultralytics") is None:
        raise Task1Error(
            "Ultralytics is not installed in the current uv environment. "
            "Run: uv sync --extra task3-yolo"
        )


def _torch_cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


def _select_auto_device() -> str:
    return "cuda:0" if _torch_cuda_available() else "cpu"


def _dilate_dynamic_mask(mask: np.ndarray, dilation: int) -> np.ndarray:
    if dilation <= 1:
        return mask
    import torch
    import torch.nn.functional as F

    device = _select_auto_device()
    dynamic = (mask == 0).astype(np.float32)
    t = torch.from_numpy(dynamic).to(device)
    expanded = F.max_pool2d(t.unsqueeze(0).unsqueeze(0), kernel_size=dilation, stride=1, padding=dilation // 2)
    out = (expanded.squeeze(0).squeeze(0) > 0.5).to(torch.uint8)
    return np.where(out.cpu().numpy() > 0, 0, 255).astype(np.uint8)


def _resize_binary_mask(mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    width, height = target_size
    if mask.shape == (height, width):
        return mask.astype(bool)
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.uint8) > 0


def _load_yolo_segmenter(model_name: str):
    _ensure_ultralytics_available()
    from ultralytics import YOLO

    return YOLO(model_name)


def _collect_yolo_dynamic_mask(result, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return np.zeros((height, width), dtype=bool)
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    mask_data = result.masks.data.detach().cpu().numpy()
    dynamic = np.zeros(mask_data.shape[1:], dtype=bool)
    for idx, class_id in enumerate(classes):
        if class_id in TASK3_YOLO_DYNAMIC_CLASS_IDS:
            dynamic |= mask_data[idx] > 0.5
    return _resize_binary_mask(dynamic, (width, height))


def _write_yolo_masks_for_images(
    *,
    images_dir: Path,
    mask_dir: Path,
    model_name: str,
    conf: float,
    imgsz: int,
    dilation: int,
) -> None:
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise Task1Error(f"No extracted frames found under {images_dir}")

    mask_dir.mkdir(parents=True, exist_ok=True)
    dilation = max(1, dilation)
    if dilation % 2 == 0:
        dilation += 1

    model = _load_yolo_segmenter(model_name)
    predict_device = _select_auto_device()
    print(f"YOLO device: {predict_device}")

    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size
        results = model.predict(
            source=str(image_path),
            save=False,
            verbose=False,
            conf=conf,
            imgsz=imgsz,
            device=predict_device,
            classes=list(TASK3_YOLO_DYNAMIC_CLASS_IDS),
            retina_masks=True,
        )
        dynamic = _collect_yolo_dynamic_mask(results[0], (width, height))
        mask = np.where(dynamic, 0, 255).astype(np.uint8)
        mask_img = Image.fromarray(mask, mode="L")
        if dilation > 1:
            # Mask convention: dynamic=0 (black), static=255 (white).
            # To dilate dynamic regions, use MinFilter so black expands.
            mask_img = mask_img.filter(ImageFilter.MinFilter(size=dilation))
        mask_img.save(mask_dir / f"{image_path.name}.png")


def _compose_overlay_frame(image_path: Path, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    if mask.shape != rgb.shape[:2]:
        mask = np.asarray(Image.fromarray(mask, mode="L").resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST), dtype=np.uint8)
    blocked = mask == 0

    overlay = rgb.copy().astype(np.float32)
    red = np.array([255.0, 40.0, 40.0], dtype=np.float32)
    overlay[blocked] = 0.55 * overlay[blocked] + 0.45 * red
    overlay = overlay.clip(0, 255).astype(np.uint8)

    panel = np.concatenate([rgb, overlay], axis=1)
    return Image.fromarray(panel, mode="RGB")


def _export_mask_overlay_video(
    *,
    images_dir: Path,
    mask_dir: Path,
    camera_mask_path: Path | None,
    ffmpeg_bin: str,
    out_video_path: Path,
    output_fps: float,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise Task1Error(f"No extracted frames found under {images_dir}")
    if out_video_path.exists() and not force:
        print(f"Reuse mask overlay video: {out_video_path}")
        return
    if dry_run:
        print(f"Would export mask overlay video: {out_video_path}")
        return

    tmp_dir = out_video_path.parent / "_overlay_frames"
    if tmp_dir.exists():
        for p in tmp_dir.glob("*.png"):
            p.unlink()
    else:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    if camera_mask_path is not None:
        camera_mask = np.asarray(Image.open(camera_mask_path).convert("L"), dtype=np.uint8)
    else:
        camera_mask = None

    for out_idx, image_path in enumerate(image_paths, start=1):
        if camera_mask is not None:
            mask = camera_mask
        else:
            mask_path = mask_dir / f"{image_path.name}.png"
            if not mask_path.exists():
                raise Task1Error(f"Mask missing for preview frame: {mask_path}")
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        panel = _compose_overlay_frame(image_path, mask)
        panel.save(tmp_dir / f"{out_idx:06d}.png")

    run_cmd(
        [
            ffmpeg_bin,
            "-y",
            "-framerate",
            f"{output_fps:g}",
            "-i",
            str(tmp_dir / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_video_path),
        ],
        dry_run=dry_run,
        error_cls=Task1Error,
    )
    for p in tmp_dir.glob("*.png"):
        p.unlink()
    tmp_dir.rmdir()


def _prepare_method_inputs(
    *,
    method: str,
    images_dir: Path,
    case_name: str,
    param_tag: str,
    mask_source: str,
    fps: float,
    output_root: Path,
    dry_run: bool,
) -> tuple[Path | None, Path | None]:
    mask_path: Path | None = None
    camera_mask_path: Path | None = None

    if method == "raw":
        return None, None

    if method == "mask":
        if mask_source == "default":
            camera_mask_path = _resolve_default_mask_path(_default_mask_root(output_root, "default"), case_name, param_tag)
            if dry_run:
                print(f"Would use default ROI camera mask: {camera_mask_path}")
                return None, camera_mask_path
            _check_default_camera_mask(camera_mask_path, missing_hint=_build_missing_mask_hint(case_name, fps, "default"))
            return None, camera_mask_path
        mask_root = _default_mask_root(output_root, mask_source)
        mask_path = _resolve_mask_dir(mask_root, case_name, param_tag)
        if dry_run:
            print(f"Would use {mask_source} masks from: {mask_path}")
            return mask_path, None
        _check_mask_dir(mask_path, images_dir, missing_hint=_build_missing_mask_hint(case_name, fps, mask_source))
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
    mask_source = _normalize_mask_source(cfg.mask_source)

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
            if method == "mask" and mask_source:
                method_root = base_root / f"mask_{mask_source}"
            else:
                method_root = base_root / method
            timing_path = method_root / TIMING_FILENAME
            model_dir = method_root / "sparse" / "0"
            print(f"\n--- Method: {method} ---")

            # Skip if already complete and not in force mode
            if not cfg.force and not cfg.dry_run and _is_method_run_complete(method_root):
                print(f"Reuse existing complete outputs: {method_root}")
                # Load existing analysis into summary_rows
                analysis_csv = method_root / "analysis.csv"
                if cfg.stage in {"all", "analyze"} and analysis_csv.exists():
                    with analysis_csv.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            summary_rows.append(row)
                continue

            if not cfg.dry_run:
                method_root.mkdir(parents=True, exist_ok=True)

            feature_mask_path: Path | None = None
            camera_mask_path: Path | None = None
            if cfg.stage in {"all", "sfm"}:
                with timed_block("prepare_inputs", timings):
                    feature_mask_path, camera_mask_path = _prepare_method_inputs(
                        method=method,
                        images_dir=images_dir,
                        case_name=case_name,
                        param_tag=param_tag,
                        mask_source=mask_source,
                        fps=cfg.fps,
                        output_root=cfg.output_root,
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
                        point_cloud = _parse_points3d_quality(points3d_txt)
                        points_xyz = point_cloud.xyz
                        points_rgb = point_cloud.rgb
                        reliable_mask = _build_reliable_point_mask(point_cloud)
                        quality_stats = _summarize_point_cloud_quality(point_cloud, reliable_mask)
                        median_step, max_step, jump_ratio = _compute_jump_stats(centers)

                        summary_rows.append(
                            _write_analysis(
                                method_root=method_root,
                                case_name=case_name,
                                method=method,
                                mask_source=(mask_source if method == "mask" else None),
                                fps=cfg.fps,
                                total_frames=total_frames,
                                registered_frames=registered_frames,
                                points3d_count=len(points_xyz),
                                points3d_reliable=int(quality_stats["points3d_reliable"]),
                                points3d_reliable_ratio=float(quality_stats["points3d_reliable_ratio"]),
                                reproj_error_median=float(quality_stats["reproj_error_median"]),
                                reproj_error_p90=float(quality_stats["reproj_error_p90"]),
                                reproj_error_p99=float(quality_stats["reproj_error_p99"]),
                                track_length_median=float(quality_stats["track_length_median"]),
                                track_length_p10=float(quality_stats["track_length_p10"]),
                                median_step=median_step,
                                max_step=max_step,
                                jump_ratio=jump_ratio,
                            )
                        )
                        _cleanup_legacy_point_cloud_plots(method_root)
                        _plot_sparse_point_cloud(
                            points_xyz[reliable_mask],
                            points_rgb[reliable_mask],
                            centers,
                            method_root / "sparse_points.png",
                            f"{case_name} {method} Sparse Point Cloud",
                            max_points=cfg.max_points_plot,
                            crop_percentile=1.0,
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
        if cfg.stage in {"all", "analyze"}:
            merge_timings: dict[str, float] = {}
            _write_merged_method_trajectory(
                base_root=base_root,
                case_name=case_name,
                force=cfg.force,
                dry_run=cfg.dry_run,
                timings=merge_timings,
            )
            if merge_timings and not cfg.dry_run:
                merge_timing_path = base_root / "trajectory_overlay_timing.csv"
                write_timing_csv(merge_timing_path, merge_timings)
                print(f"Saved merged trajectory timing: {merge_timing_path}")
                print_timing_summary(f"Timing / {case_name} / {param_tag} / merge", merge_timings)

    print("\nTask3 completed.")
    return 0


def run_task3_masks(cfg: Task3MaskConfig) -> int:
    if cfg.fps <= 0:
        raise Task1Error(f"fps must be positive, got {cfg.fps}")
    source = _normalize_mask_source(cfg.source)
    if cfg.conf < 0 or cfg.conf > 1:
        raise Task1Error(f"conf must be within [0, 1], got {cfg.conf}")
    if cfg.imgsz <= 0:
        raise Task1Error(f"imgsz must be positive, got {cfg.imgsz}")

    videos_dir = cfg.lab1_root / "assets" / "videos"
    selected_videos = S2_VIDEOS if not cfg.videos else [_normalize_video_name(v) for v in cfg.videos]
    invalid = [v for v in selected_videos if v not in S2_VIDEOS]
    if invalid:
        raise Task1Error(f"Unsupported video(s): {invalid}. Supported: {S2_VIDEOS} (or short names S2-1/S2-2)")

    if not cfg.dry_run:
        require_tool(cfg.ffmpeg_bin, error_cls=Task1Error)
        if source == "yolo":
            _ensure_ultralytics_available()

    param_tag = f"fps{_format_float_tag(cfg.fps)}"
    mask_root = _default_mask_root(cfg.output_root, source)

    for video_name in selected_videos:
        video_path = videos_dir / video_name
        if not video_path.exists():
            raise Task1Error(f"Video not found: {video_path}")

        case_name = video_path.stem
        base_root = cfg.output_root / f"{case_name}_{param_tag}"
        images_dir = base_root / "images"
        frame_map_path = base_root / FRAME_MAP_FILENAME
        mask_dir = _resolve_mask_dir(mask_root, case_name, param_tag)
        preview_video_path = mask_dir / "overlay_preview.mp4"
        timings: dict[str, float] = {}

        print(f"\n=== Task3 Mask / {source} / {case_name} / {param_tag} ===")
        if not cfg.dry_run:
            base_root.mkdir(parents=True, exist_ok=True)
            mask_root.mkdir(parents=True, exist_ok=True)

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

        if cfg.dry_run:
            if source == "default":
                print(f"Would generate default ROI camera mask under: {mask_dir}")
            else:
                print(f"Would generate {source} masks under: {mask_dir}")
        else:
            if not _has_any_frames(images_dir):
                raise Task1Error(
                    f"No extracted frames found under {images_dir}. "
                    "Run extraction first or rerun task3-mask without --dry-run."
                )
            if source == "default":
                mask_path = mask_dir / "camera_mask.png"
                if not cfg.force and _is_default_camera_mask_ready(mask_path):
                    print(f"Reuse existing default ROI camera mask: {mask_path}")
                else:
                    if not mask_dir.exists():
                        mask_dir.mkdir(parents=True, exist_ok=True)
                    with timed_block("default_mask", timings):
                        _write_static_roi_camera_mask(images_dir, mask_path, case_name)
                    print(f"Saved default ROI camera mask: {mask_path}")
                with timed_block("overlay_preview", timings):
                    _export_mask_overlay_video(
                        images_dir=images_dir,
                        mask_dir=mask_dir,
                        camera_mask_path=mask_path,
                        ffmpeg_bin=cfg.ffmpeg_bin,
                        out_video_path=preview_video_path,
                        output_fps=cfg.fps,
                        force=cfg.force,
                        dry_run=cfg.dry_run,
                    )
                print(f"Saved mask overlay preview video: {preview_video_path}")
            elif source == "motion":
                if not cfg.force and _is_mask_dir_complete(mask_dir, images_dir):
                    print(f"Reuse existing complete motion masks: {mask_dir}")
                else:
                    if cfg.force and mask_dir.exists():
                        for child in mask_dir.glob("*.png"):
                            child.unlink()
                    with timed_block("motion_mask", timings):
                        _write_motion_masks(images_dir, mask_dir, cfg.motion_threshold, cfg.motion_dilation)
                    print(f"Saved motion masks: {mask_dir}")
                with timed_block("overlay_preview", timings):
                    _export_mask_overlay_video(
                        images_dir=images_dir,
                        mask_dir=mask_dir,
                        camera_mask_path=None,
                        ffmpeg_bin=cfg.ffmpeg_bin,
                        out_video_path=preview_video_path,
                        force=cfg.force,
                        dry_run=cfg.dry_run,
                    )
                print(f"Saved mask overlay preview video: {preview_video_path}")
            else:
                if not cfg.force and _is_mask_dir_complete(mask_dir, images_dir):
                    print(f"Reuse existing complete YOLO masks: {mask_dir}")
                else:
                    if cfg.force and mask_dir.exists():
                        for child in mask_dir.glob("*.png"):
                            child.unlink()
                    with timed_block("yolo_mask", timings):
                        _write_yolo_masks_for_images(
                            images_dir=images_dir,
                            mask_dir=mask_dir,
                            model_name=cfg.model,
                            conf=cfg.conf,
                            imgsz=cfg.imgsz,
                            dilation=cfg.yolo_dilation,
                        )
                    print(f"Saved YOLO masks: {mask_dir}")
                with timed_block("overlay_preview", timings):
                    _export_mask_overlay_video(
                        images_dir=images_dir,
                        mask_dir=mask_dir,
                        camera_mask_path=None,
                        ffmpeg_bin=cfg.ffmpeg_bin,
                        out_video_path=preview_video_path,
                        force=cfg.force,
                        dry_run=cfg.dry_run,
                    )
                print(f"Saved mask overlay preview video: {preview_video_path}")

        if not cfg.dry_run:
            write_timing_csv(mask_dir / TIMING_FILENAME, timings)
            print(f"Saved timing summary: {mask_dir / TIMING_FILENAME}")
        print_timing_summary(f"Timing / Task3 Mask / {source} / {case_name} / {param_tag}", timings)

    print("\nTask3 mask generation completed.")
    return 0
