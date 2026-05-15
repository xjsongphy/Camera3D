from __future__ import annotations

import argparse
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from lab1.task4 import (
    ANNOTATION_CASES,
    _case_label,
    _compute_pairwise_auc,
    _parse_camera_intrinsics,
    _parse_images,
    _relative_pose_from_images,
    _rotation_angle_deg,
    _skew,
    _symmetric_epipolar_distance,
)


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_ROOT = ROOT / "docs" / "lab1" / "assets" / "annotations"
OUT_ROOT = ROOT / "outputs" / "lab1" / "task4_geometry_fps16"
PLOT_ROOT = ROOT / "docs" / "lab1" / "report_assets" / "task4_geometry_fps16"


@dataclass(frozen=True)
class GeometryConfig:
    target_fps: float
    max_pairs: int
    workers: int
    nfeatures: int
    ratio: float
    ransac_px: float
    image_ransac: bool
    pair_gap: int
    triangulate_epi_percentile: float
    epi_threshold_px: float
    reproj_threshold_px: float
    min_parallax_deg: float


def _video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if not np.isfinite(fps) or fps <= 1e-6:
        return 30.0
    return fps


def _sample_pose_indices(images: list[dict[str, object]], video_fps: float, target_fps: float, max_pairs: int) -> list[int]:
    if len(images) < 2:
        return []
    frame_step = max(1, int(round(video_fps / target_fps)))
    selected = [idx for idx, im in enumerate(images) if int(im["frame_idx"]) % frame_step == 0]
    if len(selected) < 2:
        selected = list(range(len(images)))
    pair_count = len(selected) - 1
    if max_pairs > 0 and pair_count > max_pairs:
        keep = np.linspace(0, len(selected) - 1, max_pairs + 1).round().astype(int)
        selected = [selected[int(i)] for i in keep]
    return selected


def _sample_pairs(selected: list[int], pair_gap: int, max_pairs: int) -> list[tuple[int, int]]:
    gap = max(1, pair_gap)
    pairs = [(selected[i], selected[i + gap]) for i in range(0, max(0, len(selected) - gap))]
    if max_pairs > 0 and len(pairs) > max_pairs:
        keep = np.linspace(0, len(pairs) - 1, max_pairs).round().astype(int)
        pairs = [pairs[int(i)] for i in keep]
    return pairs


def _read_gray(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _sift_features(
    sift: cv2.SIFT,
    frame_idx: int,
    gray: np.ndarray,
    cache: dict[int, tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]],
) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
    if frame_idx not in cache:
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        cache[frame_idx] = (tuple(keypoints), descriptors)
    return cache[frame_idx]


def _match_features(
    k1: tuple[cv2.KeyPoint, ...],
    d1: np.ndarray | None,
    k2: tuple[cv2.KeyPoint, ...],
    d2: np.ndarray | None,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if d1 is None or d2 is None or len(k1) < 16 or len(k2) < 16:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn12 = matcher.knnMatch(d1, d2, k=2)
    knn21 = matcher.knnMatch(d2, d1, k=2)
    forward: dict[int, cv2.DMatch] = {}
    reverse_best: dict[int, int] = {}
    for pair in knn12:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            forward[pair[0].queryIdx] = pair[0]
    for pair in knn21:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            reverse_best[pair[0].queryIdx] = pair[0].trainIdx

    matches = [m for qidx, m in forward.items() if reverse_best.get(m.trainIdx) == qidx]
    if len(matches) < 16:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)
    matches = sorted(matches, key=lambda m: m.distance)[:500]
    p1 = np.array([k1[m.queryIdx].pt for m in matches], dtype=float)
    p2 = np.array([k2[m.trainIdx].pt for m in matches], dtype=float)
    return p1, p2


def _ransac_filter(p1: np.ndarray, p2: np.ndarray, ransac_px: float) -> tuple[np.ndarray, np.ndarray]:
    if len(p1) < 16:
        return p1, p2
    fmat, mask = cv2.findFundamentalMat(
        p1.astype(np.float64),
        p2.astype(np.float64),
        cv2.FM_RANSAC,
        ransac_px,
        0.999,
    )
    if fmat is None or mask is None:
        return p1, p2
    keep = mask.reshape(-1).astype(bool)
    if np.count_nonzero(keep) < 12:
        return p1, p2
    return p1[keep], p2[keep]


def _project(k: np.ndarray, x_cam: np.ndarray) -> np.ndarray:
    z = np.maximum(1e-9, x_cam[:, 2:3])
    xy = x_cam[:, :2] / z
    return np.column_stack([k[0, 0] * xy[:, 0] + k[0, 2], k[1, 1] * xy[:, 1] + k[1, 2]])


def _pair_metrics(
    p1: np.ndarray,
    p2: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    r_rel: np.ndarray,
    t_rel: np.ndarray,
    cfg: GeometryConfig,
) -> tuple[float, float, float, int]:
    if len(p1) < 8:
        return math.nan, math.nan, math.nan, 0

    f_gt = np.linalg.inv(k2).T @ (_skew(t_rel) @ r_rel) @ np.linalg.inv(k1)
    epi = _symmetric_epipolar_distance(p1, p2, f_gt)
    finite = np.isfinite(epi)
    if np.count_nonzero(finite) < 8:
        return math.nan, math.nan, math.nan, 0
    p1 = p1[finite]
    p2 = p2[finite]
    epi = epi[finite]
    epi_violation = float(np.mean(epi > cfg.epi_threshold_px))

    if cfg.triangulate_epi_percentile < 100.0:
        cutoff = np.percentile(epi, cfg.triangulate_epi_percentile)
        gate = epi <= cutoff
        if np.count_nonzero(gate) < 8:
            gate = np.argsort(epi)[: min(max(24, len(epi) // 3), len(epi))]
        p1s = p1[gate]
        p2s = p2[gate]
    else:
        p1s = p1
        p2s = p2

    p1n = cv2.undistortPoints(p1s.reshape(-1, 1, 2).astype(np.float64), k1, None).reshape(-1, 2)
    p2n = cv2.undistortPoints(p2s.reshape(-1, 1, 2).astype(np.float64), k2, None).reshape(-1, 2)

    rays1 = np.column_stack([p1n, np.ones(len(p1n))])
    rays1 /= np.linalg.norm(rays1, axis=1, keepdims=True)
    rays2_local = np.column_stack([p2n, np.ones(len(p2n))])
    rays2_local /= np.linalg.norm(rays2_local, axis=1, keepdims=True)
    rays2 = (r_rel.T @ rays2_local.T).T
    parallax = np.degrees(np.arccos(np.clip(np.sum(rays1 * rays2, axis=1), -1.0, 1.0)))
    keep = parallax >= cfg.min_parallax_deg
    if np.count_nonzero(keep) >= 8:
        p1n, p2n, p1s, p2s = p1n[keep], p2n[keep], p1s[keep], p2s[keep]
    if len(p1n) < 8:
        return epi_violation, math.nan, math.nan, 0

    pmat1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    pmat2 = np.hstack([r_rel, t_rel.reshape(3, 1)])
    x4 = cv2.triangulatePoints(
        pmat1.astype(np.float64),
        pmat2.astype(np.float64),
        p1n.T.astype(np.float64),
        p2n.T.astype(np.float64),
    )
    w = x4[3]
    valid = np.isfinite(w) & (np.abs(w) > 1e-12)
    if np.count_nonzero(valid) < 8:
        return epi_violation, math.nan, math.nan, 0
    x = (x4[:3, valid] / w[valid]).T
    p1_obs = p1s[valid]
    p2_obs = p2s[valid]
    x2 = (r_rel @ x.T + t_rel.reshape(3, 1)).T
    front = (x[:, 2] > 1e-8) & (x2[:, 2] > 1e-8)
    if np.count_nonzero(front) >= 8:
        x, x2, p1_obs, p2_obs = x[front], x2[front], p1_obs[front], p2_obs[front]

    if len(x) < 8:
        return epi_violation, math.nan, math.nan, 0
    e1 = np.linalg.norm(_project(k1, x) - p1_obs, axis=1)
    e2 = np.linalg.norm(_project(k2, x2) - p2_obs, axis=1)
    reproj = 0.5 * (e1 + e2)
    reproj = reproj[np.isfinite(reproj) & (reproj < 100.0)]
    if reproj.size < 8:
        return epi_violation, math.nan, math.nan, int(len(x))
    return epi_violation, float(np.mean(reproj > cfg.reproj_threshold_px)), float(np.median(reproj)), int(reproj.size)


def _case_worker(case_name: str, cfg: GeometryConfig) -> dict[str, object]:
    case_root = ANNOTATIONS_ROOT / case_name
    images = _parse_images(case_root / "sparse" / "0" / "images.txt")
    k_map = _parse_camera_intrinsics(case_root / "sparse" / "0" / "cameras.txt")
    video_path = case_root / "video.mp4"
    video_fps = _video_fps(video_path)
    selected = _sample_pose_indices(images, video_fps, cfg.target_fps, 0)
    pairs = _sample_pairs(selected, cfg.pair_gap, cfg.max_pairs)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    sift = cv2.SIFT_create(nfeatures=cfg.nfeatures)
    feature_cache: dict[int, tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]] = {}
    gray_cache: dict[int, np.ndarray] = {}

    epi_vals: list[float] = []
    reproj_vals: list[float] = []
    reproj_medians: list[float] = []
    tri_points = 0
    match_counts: list[int] = []
    rot_compose_errs: list[float] = []

    try:
        for left, right in pairs:
            im1, im2 = images[left], images[right]
            f1, f2 = int(im1["frame_idx"]), int(im2["frame_idx"])
            if f1 not in gray_cache:
                gray = _read_gray(cap, f1)
                if gray is not None:
                    gray_cache[f1] = gray
            if f2 not in gray_cache:
                gray = _read_gray(cap, f2)
                if gray is not None:
                    gray_cache[f2] = gray
            if f1 not in gray_cache or f2 not in gray_cache:
                continue

            kp1, d1 = _sift_features(sift, f1, gray_cache[f1], feature_cache)
            kp2, d2 = _sift_features(sift, f2, gray_cache[f2], feature_cache)
            p1, p2 = _match_features(kp1, d1, kp2, d2, cfg.ratio)
            if cfg.image_ransac:
                p1, p2 = _ransac_filter(p1, p2, cfg.ransac_px)
            match_counts.append(len(p1))
            if len(p1) < 8:
                continue

            k1 = k_map[int(im1["camera_id"])]
            k2 = k_map[int(im2["camera_id"])]
            r_rel, t_rel = _relative_pose_from_images(im1, im2)
            epi, reproj, reproj_median, n_tri = _pair_metrics(p1, p2, k1, k2, r_rel, t_rel, cfg)
            if np.isfinite(epi):
                epi_vals.append(epi)
            if np.isfinite(reproj):
                reproj_vals.append(reproj)
            if np.isfinite(reproj_median):
                reproj_medians.append(reproj_median)
            tri_points += n_tri

        for a, b, c in zip(selected[:-2], selected[1:-1], selected[2:]):
            r_ab, _t_ab = _relative_pose_from_images(images[a], images[b])
            r_bc, _t_bc = _relative_pose_from_images(images[b], images[c])
            r_ac, _t_ac = _relative_pose_from_images(images[a], images[c])
            rot_compose_errs.append(_rotation_angle_deg((r_bc @ r_ab) @ r_ac.T))
    finally:
        cap.release()

    return {
        "case": case_name,
        "label": _case_label(case_name),
        "video_fps": video_fps,
        "sampled_frames": len(selected),
        "pairs": len(pairs),
        "matched_pairs": len(epi_vals),
        "median_matches": float(np.median(match_counts)) if match_counts else 0.0,
        "triangulated_points": tri_points,
        "epi_dist_px": float(np.median(epi_vals)) if epi_vals else 1.0,
        "reproj_err_px": float(np.median(reproj_vals)) if reproj_vals else 1.0,
        "reproj_median_px": float(np.median(reproj_medians)) if reproj_medians else math.nan,
        "compose_rot_err_deg": float(np.mean(np.array(rot_compose_errs) > 1.0)) if rot_compose_errs else 1.0,
    }


def _metric_eval(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = [float(r[key]) for r in rows]
    labels = [1 if r["label"] == "bad" else 0 for r in rows]
    best_acc = -1.0
    best_threshold = 0.0
    best_preds: list[int] = []
    for threshold in sorted(set(values)):
        preds = [1 if value >= threshold else 0 for value in values]
        acc = sum(pred == label for pred, label in zip(preds, labels)) / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            best_preds = preds
    return {
        "best_threshold": best_threshold,
        "accuracy": best_acc,
        "auc": _compute_pairwise_auc(values, labels),
        "tp": sum(p == 1 and y == 1 for p, y in zip(best_preds, labels)),
        "tn": sum(p == 0 and y == 0 for p, y in zip(best_preds, labels)),
        "fp": sum(p == 1 and y == 0 for p, y in zip(best_preds, labels)),
        "fn": sum(p == 0 and y == 1 for p, y in zip(best_preds, labels)),
    }


def _write_outputs(rows: list[dict[str, object]], out_root: Path, plot_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "case_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = ["epi_dist_px", "reproj_err_px", "reproj_median_px", "compose_rot_err_deg"]
    stats = {key: _metric_eval(rows, key) for key in metric_keys}
    lines = ["task4_video_geometry_fps16", f"cases={len(rows)}"]
    for key in metric_keys:
        metric_stats = stats[key]
        lines.extend(
            [
                f"{key}.best_threshold={metric_stats['best_threshold']:.6f}",
                f"{key}.accuracy={metric_stats['accuracy']:.6f}",
                f"{key}.auc={metric_stats['auc']:.6f}",
                f"{key}.tp={metric_stats['tp']}",
                f"{key}.tn={metric_stats['tn']}",
                f"{key}.fp={metric_stats['fp']}",
                f"{key}.fn={metric_stats['fn']}",
            ]
        )
    (out_root / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    cases = [str(r["case"]) for r in rows]
    colors = ["tab:red" if r["label"] == "bad" else "tab:green" for r in rows]
    x = np.arange(len(rows))
    metrics = [
        ("epi_dist_px", [float(r["epi_dist_px"]) for r in rows]),
        ("reproj_err_px", [float(r["reproj_err_px"]) for r in rows]),
        ("reproj_median_px", [float(r["reproj_median_px"]) for r in rows]),
        ("triangulated_points", [float(r["triangulated_points"]) for r in rows]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True)
    for ax, (name, values) in zip(axes.flatten(), metrics):
        ax.bar(x, values, color=colors)
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels(cases)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_root / "task4_video_geometry_fps16_metrics.png", dpi=180)
    plt.close(fig)


def _print_progress(done: int, total: int, *, width: int = 28) -> None:
    total = max(total, 1)
    filled = int(round(width * done / total))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\rProgress [{bar}] {done}/{total}")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Task4 video-based geometry metrics at fps16.")
    parser.add_argument("--target-fps", type=float, default=16.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=160, help="uniform pair cap per case; use 0 for all fps16 pairs")
    parser.add_argument("--nfeatures", type=int, default=1000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-px", type=float, default=1.5)
    parser.add_argument("--image-ransac", action="store_true", help="filter matches with image-estimated F before pose evaluation")
    parser.add_argument("--pair-gap", type=int, default=1, help="gap between sampled fps16 frames used as a pair")
    parser.add_argument(
        "--triangulate-epi-percentile",
        type=float,
        default=100.0,
        help="use only this best epipolar-distance percentile for triangulation; 100 means all matches",
    )
    parser.add_argument("--epi-threshold-px", type=float, default=1.5)
    parser.add_argument("--reproj-threshold-px", type=float, default=2.0)
    parser.add_argument("--min-parallax-deg", type=float, default=0.2)
    parser.add_argument("--cases", nargs="+", default=list(ANNOTATION_CASES))
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--plot-root", type=Path, default=PLOT_ROOT)
    args = parser.parse_args()

    cfg = GeometryConfig(
        target_fps=args.target_fps,
        max_pairs=args.max_pairs,
        workers=args.workers,
        nfeatures=args.nfeatures,
        ratio=args.ratio,
        ransac_px=args.ransac_px,
        image_ransac=args.image_ransac,
        pair_gap=args.pair_gap,
        triangulate_epi_percentile=args.triangulate_epi_percentile,
        epi_threshold_px=args.epi_threshold_px,
        reproj_threshold_px=args.reproj_threshold_px,
        min_parallax_deg=args.min_parallax_deg,
    )
    cases = [f"{int(c):02d}" if str(c).isdigit() else str(c) for c in args.cases]

    rows: list[dict[str, object]] = []
    completed = 0
    _print_progress(completed, len(cases))
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(_case_worker, case, cfg): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows.append(row)
            completed += 1
            _print_progress(completed, len(cases))
            print(
                f"  {case}: reproj={row['reproj_err_px']:.4f}, "
                f"epi={row['epi_dist_px']:.4f}, tri={row['triangulated_points']}"
            )

    rows.sort(key=lambda r: str(r["case"]))
    _write_outputs(rows, args.out_root, args.plot_root)
    print(f"Saved metrics: {args.out_root / 'case_metrics.csv'}")
    print(f"Saved summary: {args.out_root / 'summary.txt'}")
    print(f"Saved plot: {args.plot_root / 'task4_video_geometry_fps16_metrics.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
