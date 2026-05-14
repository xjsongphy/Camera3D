from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from lab1.geometry_utils import quat_to_rot
from lab1.logging_utils import print_timing_summary, timed_block, write_timing_csv
from lab1.task1 import TIMING_FILENAME, Task1Error


ANNOTATION_CASES = tuple(f"{idx:02d}" for idx in range(1, 11))


@dataclass
class Task4Config:
    lab1_root: Path
    output_root: Path
    force: bool
    dry_run: bool
    cases: list[str] | None = None


def _normalize_case_name(name: str) -> str:
    value = name.strip()
    if value.isdigit():
        value = f"{int(value):02d}"
    if value not in ANNOTATION_CASES:
        raise Task1Error(f"Unsupported task4 case: {name}. Supported: {ANNOTATION_CASES}")
    return value


def _parse_camera_intrinsics(cameras_txt: Path) -> dict[int, np.ndarray]:
    k_map: dict[int, np.ndarray] = {}
    with cameras_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            params = list(map(float, parts[4:]))
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
                f0, cx, cy = params[:3]
                fx, fy = f0, f0
            else:
                raise Task1Error(f"Unsupported camera model for task4: {model}")
            k_map[cam_id] = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    if not k_map:
        raise Task1Error(f"No camera intrinsics parsed from {cameras_txt}")
    return k_map


def _parse_images(images_txt: Path) -> list[dict[str, object]]:
    images: list[dict[str, object]] = []
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
                raise Task1Error(f"Malformed images pose line: {pose_line}")
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]
            r = quat_to_rot(qw, qx, qy, qz, error_cls=Task1Error)
            t = np.array([tx, ty, tz], dtype=float)
            c = -r.T @ t
            m = re.findall(r"(\d+)", name)
            if not m:
                raise Task1Error(f"Cannot parse frame index from image name: {name}")
            frame_idx = int(m[-1])
            images.append(
                {
                    "image_id": image_id,
                    "name": name,
                    "frame_idx": frame_idx,
                    "camera_id": camera_id,
                    "R": r,
                    "t": t,
                    "C": c,
                }
            )
            _ = f.readline()
    if not images:
        raise Task1Error(f"No images parsed from {images_txt}")
    images.sort(key=lambda d: str(d["name"]))
    return images


def _sample_images(images: list[dict[str, object]], max_samples: int = 140) -> list[dict[str, object]]:
    if len(images) <= max_samples:
        return images
    idx = np.linspace(0, len(images) - 1, max_samples).astype(int)
    return [images[int(i)] for i in idx]


def _read_frame(video_path: Path, frame_idx: int, cache: dict[int, np.ndarray]) -> np.ndarray | None:
    if frame_idx in cache:
        return cache[frame_idx]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cache[frame_idx] = gray
    return gray


def _match_pair(gray1: np.ndarray, gray2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    orb = cv2.ORB_create(nfeatures=2000)
    k1, d1 = orb.detectAndCompute(gray1, None)
    k2, d2 = orb.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 20:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)
    matches = sorted(matches, key=lambda m: m.distance)[:400]
    p1 = np.array([k1[m.queryIdx].pt for m in matches], dtype=float)
    p2 = np.array([k2[m.trainIdx].pt for m in matches], dtype=float)
    return p1, p2


def _symmetric_epipolar_distance(p1: np.ndarray, p2: np.ndarray, fmat: np.ndarray) -> np.ndarray:
    p1h = np.column_stack([p1, np.ones(len(p1), dtype=float)])
    p2h = np.column_stack([p2, np.ones(len(p2), dtype=float)])
    l2 = (fmat @ p1h.T).T
    l1 = (fmat.T @ p2h.T).T
    d2 = np.abs(np.sum(p2h * l2, axis=1)) / np.sqrt(np.maximum(1e-12, l2[:, 0] ** 2 + l2[:, 1] ** 2))
    d1 = np.abs(np.sum(p1h * l1, axis=1)) / np.sqrt(np.maximum(1e-12, l1[:, 0] ** 2 + l1[:, 1] ** 2))
    return 0.5 * (d1 + d2)


def _rotation_angle_deg(r: np.ndarray) -> float:
    val = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def _estimate_pair_relative_pose(p1: np.ndarray, p2: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if len(p1) < 20:
        return None
    e, inlier = cv2.findEssentialMat(p1, p2, cameraMatrix=k, method=cv2.RANSAC, prob=0.999, threshold=1.5)
    if e is None or inlier is None:
        return None
    inlier_mask = inlier.ravel().astype(bool)
    if np.count_nonzero(inlier_mask) < 16:
        return None
    p1i = p1[inlier_mask]
    p2i = p2[inlier_mask]
    ok, r, t, mask_pose = cv2.recoverPose(e, p1i, p2i, cameraMatrix=k)
    if ok < 12 or mask_pose is None:
        return None
    pose_mask = mask_pose.ravel().astype(bool)
    p1f = p1i[pose_mask]
    p2f = p2i[pose_mask]
    if len(p1f) < 12:
        return None
    return r, t.reshape(3), p1f, p2f


def _compute_smoothness_metric(images: list[dict[str, object]]) -> float:
    centers = np.array([np.asarray(im["C"], dtype=float) for im in images], dtype=float)
    if len(centers) < 3:
        return 1.0
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    step_med = float(np.median(step))
    if step_med <= 1e-12:
        return 0.0
    return float(np.mean(step > 3.0 * step_med))


def _compute_geometry_metrics(
    images: list[dict[str, object]],
    k_map: dict[int, np.ndarray],
    video_path: Path,
) -> tuple[float, float, float]:
    frame_cache: dict[int, np.ndarray] = {}
    pair_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray] | None] = {}
    epi_all: list[float] = []
    reproj_all: list[float] = []
    comp_all: list[float] = []

    def get_pair(a: int, b: int) -> tuple[np.ndarray, np.ndarray] | None:
        key = (a, b)
        if key in pair_cache:
            return pair_cache[key]
        im1 = images[a]
        im2 = images[b]
        g1 = _read_frame(video_path, int(im1["frame_idx"]), frame_cache)
        g2 = _read_frame(video_path, int(im2["frame_idx"]), frame_cache)
        if g1 is None or g2 is None:
            pair_cache[key] = None
            return None
        p1, p2 = _match_pair(g1, g2)
        if len(p1) < 20:
            pair_cache[key] = None
            return None
        k = k_map[int(im1["camera_id"])]
        rel = _estimate_pair_relative_pose(p1, p2, k)
        if rel is None:
            pair_cache[key] = None
            return None
        r, t, p1f, p2f = rel

        fmat = np.linalg.inv(k).T @ (_skew(t) @ r) @ np.linalg.inv(k)
        epi_all.extend(_symmetric_epipolar_distance(p1f, p2f, fmat).tolist())

        pmat1 = k @ np.hstack([np.eye(3), np.zeros((3, 1), dtype=float)])
        pmat2 = k @ np.hstack([r, t.reshape(3, 1)])
        x4 = cv2.triangulatePoints(pmat1, pmat2, p1f.T.astype(np.float64), p2f.T.astype(np.float64))
        x3 = (x4[:3] / np.maximum(1e-12, x4[3:4])).T
        x3_h = np.column_stack([x3, np.ones(len(x3), dtype=float)])
        z1 = (pmat1 @ x3_h.T).T[:, 2]
        z2 = (pmat2 @ x3_h.T).T[:, 2]
        valid_depth = (z1 > 1e-6) & (z2 > 1e-6) & np.isfinite(z1) & np.isfinite(z2)
        if np.count_nonzero(valid_depth) < 8:
            pair_cache[key] = (r, t)
            return pair_cache[key]
        x3 = x3[valid_depth]
        p1f = p1f[valid_depth]
        p2f = p2f[valid_depth]
        pr1 = (pmat1 @ np.column_stack([x3, np.ones(len(x3), dtype=float)]).T).T
        pr2 = (pmat2 @ np.column_stack([x3, np.ones(len(x3), dtype=float)]).T).T
        uv1 = pr1[:, :2] / np.maximum(1e-12, pr1[:, 2:3])
        uv2 = pr2[:, :2] / np.maximum(1e-12, pr2[:, 2:3])
        e1 = np.linalg.norm(uv1 - p1f, axis=1)
        e2 = np.linalg.norm(uv2 - p2f, axis=1)
        e = 0.5 * (e1 + e2)
        e = e[np.isfinite(e) & (e < 1000.0)]
        if e.size:
            reproj_all.extend(e.tolist())

        pair_cache[key] = (r, t)
        return pair_cache[key]

    for i in range(len(images) - 1):
        _ = get_pair(i, i + 1)

    for i in range(len(images) - 2):
        ij = get_pair(i, i + 1)
        jk = get_pair(i + 1, i + 2)
        ik = get_pair(i, i + 2)
        if ij is None or jk is None or ik is None:
            continue
        rij, _tij = ij
        rjk, _tjk = jk
        rik, _tik = ik
        comp_all.append(_rotation_angle_deg(rik.T @ (rjk @ rij)))

    epi_metric = float(np.median(np.array(epi_all, dtype=float))) if epi_all else 1e6
    reproj_metric = float(np.median(np.array(reproj_all, dtype=float))) if reproj_all else 1e6
    comp_metric = float(np.median(np.array(comp_all, dtype=float))) if comp_all else 180.0
    return epi_metric, reproj_metric, comp_metric


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float)


def _compute_quality_score(metrics: dict[str, float]) -> tuple[float, float]:
    smooth_term = metrics["smooth_jump_ratio"] * 6.0
    epi_term = math.log1p(metrics["epi_dist_px"] / 1.5)
    reproj_term = math.log1p(metrics["reproj_err_px"] / 1.5)
    comp_term = math.log1p(metrics["compose_rot_err_deg"] / 2.0)
    penalty = 0.25 * smooth_term + 0.25 * epi_term + 0.25 * reproj_term + 0.25 * comp_term
    score = 100.0 * math.exp(-penalty)
    return score, penalty


def _case_label(case_name: str) -> str:
    return "bad" if int(case_name) <= 5 else "good"


def _evaluate_threshold(rows: list[dict[str, object]]) -> dict[str, float]:
    scores = [float(row["quality_score"]) for row in rows]
    labels = [1 if row["label"] == "good" else 0 for row in rows]

    candidate_thresholds = sorted(set(scores))
    best_acc = -1.0
    best_threshold = candidate_thresholds[0] if candidate_thresholds else 0.0
    best_preds: list[int] = []
    for threshold in candidate_thresholds:
        preds = [1 if score >= threshold else 0 for score in scores]
        acc = float(np.mean([pred == label for pred, label in zip(preds, labels)]))
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            best_preds = preds

    tp = sum(pred == 1 and label == 1 for pred, label in zip(best_preds, labels))
    tn = sum(pred == 0 and label == 0 for pred, label in zip(best_preds, labels))
    fp = sum(pred == 1 and label == 0 for pred, label in zip(best_preds, labels))
    fn = sum(pred == 0 and label == 1 for pred, label in zip(best_preds, labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    auc = _compute_pairwise_auc(scores, labels)
    return {
        "best_threshold": float(best_threshold),
        "accuracy": best_acc,
        "precision_good": precision,
        "recall_good": recall,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "auc": auc,
    }


def _compute_pairwise_auc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def _plot_quality_scores(rows: list[dict[str, object]], out_path: Path) -> None:
    case_names = [str(row["case"]) for row in rows]
    scores = [float(row["quality_score"]) for row in rows]
    colors = ["tab:red" if row["label"] == "bad" else "tab:green" for row in rows]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(case_names, scores, color=colors)
    ax.set_xlabel("Case")
    ax.set_ylabel("Quality Score")
    ax.set_title("Task4 Pose Quality Scores")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _write_summary(
    out_path: Path,
    rows: list[dict[str, object]],
    eval_stats: dict[str, float],
) -> None:
    lines = [
        "task4_pose_quality_summary",
        f"cases={len(rows)}",
        f"best_threshold={eval_stats['best_threshold']:.6f}",
        f"accuracy={eval_stats['accuracy']:.6f}",
        f"precision_good={eval_stats['precision_good']:.6f}",
        f"recall_good={eval_stats['recall_good']:.6f}",
        f"auc={eval_stats['auc']:.6f}",
        f"tp={int(eval_stats['tp'])}",
        f"tn={int(eval_stats['tn'])}",
        f"fp={int(eval_stats['fp'])}",
        f"fn={int(eval_stats['fn'])}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_task4(cfg: Task4Config) -> int:
    annotations_root = cfg.lab1_root / "assets" / "annotations"
    selected_cases = ANNOTATION_CASES if not cfg.cases else tuple(_normalize_case_name(c) for c in cfg.cases)
    out_root = cfg.output_root

    if not cfg.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    timings: dict[str, float] = {}

    for case_name in selected_cases:
        case_root = annotations_root / case_name
        cameras_txt = case_root / "sparse" / "0" / "cameras.txt"
        images_txt = case_root / "sparse" / "0" / "images.txt"
        points3d_txt = case_root / "sparse" / "0" / "points3D.txt"
        video_path = case_root / "video.mp4"
        if not cameras_txt.exists() or not images_txt.exists() or not points3d_txt.exists() or not video_path.exists():
            raise Task1Error(f"Missing annotation files under {case_root}")

        print(f"\n=== Task4 / {case_name} ===")
        with timed_block(case_name, timings):
            k_map = _parse_camera_intrinsics(cameras_txt)
            images = _sample_images(_parse_images(images_txt))
            points3d_count = sum(1 for _ in points3d_txt.open("r", encoding="utf-8") if _.strip() and not _.startswith("#"))
            epi_dist_px, reproj_err_px, compose_rot_err_deg = _compute_geometry_metrics(images, k_map, video_path)
            metrics = {
                "case": case_name,
                "label": _case_label(case_name),
                "num_poses": len(images),
                "points3d": points3d_count,
                "smooth_jump_ratio": _compute_smoothness_metric(images),
                "epi_dist_px": epi_dist_px,
                "reproj_err_px": reproj_err_px,
                "compose_rot_err_deg": compose_rot_err_deg,
            }
            quality_score, penalty = _compute_quality_score(metrics)
            metrics["quality_score"] = quality_score
            metrics["penalty"] = penalty
            rows.append(metrics)

    rows.sort(key=lambda row: str(row["case"]))

    if cfg.dry_run:
        print("Dry run: skip writing task4 outputs.")
        print_timing_summary("Timing / task4", timings)
        return 0

    csv_path = out_root / "case_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    eval_stats = _evaluate_threshold(rows)
    summary_path = out_root / "summary.txt"
    _write_summary(summary_path, rows, eval_stats)

    plot_path = out_root / "quality_scores.png"
    _plot_quality_scores(rows, plot_path)

    write_timing_csv(out_root / TIMING_FILENAME, timings)
    print(f"Saved metrics: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {plot_path}")
    print_timing_summary("Timing / task4", timings)
    return 0
