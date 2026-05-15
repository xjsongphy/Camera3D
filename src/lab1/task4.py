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
ACCEL_JUMP_RATIO = 4.0
EPI_INLIER_THRESHOLD_PX = 0.3
EPI_TRIANG_GATE_PX = 3.0
REPROJ_INLIER_THRESHOLD_PX = 0.2
MIN_PARALLAX_DEG_FOR_TRIANG = 0.2
COMPOSE_MIN_MATCHES = 30
COMPOSE_ROT_THRESHOLD_DEG = 0.05


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


def _geometry_eval_indices(num_images: int, max_triplets: int = 140) -> list[int]:
    if num_images < 3:
        return []
    if num_images - 2 <= max_triplets:
        return list(range(num_images - 2))
    stride = max(1, (num_images - 2) // max_triplets)
    return list(range(0, num_images - 2, stride))


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
    sift = cv2.SIFT_create(nfeatures=2000)
    k1, d1 = sift.detectAndCompute(gray1, None)
    k2, d2 = sift.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn12 = bf.knnMatch(d1, d2, k=2)
    knn21 = bf.knnMatch(d2, d1, k=2)
    forward: dict[int, cv2.DMatch] = {}
    reverse_best: dict[int, int] = {}
    for pair in knn12:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            forward[m.queryIdx] = m
    for pair in knn21:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            reverse_best[m.queryIdx] = m.trainIdx
    matches = [m for qidx, m in forward.items() if reverse_best.get(m.trainIdx) == qidx]
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


def _finite_median(values: np.ndarray, fallback: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    return float(np.median(finite))


def _relative_pose_from_images(im1: dict[str, object], im2: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    r1 = np.asarray(im1["R"], dtype=float)
    t1 = np.asarray(im1["t"], dtype=float).reshape(3)
    r2 = np.asarray(im2["R"], dtype=float)
    t2 = np.asarray(im2["t"], dtype=float).reshape(3)
    r_rel = r2 @ r1.T
    t_rel = t2 - r_rel @ t1
    return r_rel, t_rel


def _compute_accel_jump_ratio(images: list[dict[str, object]], ratio: float = ACCEL_JUMP_RATIO) -> float:
    centers = np.array([np.asarray(im["C"], dtype=float) for im in images], dtype=float)
    frame_idx = np.array([int(im["frame_idx"]) for im in images], dtype=float)
    if len(centers) < 3:
        return 1.0

    dt = np.diff(frame_idx)
    valid_dt = dt > 1e-9
    if np.count_nonzero(valid_dt) < 2:
        return 1.0

    velocity = np.diff(centers, axis=0)[valid_dt] / dt[valid_dt, None]
    dt_valid = dt[valid_dt]
    if len(velocity) < 2:
        return 1.0

    accel_dt = 0.5 * (dt_valid[:-1] + dt_valid[1:])
    valid_accel_dt = accel_dt > 1e-9
    if np.count_nonzero(valid_accel_dt) == 0:
        return 1.0

    accel = (velocity[1:] - velocity[:-1])[valid_accel_dt] / accel_dt[valid_accel_dt, None]
    accel_norm = np.linalg.norm(accel, axis=1)
    finite = accel_norm[np.isfinite(accel_norm)]
    if finite.size == 0:
        return 1.0
    accel_ref = float(np.median(finite))
    if accel_ref <= 1e-12:
        return 0.0
    return float(np.mean(finite > ratio * accel_ref))


def _project_points_px(k: np.ndarray, x_cam: np.ndarray) -> np.ndarray:
    z = np.maximum(1e-9, x_cam[:, 2:3])
    xy = x_cam[:, :2] / z
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    return np.column_stack([fx * xy[:, 0] + cx, fy * xy[:, 1] + cy])


def _robust_pair_reproj_px(
    p1: np.ndarray,
    p2: np.ndarray,
    k: np.ndarray,
    r_gt: np.ndarray,
    t_gt: np.ndarray,
    epi_raw: np.ndarray,
) -> tuple[float, float]:
    if len(p1) < 8:
        return float("nan"), float("nan")
    tri_mask = epi_raw < EPI_TRIANG_GATE_PX
    if np.count_nonzero(tri_mask) < 8:
        keep = min(max(24, len(p1) // 4), len(p1))
        best_idx = np.argsort(epi_raw)[:keep]
    else:
        best_idx = np.flatnonzero(tri_mask)

    p1s = p1[best_idx]
    p2s = p2[best_idx]
    p1n = cv2.undistortPoints(p1s.reshape(-1, 1, 2).astype(np.float64), cameraMatrix=k, distCoeffs=None).reshape(-1, 2)
    p2n = cv2.undistortPoints(p2s.reshape(-1, 1, 2).astype(np.float64), cameraMatrix=k, distCoeffs=None).reshape(-1, 2)
    rays1 = np.column_stack([p1n, np.ones(len(p1n), dtype=float)])
    rays1 /= np.linalg.norm(rays1, axis=1, keepdims=True)
    rays2_local = np.column_stack([p2n, np.ones(len(p2n), dtype=float)])
    rays2_local /= np.linalg.norm(rays2_local, axis=1, keepdims=True)
    rays2 = (r_gt.T @ rays2_local.T).T
    cos_parallax = np.sum(rays1 * rays2, axis=1)
    parallax_deg = np.degrees(np.arccos(np.clip(cos_parallax, -1.0, 1.0)))
    good_parallax = parallax_deg >= MIN_PARALLAX_DEG_FOR_TRIANG
    if np.count_nonzero(good_parallax) >= 8:
        p1n = p1n[good_parallax]
        p2n = p2n[good_parallax]
        p1s = p1s[good_parallax]
        p2s = p2s[good_parallax]
    if len(p1n) < 8:
        return float("nan"), float("nan")

    pmat1 = np.hstack([np.eye(3, dtype=float), np.zeros((3, 1), dtype=float)])
    pmat2 = np.hstack([r_gt, t_gt.reshape(3, 1)])
    x4 = cv2.triangulatePoints(
        pmat1.astype(np.float64),
        pmat2.astype(np.float64),
        p1n.T.astype(np.float64),
        p2n.T.astype(np.float64),
    )
    w = x4[3]
    finite_mask = np.isfinite(w) & (np.abs(w) > 1e-12)
    if np.count_nonzero(finite_mask) < 8:
        return float("nan"), float("nan")
    x = (x4[:3, finite_mask] / w[finite_mask]).T

    z1 = x[:, 2]
    x2 = (r_gt @ x.T + t_gt.reshape(3, 1)).T
    z2 = x2[:, 2]
    cheirality = (z1 > 1e-8) & (z2 > 1e-8)
    if np.count_nonzero(cheirality) >= 8:
        x = x[cheirality]
        x2 = x2[cheirality]
        p1_obs = p1s[finite_mask][cheirality]
        p2_obs = p2s[finite_mask][cheirality]
    else:
        p1_obs = p1s[finite_mask]
        p2_obs = p2s[finite_mask]

    if len(x) < 8:
        return float("nan"), float("nan")
    p1_rep = _project_points_px(k, x)
    p2_rep = _project_points_px(k, x2)
    e1 = np.linalg.norm(p1_rep - p1_obs, axis=1)
    e2 = np.linalg.norm(p2_rep - p2_obs, axis=1)
    errs = 0.5 * (e1 + e2)
    finite = errs[np.isfinite(errs)]
    if finite.size == 0:
        return float("nan"), float("nan")
    finite = finite[finite < 80.0]
    if finite.size < 8:
        return float("nan"), float("nan")
    violation_ratio = float(np.mean(finite > REPROJ_INLIER_THRESHOLD_PX))
    median_err = float(np.median(finite))
    return violation_ratio, median_err


def _estimate_relative_rotation_from_matches(
    p1: np.ndarray,
    p2: np.ndarray,
    k: np.ndarray,
) -> np.ndarray | None:
    if len(p1) < COMPOSE_MIN_MATCHES:
        return None
    e, mask = cv2.findEssentialMat(
        p1.astype(np.float64),
        p2.astype(np.float64),
        cameraMatrix=k.astype(np.float64),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.5,
    )
    if e is None:
        return None
    if e.ndim == 2 and e.shape == (3, 3):
        e_candidates = [e]
    else:
        e_candidates = [e[i : i + 3, :] for i in range(0, e.shape[0], 3)]
    best_r = None
    best_inliers = -1
    for e_i in e_candidates:
        ok, r_est, _t_est, pose_mask = cv2.recoverPose(
            e_i.astype(np.float64),
            p1.astype(np.float64),
            p2.astype(np.float64),
            cameraMatrix=k.astype(np.float64),
            mask=mask,
        )
        if ok <= 0:
            continue
        inliers = int(np.count_nonzero(pose_mask)) if pose_mask is not None else int(ok)
        if inliers > best_inliers:
            best_inliers = inliers
            best_r = r_est
    return best_r


def _ransac_filter_matches(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(p1) < 16:
        return p1, p2
    try:
        f_img, mask = cv2.findFundamentalMat(
            p1.astype(np.float64),
            p2.astype(np.float64),
            cv2.FM_RANSAC,
            1.5,
            0.999,
        )
    except cv2.error:
        return p1, p2
    if f_img is None or mask is None:
        return p1, p2
    inliers = mask.reshape(-1).astype(bool)
    if np.count_nonzero(inliers) < 12:
        return p1, p2
    return p1[inliers], p2[inliers]


def _compute_geometry_metrics(
    images: list[dict[str, object]],
    k_map: dict[int, np.ndarray],
    video_path: Path,
) -> tuple[float, float, float]:
    frame_cache: dict[int, np.ndarray] = {}
    epi_violation_all: list[float] = []
    reproj_violation_all: list[float] = []
    compose_violation_all: list[float] = []
    pair_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
    rot_adj_cache: dict[tuple[int, int], np.ndarray] = {}

    triplet_indices = _geometry_eval_indices(len(images), max_triplets=140)
    pair_indices = sorted({i for start in triplet_indices for i in (start, start + 1) if i + 1 < len(images)})
    for a in pair_indices:
        im1 = images[a]
        im2 = images[a + 1]
        g1 = _read_frame(video_path, int(im1["frame_idx"]), frame_cache)
        g2 = _read_frame(video_path, int(im2["frame_idx"]), frame_cache)
        if g1 is None or g2 is None:
            continue
        p1, p2 = _match_pair(g1, g2)
        if len(p1) < 20:
            continue
        p1, p2 = _ransac_filter_matches(p1, p2)
        if len(p1) < 12:
            pair_cache[(a, a + 1)] = None
            continue
        k = k_map[int(im1["camera_id"])]
        r_gt, t_gt = _relative_pose_from_images(im1, im2)

        fmat_gt = np.linalg.inv(k).T @ (_skew(t_gt) @ r_gt) @ np.linalg.inv(k)
        epi_raw = _symmetric_epipolar_distance(p1, p2, fmat_gt)
        finite_epi_mask = np.isfinite(epi_raw)
        if np.count_nonzero(finite_epi_mask) < 8:
            pair_cache[(a, a + 1)] = None
            continue
        epi_raw = epi_raw[finite_epi_mask]
        p1 = p1[finite_epi_mask]
        p2 = p2[finite_epi_mask]
        epi_violation_all.append(float(np.mean(epi_raw > EPI_INLIER_THRESHOLD_PX)))
        pair_cache[(a, a + 1)] = (p1, p2, k)
        reproj_violation, _reproj_median = _robust_pair_reproj_px(p1, p2, k, r_gt, t_gt, epi_raw)
        if np.isfinite(reproj_violation):
            reproj_violation_all.append(float(reproj_violation))
        r_est_adj = _estimate_relative_rotation_from_matches(p1, p2, k)
        if r_est_adj is not None:
            rot_adj_cache[(a, a + 1)] = r_est_adj

    # Triplet composition consistency against GT two-step relative rotation:
    # R_gt(i,i+2) vs R_est(i+1,i+2) * R_est(i,i+1)
    for i in triplet_indices:
        r12 = rot_adj_cache.get((i, i + 1))
        r23 = rot_adj_cache.get((i + 1, i + 2))
        if r12 is None or r23 is None:
            continue
        r13_gt, _ = _relative_pose_from_images(images[i], images[i + 2])
        r_err = (r23 @ r12) @ r13_gt.T
        compose_violation_all.append(float(_rotation_angle_deg(r_err) > COMPOSE_ROT_THRESHOLD_DEG))

    epi_metric = float(np.median(np.array(epi_violation_all, dtype=float))) if epi_violation_all else 1.0
    reproj_metric = float(np.median(np.array(reproj_violation_all, dtype=float))) if reproj_violation_all else 1.0
    compose_metric = float(np.mean(np.array(compose_violation_all, dtype=float))) if compose_violation_all else 1.0
    return epi_metric, reproj_metric, compose_metric


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float)


def _compute_quality_score(metrics: dict[str, float]) -> tuple[float, float]:
    smooth_term = math.log1p(4.0 * metrics["smooth_jump_ratio"])
    epi_term = math.log1p(4.0 * metrics["epi_dist_px"])
    reproj_term = math.log1p(4.0 * metrics["reproj_err_px"])
    compose_term = math.log1p(4.0 * metrics["compose_rot_err_deg"])
    penalty = 0.30 * smooth_term + 0.25 * epi_term + 0.25 * reproj_term + 0.20 * compose_term
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
            images = _parse_images(images_txt)
            points3d_count = sum(1 for _ in points3d_txt.open("r", encoding="utf-8") if _.strip() and not _.startswith("#"))
            epi_dist_px, reproj_err_px, compose_rot_err_deg = _compute_geometry_metrics(images, k_map, video_path)
            metrics = {
                "case": case_name,
                "label": _case_label(case_name),
                "num_poses": len(images),
                "points3d": points3d_count,
                "smooth_jump_ratio": _compute_accel_jump_ratio(images),
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
