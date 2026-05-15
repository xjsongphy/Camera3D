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
from lab1.task1 import TIMING_FILENAME, Task1Error, _parse_image_poses, _plot_trajectory


ANNOTATION_CASES = tuple(f"{idx:02d}" for idx in range(1, 11))
ACCEL_JUMP_RATIO = 4.0
ROT_ACCEL_JUMP_RATIO = 4.0
EPI_INLIER_THRESHOLD_PX = 1.5
EPI_TRIANG_GATE_PX = 3.0
REPROJ_INLIER_THRESHOLD_PX = 2.0
MIN_PARALLAX_DEG_FOR_TRIANG = 0.2
COMPOSE_MIN_MATCHES = 30
COMPOSE_ROT_THRESHOLD_DEG = 2.0
ZIGZAG_WINDOW = 7
ZIGZAG_RESIDUAL_THRESHOLD = 2.0
POSE_COMPOSE_THRESHOLD_DEG = 1.0


@dataclass
class Task4Config:
    lab1_root: Path
    output_root: Path
    force: bool
    dry_run: bool
    cases: list[str] | None = None
    mode: str = "run"
    direction_arrows: int = 12
    heavy_geometry: bool = False
    epi_threshold_px: float = EPI_INLIER_THRESHOLD_PX
    reproj_threshold_px: float = REPROJ_INLIER_THRESHOLD_PX
    compose_threshold_deg: float = POSE_COMPOSE_THRESHOLD_DEG
    geometry_max_triplets: int = 80
    zigzag_residual_threshold: float = ZIGZAG_RESIDUAL_THRESHOLD
    accel_jump_ratio: float = ACCEL_JUMP_RATIO
    rot_accel_jump_ratio: float = ROT_ACCEL_JUMP_RATIO


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
    stride = max(1, math.ceil((num_images - 2) / max_triplets))
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


def _detect_sift(
    frame_idx: int,
    gray: np.ndarray,
    sift: cv2.SIFT,
    feature_cache: dict[int, tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]],
) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
    if frame_idx not in feature_cache:
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        feature_cache[frame_idx] = (tuple(keypoints), descriptors)
    return feature_cache[frame_idx]


def _match_pair(
    frame_idx1: int,
    gray1: np.ndarray,
    frame_idx2: int,
    gray2: np.ndarray,
    sift: cv2.SIFT,
    feature_cache: dict[int, tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]],
) -> tuple[np.ndarray, np.ndarray]:
    k1, d1 = _detect_sift(frame_idx1, gray1, sift, feature_cache)
    k2, d2 = _detect_sift(frame_idx2, gray2, sift, feature_cache)
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


def _compute_trajectory_smoothness(images: list[dict[str, object]], window: int = 5) -> float:
    centers = np.array([np.asarray(im["C"], dtype=float) for im in images], dtype=float)
    if len(centers) < 2 * window + 1:
        return 0.0
    smoothed = np.zeros_like(centers)
    kernel = np.ones(window) / window
    for axis in range(3):
        smoothed[:, axis] = np.convolve(centers[:, axis], kernel, mode="same")
    deviation = np.linalg.norm(centers - smoothed, axis=1)
    step_dists = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    med_step = float(np.median(step_dists))
    if med_step <= 1e-12:
        return 0.0
    return float(np.percentile(deviation, 95) / med_step)


def _compute_zigzag_metrics(
    images: list[dict[str, object]],
    window: int = ZIGZAG_WINDOW,
    residual_threshold: float = ZIGZAG_RESIDUAL_THRESHOLD,
) -> tuple[float, float, float]:
    centers = np.array([np.asarray(im["C"], dtype=float) for im in images], dtype=float)
    if len(centers) < window:
        return 0.0, 0.0, 0.0

    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    valid_steps = steps[np.isfinite(steps) & (steps > 1e-12)]
    if valid_steps.size == 0:
        return 0.0, 0.0, 0.0
    med_step = float(np.median(valid_steps))

    half = window // 2
    residuals: list[float] = []
    for idx in range(half, len(centers) - half):
        local = centers[idx - half : idx + half + 1]
        local_center = local.mean(axis=0)
        try:
            _u, _s, vt = np.linalg.svd(local - local_center, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        direction = vt[0]
        offset = centers[idx] - local_center
        residual = offset - direction * float(np.dot(offset, direction))
        residuals.append(float(np.linalg.norm(residual) / med_step))

    if not residuals:
        return 0.0, 0.0, 0.0

    values = np.array(residuals, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0
    violation_ratio = float(np.mean(finite > residual_threshold))
    p95 = float(np.percentile(finite, 95))
    score = 0.8 * violation_ratio + 0.2 * min(1.0, p95 / 4.0)
    return score, violation_ratio, p95


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


def _compute_rot_accel_metrics(
    images: list[dict[str, object]],
    ratio: float = ROT_ACCEL_JUMP_RATIO,
) -> float:
    if len(images) < 3:
        return 0.0

    frame_idx = np.array([int(im["frame_idx"]) for im in images], dtype=float)
    dirs = []
    for im in images:
        r = np.asarray(im["R"], dtype=float)
        d = r.T @ np.array([0.0, 0.0, 1.0], dtype=float)
        norm = float(np.linalg.norm(d))
        if norm <= 1e-12:
            dirs.append(np.array([0.0, 0.0, 1.0], dtype=float))
        else:
            dirs.append(d / norm)
    dirs_arr = np.asarray(dirs, dtype=float)

    dt = np.diff(frame_idx)
    valid_dt = dt > 1e-9
    if np.count_nonzero(valid_dt) < 2:
        return 0.0

    dots = np.sum(dirs_arr[:-1] * dirs_arr[1:], axis=1)
    angles_deg = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    ang_vel = angles_deg[valid_dt] / dt[valid_dt]
    dt_valid = dt[valid_dt]
    if len(ang_vel) < 2:
        return 0.0

    accel_dt = 0.5 * (dt_valid[:-1] + dt_valid[1:])
    valid_accel_dt = accel_dt > 1e-9
    if np.count_nonzero(valid_accel_dt) == 0:
        return 0.0

    ang_accel = np.abs((ang_vel[1:] - ang_vel[:-1])[valid_accel_dt] / accel_dt[valid_accel_dt])
    finite = ang_accel[np.isfinite(ang_accel)]
    if finite.size == 0:
        return 0.0
    ref = float(np.median(finite))
    if ref <= 1e-12:
        return 0.0
    return float(np.mean(finite > ratio * ref))


def _compute_pose_compose_ratio(
    images: list[dict[str, object]],
    threshold_deg: float = POSE_COMPOSE_THRESHOLD_DEG,
) -> float:
    if len(images) < 3:
        return 0.0

    residuals: list[float] = []
    for i in range(len(images) - 2):
        r12, _t12 = _relative_pose_from_images(images[i], images[i + 1])
        r23, _t23 = _relative_pose_from_images(images[i + 1], images[i + 2])
        r13, _t13 = _relative_pose_from_images(images[i], images[i + 2])
        r_err = (r23 @ r12) @ r13.T
        residuals.append(_rotation_angle_deg(r_err))

    values = np.array(residuals, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite > threshold_deg))


def _project_points_px(k: np.ndarray, x_cam: np.ndarray) -> np.ndarray:
    z = np.maximum(1e-9, x_cam[:, 2:3])
    xy = x_cam[:, :2] / z
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    return np.column_stack([fx * xy[:, 0] + cx, fy * xy[:, 1] + cy])


def _robust_pair_reproj_px(
    p1: np.ndarray,
    p2: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    r_gt: np.ndarray,
    t_gt: np.ndarray,
    epi_raw: np.ndarray,
    reproj_threshold_px: float,
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
    p1n = cv2.undistortPoints(p1s.reshape(-1, 1, 2).astype(np.float64), cameraMatrix=k1, distCoeffs=None).reshape(-1, 2)
    p2n = cv2.undistortPoints(p2s.reshape(-1, 1, 2).astype(np.float64), cameraMatrix=k2, distCoeffs=None).reshape(-1, 2)
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
    p1_rep = _project_points_px(k1, x)
    p2_rep = _project_points_px(k2, x2)
    e1 = np.linalg.norm(p1_rep - p1_obs, axis=1)
    e2 = np.linalg.norm(p2_rep - p2_obs, axis=1)
    errs = 0.5 * (e1 + e2)
    finite = errs[np.isfinite(errs)]
    if finite.size == 0:
        return float("nan"), float("nan")
    finite = finite[finite < 80.0]
    if finite.size < 8:
        return float("nan"), float("nan")
    violation_ratio = float(np.mean(finite > reproj_threshold_px))
    median_err = float(np.median(finite))
    return violation_ratio, median_err


def _estimate_relative_rotation_from_matches(
    p1: np.ndarray,
    p2: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
) -> np.ndarray | None:
    if len(p1) < COMPOSE_MIN_MATCHES:
        return None
    p1n = cv2.undistortPoints(p1.reshape(-1, 1, 2).astype(np.float64), cameraMatrix=k1, distCoeffs=None).reshape(-1, 2)
    p2n = cv2.undistortPoints(p2.reshape(-1, 1, 2).astype(np.float64), cameraMatrix=k2, distCoeffs=None).reshape(-1, 2)
    focal_ref = max(1e-9, 0.25 * (k1[0, 0] + k1[1, 1] + k2[0, 0] + k2[1, 1]))
    e, mask = cv2.findEssentialMat(
        p1n.astype(np.float64),
        p2n.astype(np.float64),
        cameraMatrix=np.eye(3, dtype=np.float64),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.5 / focal_ref,
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
            p1n.astype(np.float64),
            p2n.astype(np.float64),
            cameraMatrix=np.eye(3, dtype=np.float64),
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
    *,
    epi_threshold_px: float,
    reproj_threshold_px: float,
    compose_threshold_deg: float,
    max_triplets: int,
) -> tuple[float, float, float]:
    frame_cache: dict[int, np.ndarray] = {}
    feature_cache: dict[int, tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]] = {}
    sift = cv2.SIFT_create(nfeatures=1200)
    epi_violation_all: list[float] = []
    reproj_violation_all: list[float] = []
    compose_violation_all: list[float] = []
    pair_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
    rot_adj_cache: dict[tuple[int, int], np.ndarray] = {}

    triplet_indices = _geometry_eval_indices(len(images), max_triplets=max_triplets)
    pair_indices = sorted({i for start in triplet_indices for i in (start, start + 1) if i + 1 < len(images)})
    for a in pair_indices:
        im1 = images[a]
        im2 = images[a + 1]
        frame_idx1 = int(im1["frame_idx"])
        frame_idx2 = int(im2["frame_idx"])
        g1 = _read_frame(video_path, frame_idx1, frame_cache)
        g2 = _read_frame(video_path, frame_idx2, frame_cache)
        if g1 is None or g2 is None:
            continue
        p1, p2 = _match_pair(frame_idx1, g1, frame_idx2, g2, sift, feature_cache)
        if len(p1) < 20:
            continue
        p1, p2 = _ransac_filter_matches(p1, p2)
        if len(p1) < 12:
            pair_cache[(a, a + 1)] = None
            continue
        k1 = k_map[int(im1["camera_id"])]
        k2 = k_map[int(im2["camera_id"])]
        r_gt, t_gt = _relative_pose_from_images(im1, im2)

        fmat_gt = np.linalg.inv(k2).T @ (_skew(t_gt) @ r_gt) @ np.linalg.inv(k1)
        epi_raw = _symmetric_epipolar_distance(p1, p2, fmat_gt)
        finite_epi_mask = np.isfinite(epi_raw)
        if np.count_nonzero(finite_epi_mask) < 8:
            pair_cache[(a, a + 1)] = None
            continue
        epi_raw = epi_raw[finite_epi_mask]
        p1 = p1[finite_epi_mask]
        p2 = p2[finite_epi_mask]
        epi_violation_all.append(float(np.mean(epi_raw > epi_threshold_px)))
        pair_cache[(a, a + 1)] = (p1, p2, k1)
        reproj_violation, _reproj_median = _robust_pair_reproj_px(
            p1,
            p2,
            k1,
            k2,
            r_gt,
            t_gt,
            epi_raw,
            reproj_threshold_px,
        )
        if np.isfinite(reproj_violation):
            reproj_violation_all.append(float(reproj_violation))
        r_est_adj = _estimate_relative_rotation_from_matches(p1, p2, k1, k2)
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
        compose_violation_all.append(float(_rotation_angle_deg(r_err) > compose_threshold_deg))

    epi_metric = float(np.median(np.array(epi_violation_all, dtype=float))) if epi_violation_all else 1.0
    reproj_metric = float(np.median(np.array(reproj_violation_all, dtype=float))) if reproj_violation_all else 1.0
    compose_metric = float(np.mean(np.array(compose_violation_all, dtype=float))) if compose_violation_all else 1.0
    return epi_metric, reproj_metric, compose_metric


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float)


def _case_label(case_name: str) -> str:
    return "bad" if int(case_name) <= 5 else "good"


def _plot_trajectories(
    rows: list[dict[str, object]],
    annotations_root: Path,
    out_dir: Path,
    direction_arrows: int,
) -> None:
    cases = [str(row["case"]) for row in rows]
    labels = [str(row["label"]) for row in rows]
    n = len(cases)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows), subplot_kw={"projection": "3d"})
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (case, label) in enumerate(zip(cases, labels)):
        ax = axes_flat[idx]
        images_txt = annotations_root / case / "sparse" / "0" / "images.txt"
        centers, _forward_dirs, _names = _parse_image_poses(images_txt)

        color = "tab:red" if label == "bad" else "tab:green"
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color=color, linewidth=0.5, alpha=0.8)
        ax.scatter([centers[0, 0]], [centers[0, 1]], [centers[0, 2]], c="blue", s=20, marker="o", zorder=5)
        ax.scatter([centers[-1, 0]], [centers[-1, 1]], [centers[-1, 2]], c="black", s=20, marker="s", zorder=5)
        ax.set_title(f"{case} ({label})", fontsize=11)
        ax.tick_params(labelsize=6)

        x_range = np.ptp(centers[:, 0])
        y_range = np.ptp(centers[:, 1])
        z_range = np.ptp(centers[:, 2])
        max_range = max(x_range, y_range, z_range, 1e-6)
        mid_x = np.mean([centers[:, 0].min(), centers[:, 0].max()])
        mid_y = np.mean([centers[:, 1].min(), centers[:, 1].max()])
        mid_z = np.mean([centers[:, 2].min(), centers[:, 2].max()])
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Camera Trajectories (blue=start, black=end)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=170)
    plt.close(fig)

    for idx, (case, label) in enumerate(zip(cases, labels)):
        images_txt = annotations_root / case / "sparse" / "0" / "images.txt"
        centers, forward_dirs, _names = _parse_image_poses(images_txt)
        _plot_trajectory(
            centers,
            out_dir / f"trajectory_{case}.png",
            f"Task4 Case {case} ({label}) Trajectory",
            forward_dirs=forward_dirs,
            direction_arrows=direction_arrows,
        )


def _write_summary(
    out_path: Path,
    rows: list[dict[str, object]],
) -> None:
    lines = [
        "task4_pose_quality_summary",
        f"cases={len(rows)}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_rows(selected_cases: tuple[str, ...]) -> list[dict[str, object]]:
    return [{"case": case_name, "label": _case_label(case_name)} for case_name in selected_cases]


def _run_task4_plot(
    *,
    annotations_root: Path,
    selected_cases: tuple[str, ...],
    out_root: Path,
    direction_arrows: int,
    dry_run: bool,
) -> int:
    traj_dir = out_root / "trajectories"
    rows = _case_rows(selected_cases)
    if dry_run:
        print(f"Would save task4 trajectory grid: {traj_dir / 'trajectories.png'}")
        for case_name in selected_cases:
            print(f"Would save task4 trajectory: {traj_dir / f'trajectory_{case_name}.png'}")
        return 0

    traj_dir.mkdir(parents=True, exist_ok=True)
    _plot_trajectories(rows, annotations_root, traj_dir, direction_arrows=direction_arrows)
    print(f"Saved trajectories: {traj_dir}/")
    return 0


def run_task4(cfg: Task4Config) -> int:
    annotations_root = cfg.lab1_root / "assets" / "annotations"
    selected_cases = ANNOTATION_CASES if not cfg.cases else tuple(_normalize_case_name(c) for c in cfg.cases)
    out_root = cfg.output_root
    if cfg.mode not in {"run", "plot"}:
        raise Task1Error(f"Unsupported task4 mode: {cfg.mode}. Choose from run|plot")

    if not cfg.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    if cfg.mode == "plot":
        return _run_task4_plot(
            annotations_root=annotations_root,
            selected_cases=selected_cases,
            out_root=out_root,
            direction_arrows=cfg.direction_arrows,
            dry_run=cfg.dry_run,
        )

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
            zigzag_score, _zigzag_violation_ratio, _zigzag_residual_p95 = _compute_zigzag_metrics(
                images,
                residual_threshold=cfg.zigzag_residual_threshold,
            )
            rot_accel_jump_ratio = _compute_rot_accel_metrics(
                images,
                ratio=cfg.rot_accel_jump_ratio,
            )
            if cfg.heavy_geometry:
                raise Task1Error(
                    "--heavy-geometry is disabled for annotations because images.txt has empty POINTS2D "
                    "and points3D.txt has no reconstructed points; task4 now avoids reprocessing video."
                )
            epi_dist_px = float("nan")
            reproj_err_px = float("nan")
            compose_rot_err_deg = _compute_pose_compose_ratio(images, threshold_deg=cfg.compose_threshold_deg)
            metrics = {
                "case": case_name,
                "label": _case_label(case_name),
                "num_poses": len(images),
                "points3d": points3d_count,
                "zigzag_score": zigzag_score,
                "smooth_jump_ratio": _compute_accel_jump_ratio(images, ratio=cfg.accel_jump_ratio),
                "traj_smoothness": _compute_trajectory_smoothness(images),
                "rot_accel_jump_ratio": rot_accel_jump_ratio,
                "epi_dist_px": epi_dist_px,
                "reproj_err_px": reproj_err_px,
                "compose_rot_err_deg": compose_rot_err_deg,
            }
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

    summary_path = out_root / "summary.txt"
    _write_summary(summary_path, rows)

    traj_dir = out_root / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    _plot_trajectories(rows, annotations_root, traj_dir, direction_arrows=cfg.direction_arrows)

    write_timing_csv(out_root / TIMING_FILENAME, timings)
    print(f"Saved metrics: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved trajectories: {traj_dir}/")
    print_timing_summary("Timing / task4", timings)
    return 0
