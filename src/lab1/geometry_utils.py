from __future__ import annotations

from pathlib import Path

import numpy as np


def quat_to_rot(qw: float, qx: float, qy: float, qz: float, *, error_cls: type[Exception] = RuntimeError) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        raise error_cls("Encountered near-zero quaternion norm while parsing COLMAP poses.")
    q /= norm
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def parse_colmap_pose_map(
    images_txt: Path, *, error_cls: type[Exception] = RuntimeError
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
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
                raise error_cls(f"Malformed COLMAP images.txt pose line: {pose_line}")
            if not parts[0].isdigit():
                raise error_cls(f"Expected image id at pose line start, got: {pose_line}")

            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = parts[9]
            r = quat_to_rot(qw, qx, qy, qz, error_cls=error_cls)
            t = np.array([tx, ty, tz], dtype=float)
            c = -r.T @ t
            poses[name] = (c, np.array([qw, qx, qy, qz, tx, ty, tz], dtype=float))

            _ = f.readline()
    if not poses:
        raise error_cls(f"No image poses parsed from {images_txt}")
    return poses


def parse_image_poses_sorted(
    images_txt: Path, *, error_cls: type[Exception] = RuntimeError
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pose_map = parse_colmap_pose_map(images_txt, error_cls=error_cls)
    names_sorted = sorted(pose_map.keys())
    centers = np.array([pose_map[name][0] for name in names_sorted], dtype=float)

    forward_dirs: list[np.ndarray] = []
    for name in names_sorted:
        qw, qx, qy, qz, _tx, _ty, _tz = pose_map[name][1]
        r = quat_to_rot(float(qw), float(qx), float(qy), float(qz), error_cls=error_cls)
        forward = r.T @ np.array([0.0, 0.0, 1.0], dtype=float)
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-12:
            forward = forward / forward_norm
        forward_dirs.append(forward)
    return centers, np.array(forward_dirs, dtype=float), names_sorted


def parse_image_centers_sorted(images_txt: Path, *, error_cls: type[Exception] = RuntimeError) -> tuple[np.ndarray, list[str]]:
    centers, _forward_dirs, names = parse_image_poses_sorted(images_txt, error_cls=error_cls)
    return centers, names


def umeyama_sim3(
    src: np.ndarray, dst: np.ndarray, *, error_cls: type[Exception] = RuntimeError
) -> tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.shape[0] < 3:
        raise error_cls("Sim(3) alignment requires matched trajectories with at least 3 points.")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s[2, 2] = -1
    r = u @ s @ vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.trace(np.diag(d) @ s) / var_src)
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (scale * (rot @ points.T)).T + trans
