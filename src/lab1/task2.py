from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lab1.colmap_utils import (
    require_tool,
    run_cmd,
    run_feature_extractor,
    run_model_converter,
    run_sequential_matcher,
)
from lab1.logging_utils import print_timing_summary, timed_block, write_timing_csv
from lab1.task1 import Task1Error, _format_float_tag, _has_any_frames, _quat_to_rot

TIMING_FILENAME = "timing.csv"


@dataclass
class Task2Config:
    lab1_root: Path
    task1_output_root: Path
    output_root: Path
    source_fps: float
    colmap_bin: str
    force: bool
    dry_run: bool
    stage: str = "all"
    subseq_specs: tuple[str, ...] = ()


def _parse_colmap_poses(images_txt: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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
                raise Task1Error(f"Malformed COLMAP images.txt pose line: {pose_line}")
            if not parts[0].isdigit():
                raise Task1Error(f"Expected image id at pose line start, got: {pose_line}")

            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = parts[9]
            r = _quat_to_rot(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=float)
            c = -r.T @ t
            poses[name] = (c, np.array([qw, qx, qy, qz, tx, ty, tz], dtype=float))

            _ = f.readline()
    if not poses:
        raise Task1Error(f"No image poses parsed from {images_txt}")
    return poses


def _write_subset_images_txt(
    src_images_txt: Path,
    dst_images_txt: Path,
    keep_names: set[str],
) -> int:
    kept = 0
    next_id = 1
    with src_images_txt.open("r", encoding="utf-8") as fin, dst_images_txt.open("w", encoding="utf-8") as fout:
        fout.write("# Subset from full-sequence reconstruction\n")
        while True:
            pose_line = fin.readline()
            if not pose_line:
                break
            if pose_line.startswith("#") or not pose_line.strip():
                continue
            _ = fin.readline()
            parts = pose_line.strip().split()
            if len(parts) < 10 or not parts[0].isdigit():
                raise Task1Error(f"Malformed COLMAP images.txt pose line: {pose_line.strip()}")
            name = parts[9]
            if name not in keep_names:
                continue
            parts[0] = str(next_id)
            next_id += 1
            fout.write(" ".join(parts) + "\n")
            fout.write("\n")
            kept += 1
    return kept


def _run_colmap_subset(
    images_dir: Path,
    sparse_root: Path,
    db_path: Path,
    colmap_bin: str,
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
            mask_path=None,
            camera_mask_path=None,
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
    with timed_block("mapper", timings):
        run_cmd(
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
            error_cls=Task1Error,
        )
    model_dir = sparse_root / "0"
    if not dry_run and not model_dir.exists():
        raise Task1Error(f"COLMAP mapper produced no model under: {model_dir}")
    with timed_block("model_converter", timings):
        run_model_converter(
            colmap_bin=colmap_bin,
            model_dir=model_dir,
            dry_run=dry_run,
            error_cls=Task1Error,
        )
    return model_dir


def _require_prepared_subset_images(images_dir: Path, expected_names: list[str]) -> None:
    if not images_dir.exists() or not _has_any_frames(images_dir):
        raise Task1Error(
            f"Prepared subset frames not found under {images_dir}. "
            "Run: uv run lab1 task2 --source-fps ... --stage prepare"
        )
    missing = [name for name in expected_names if not (images_dir / name).exists()]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise Task1Error(
            f"Prepared subset frames are incomplete under {images_dir}. "
            f"Missing {len(missing)} frame(s): {preview}{suffix}. "
            "Run: uv run lab1 task2 --source-fps ... --stage prepare --force"
        )


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.shape[0] < 3:
        raise Task1Error("Sim(3) alignment requires matched trajectories with at least 3 points.")
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


def _apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (scale * (rot @ points.T)).T + trans


def _ate(ref: np.ndarray, est: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((ref - est) ** 2, axis=1))))


def _plot_overlay(ref: np.ndarray, est_aligned: np.ndarray, out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], marker="o", markersize=2, label="Method A (subset from full)")
    ax.plot(est_aligned[:, 0], est_aligned[:, 1], est_aligned[:, 2], marker="^", markersize=2, label="Method B (subset SfM, aligned)")
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


def _default_subseq_ranges(n: int) -> list[tuple[int, int, str]]:
    # 0-based half-open ranges.
    # These defaults are chosen from the S1-2 full trajectory geometry:
    # 1) a medium-length return segment with larger spatial extent than the failed short local return,
    # 2) a clean single-direction scan segment that typically reconstructs stably,
    # 3) a long return segment that contains the strongest global self-return cue short of the full sequence.
    specs = [
        (round(n * 0.478261), round(n * 0.681159), "return_mid"),
        (round(n * 0.594203), round(n * 0.768116), "scan_stable"),
        (round(n * 0.492754), round(n * 0.927536), "return_long"),
    ]
    fixed: list[tuple[int, int, str]] = []
    for s, e, name in specs:
        s = max(0, min(s, n - 1))
        e = max(s + 1, min(e, n))
        fixed.append((s, e, name))
    return fixed


def _parse_subseq_specs(specs: tuple[str, ...], n: int) -> list[tuple[int, int, str]]:
    parsed: list[tuple[int, int, str]] = []
    seen_names: set[str] = set()
    for raw in specs:
        parts = raw.split(":")
        if len(parts) != 3:
            raise Task1Error(
                f"Invalid --subseq value: {raw}. Expected format START:END:NAME "
                "(1-based inclusive frame indices)."
            )
        start_s, end_s, name = parts
        if not start_s.isdigit() or not end_s.isdigit():
            raise Task1Error(f"Invalid --subseq value: {raw}. START and END must be positive integers.")
        start = int(start_s)
        end = int(end_s)
        if start < 1 or end < 1 or start > end:
            raise Task1Error(f"Invalid --subseq range: {raw}. Require 1 <= START <= END.")
        if end > n:
            raise Task1Error(f"Invalid --subseq range: {raw}. END exceeds available frames ({n}).")
        clean_name = name.strip()
        if not clean_name:
            raise Task1Error(f"Invalid --subseq value: {raw}. NAME must be non-empty.")
        if clean_name in seen_names:
            raise Task1Error(f"Duplicate --subseq name: {clean_name}")
        seen_names.add(clean_name)
        parsed.append((start - 1, end, clean_name))
    return parsed


def run_task2(cfg: Task2Config) -> int:
    if cfg.source_fps <= 0:
        raise Task1Error(f"source_fps must be positive, got {cfg.source_fps}")
    if cfg.stage not in {"all", "prepare", "sfm", "analyze"}:
        raise Task1Error(f"Unsupported stage: {cfg.stage}. Choose from all|prepare|sfm|analyze")

    param_tag = f"fps{_format_float_tag(cfg.source_fps)}"
    task1_case_root = cfg.task1_output_root / f"S1-2_{param_tag}"
    full_images_dir = task1_case_root / "images"
    full_sparse0 = task1_case_root / "sparse" / "0"
    full_images_txt = full_sparse0 / "images.txt"
    full_cameras_txt = full_sparse0 / "cameras.txt"

    if not full_images_dir.exists() or not full_images_txt.exists() or not full_cameras_txt.exists():
        raise Task1Error(
            f"Task2 requires task1 S1-2 outputs first: {task1_case_root}\n"
            f"Run: uv run lab1 task1 --videos S1-2 --fps {cfg.source_fps:g} --stage all"
        )
    if not cfg.dry_run and cfg.stage in {"all", "sfm"}:
        require_tool(cfg.colmap_bin, error_cls=Task1Error)

    all_frames = sorted([p.name for p in full_images_dir.glob("*.jpg")])
    if len(all_frames) < 10:
        raise Task1Error(f"Not enough frames in {full_images_dir} (found {len(all_frames)})")

    all_poses = _parse_colmap_poses(full_images_txt)
    task2_root = cfg.output_root / f"S1-2_{param_tag}"
    if not cfg.dry_run:
        task2_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    subseq_ranges = _parse_subseq_specs(cfg.subseq_specs, len(all_frames)) if cfg.subseq_specs else _default_subseq_ranges(len(all_frames))
    for idx, (start, end, seq_name) in enumerate(subseq_ranges, start=1):
        timings: dict[str, float] = {}
        subset_names = all_frames[start:end]
        keep = set(subset_names)
        sub_tag = f"seq{idx:02d}_{seq_name}_{start+1:06d}-{end:06d}"
        sub_root = task2_root / sub_tag
        method_a_sparse = sub_root / "method_a" / "sparse" / "0"
        method_b_root = sub_root / "method_b"
        method_b_images = method_b_root / "images"
        method_b_sparse = method_b_root / "sparse"
        method_b_db = method_b_root / "database.db"
        overlay_png = sub_root / "trajectory_overlay.png"
        metrics_txt = sub_root / "metrics.txt"
        timing_path = sub_root / TIMING_FILENAME

        print(f"\n=== Task2 / {sub_tag} ===")
        if not cfg.dry_run:
            method_a_sparse.mkdir(parents=True, exist_ok=True)
            method_b_images.mkdir(parents=True, exist_ok=True)

        if cfg.stage in {"all", "prepare"}:
            with timed_block("prepare", timings):
                if cfg.force and not cfg.dry_run and method_b_images.exists():
                    shutil.rmtree(method_b_images)
                    method_b_images.mkdir(parents=True, exist_ok=True)
                for name in subset_names:
                    src = full_images_dir / name
                    dst = method_b_images / name
                    if not dst.exists() or cfg.force:
                        if not cfg.dry_run:
                            shutil.copy2(src, dst)
                if not cfg.dry_run:
                    shutil.copy2(full_cameras_txt, method_a_sparse / "cameras.txt")
                    kept = _write_subset_images_txt(full_images_txt, method_a_sparse / "images.txt", keep)
                    (method_a_sparse / "points3D.txt").write_text("", encoding="utf-8")
                    print(f"Method A poses kept: {kept}/{len(subset_names)}")

        if cfg.stage in {"all", "sfm"}:
            if cfg.stage == "sfm":
                _require_prepared_subset_images(method_b_images, subset_names)
            with timed_block("sfm_total", timings):
                _run_colmap_subset(
                    images_dir=method_b_images,
                    sparse_root=method_b_sparse,
                    db_path=method_b_db,
                    colmap_bin=cfg.colmap_bin,
                    force=cfg.force,
                    dry_run=cfg.dry_run,
                    timings=timings,
                )

        if cfg.stage in {"all", "analyze"}:
            if cfg.dry_run:
                print_timing_summary(f"Timing / {sub_tag}", timings)
                continue
            with timed_block("analyze", timings):
                b_images_txt = method_b_sparse / "0" / "images.txt"
                if not b_images_txt.exists():
                    raise Task1Error(f"Method B sparse result missing: {b_images_txt}")
                poses_b = _parse_colmap_poses(b_images_txt)

                common = sorted([n for n in subset_names if n in all_poses and n in poses_b])
                if len(common) < 3:
                    raise Task1Error(f"Too few common registered frames for alignment in {sub_tag}: {len(common)}")

                a_xyz = np.stack([all_poses[n][0] for n in common], axis=0)
                b_xyz = np.stack([poses_b[n][0] for n in common], axis=0)
                scale, rot, trans = _umeyama_sim3(b_xyz, a_xyz)
                b_aligned = _apply_sim3(b_xyz, scale, rot, trans)
                ate_val = _ate(a_xyz, b_aligned)
                endpoint_distance = float(np.linalg.norm(a_xyz[-1] - a_xyz[0]))
                path_length = float(np.linalg.norm(np.diff(a_xyz, axis=0), axis=1).sum()) if len(a_xyz) >= 2 else 0.0
                endpoint_ratio = endpoint_distance / path_length if path_length > 0 else 0.0
                _plot_overlay(a_xyz, b_aligned, overlay_png, f"{sub_tag} | source_fps={cfg.source_fps:g} | ATE={ate_val:.4f}")

                row = {
                    "subseq": sub_tag,
                    "subset_frames": len(subset_names),
                    "common_registered": len(common),
                    "ate": ate_val,
                    "scale": scale,
                    "endpoint_distance": endpoint_distance,
                    "path_length": path_length,
                    "endpoint_ratio": endpoint_ratio,
                }
                summary_rows.append(row)
                metrics_txt.write_text(
                    "\n".join(
                        [
                            f"subseq={sub_tag}",
                            f"source_fps={cfg.source_fps:g}",
                            f"subset_frames={len(subset_names)}",
                            f"common_registered={len(common)}",
                            f"sim3_scale={scale:.8f}",
                            f"endpoint_distance={endpoint_distance:.8f}",
                            f"path_length={path_length:.8f}",
                            f"endpoint_path_ratio={endpoint_ratio:.8f}",
                            f"ate={ate_val:.8f}",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            print(f"Saved: {overlay_png}")
            print(f"ATE={ate_val:.6f}, common={len(common)}")
        if not cfg.dry_run:
            write_timing_csv(timing_path, timings)
            print(f"Saved timing summary: {timing_path}")
        print_timing_summary(f"Timing / {sub_tag}", timings)

    if cfg.stage in {"all", "analyze"} and not cfg.dry_run:
        summary_path = task2_root / "summary.csv"
        lines = ["subseq,subset_frames,common_registered,ate,scale,endpoint_distance,path_length,endpoint_ratio"]
        for r in summary_rows:
            lines.append(
                f"{r['subseq']},{r['subset_frames']},{r['common_registered']},{r['ate']:.8f},"
                f"{r['scale']:.8f},{r['endpoint_distance']:.8f},{r['path_length']:.8f},{r['endpoint_ratio']:.8f}"
            )
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved summary: {summary_path}")

    print("\nTask2 completed.")
    return 0
