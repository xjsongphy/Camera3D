from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lab1.geometry_utils import parse_image_poses_sorted
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


def _count_points3d(points3d_txt: Path) -> int:
    count = 0
    with points3d_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            count += 1
    return count


def _compute_step_metrics(centers: np.ndarray) -> dict[str, float]:
    if len(centers) < 3:
        return {
            "step_median": 0.0,
            "step_p95": 0.0,
            "step_max": 0.0,
            "step_jump_ratio": 0.0,
            "accel_p95": 0.0,
            "accel_jump_ratio": 0.0,
        }

    step_vectors = np.diff(centers, axis=0)
    step = np.linalg.norm(step_vectors, axis=1)
    step_median = float(np.median(step))
    step_p95 = float(np.percentile(step, 95))
    step_max = float(np.max(step))
    step_jump_ratio = float(np.mean(step > max(1e-12, 3.0 * step_median)))

    if len(step_vectors) < 2:
        accel = np.zeros(0, dtype=float)
    else:
        accel = np.linalg.norm(np.diff(step_vectors, axis=0), axis=1)
    accel_p95 = float(np.percentile(accel, 95)) if accel.size else 0.0
    accel_median = float(np.median(accel)) if accel.size else 0.0
    accel_jump_ratio = float(np.mean(accel > max(1e-12, 3.0 * accel_median))) if accel.size else 0.0

    return {
        "step_median": step_median,
        "step_p95": step_p95,
        "step_max": step_max,
        "step_jump_ratio": step_jump_ratio,
        "accel_p95": accel_p95,
        "accel_jump_ratio": accel_jump_ratio,
    }


def _compute_rotation_metrics(forward_dirs: np.ndarray) -> dict[str, float]:
    if len(forward_dirs) < 3:
        return {
            "rot_deg_median": 0.0,
            "rot_deg_p95": 0.0,
            "rot_deg_max": 0.0,
            "rot_jump_ratio": 0.0,
        }

    dots = np.sum(forward_dirs[:-1] * forward_dirs[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))
    rot_median = float(np.median(angles_deg))
    rot_p95 = float(np.percentile(angles_deg, 95))
    rot_max = float(np.max(angles_deg))
    rot_jump_ratio = float(np.mean(angles_deg > max(1e-12, 3.0 * rot_median)))
    return {
        "rot_deg_median": rot_median,
        "rot_deg_p95": rot_p95,
        "rot_deg_max": rot_max,
        "rot_jump_ratio": rot_jump_ratio,
    }


def _compute_quality_score(metrics: dict[str, float]) -> tuple[float, float]:
    step_scale = metrics["step_p95"] / max(metrics["step_median"], 1e-9)
    accel_scale = metrics["accel_p95"] / max(metrics["step_median"], 1e-9)
    rot_scale = metrics["rot_deg_p95"] / max(metrics["rot_deg_median"], 1e-6)

    penalty = (
        0.30 * math.log1p(max(0.0, step_scale - 1.0))
        + 0.25 * metrics["step_jump_ratio"] * 4.0
        + 0.25 * math.log1p(accel_scale)
        + 0.10 * metrics["accel_jump_ratio"] * 4.0
        + 0.10 * math.log1p(max(0.0, rot_scale - 1.0))
    )
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
        images_txt = case_root / "sparse" / "0" / "images.txt"
        points3d_txt = case_root / "sparse" / "0" / "points3D.txt"
        if not images_txt.exists() or not points3d_txt.exists():
            raise Task1Error(f"Missing annotation files under {case_root}")

        print(f"\n=== Task4 / {case_name} ===")
        with timed_block(case_name, timings):
            centers, forward_dirs, _names = parse_image_poses_sorted(images_txt, error_cls=Task1Error)
            points3d_count = _count_points3d(points3d_txt)
            step_metrics = _compute_step_metrics(centers)
            rot_metrics = _compute_rotation_metrics(forward_dirs)
            metrics = {
                "case": case_name,
                "label": _case_label(case_name),
                "num_poses": len(centers),
                "points3d": points3d_count,
                **step_metrics,
                **rot_metrics,
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
