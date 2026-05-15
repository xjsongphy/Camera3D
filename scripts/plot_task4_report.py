from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs" / "lab1" / "task4" / "case_metrics.csv"
OUT = ROOT / "docs" / "lab1" / "report_assets" / "task4"


def _load_rows():
    rows = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "case": r["case"],
                "label": r["label"],
                "zigzag_score": float(r["zigzag_score"]),
                "zigzag_residual_p95": float(r["zigzag_residual_p95"]),
                "smooth_jump_ratio": float(r["smooth_jump_ratio"]),
                "traj_smoothness": float(r["traj_smoothness"]),
                "quality_score": float(r["quality_score"]),
            })
    return rows


def main() -> None:
    rows = _load_rows()
    OUT.mkdir(parents=True, exist_ok=True)

    cases = [r["case"] for r in rows]
    labels = [r["label"] for r in rows]
    colors = ["tab:red" if v == "bad" else "tab:green" for v in labels]

    zigzag = np.array([r["zigzag_score"] for r in rows], dtype=float)
    zigzag_p95 = np.array([r["zigzag_residual_p95"] for r in rows], dtype=float)
    smooth = np.array([r["smooth_jump_ratio"] for r in rows], dtype=float)
    traj = np.array([r["traj_smoothness"] for r in rows], dtype=float)
    qscore = np.array([r["quality_score"] for r in rows], dtype=float)

    # individual metrics
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.flatten()
    x = np.arange(len(rows))
    items = [
        (zigzag, "zigzag_score"),
        (zigzag_p95, "zigzag_residual_p95"),
        (smooth, "smooth_jump_ratio"),
        (traj, "traj_smoothness"),
    ]
    for ax, (vals, title) in zip(axes, items):
        ax.bar(x, vals, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(cases)
    fig.suptitle("Task4 Lightweight Trajectory Metrics")
    fig.tight_layout()
    fig.savefig(OUT / "task4_individual_metrics.png", dpi=180)
    plt.close(fig)

    # penalty breakdown
    zigzag_term = 0.90 * np.log1p(50.0 * zigzag)
    accel_term = 0.05 * np.log1p(4.0 * smooth)
    traj_term = 0.05 * np.log1p(0.5 * traj)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bottom = np.zeros(len(rows), dtype=float)
    for vals, name, color in [
        (zigzag_term, "zigzag", "#4e79a7"),
        (accel_term, "accel", "#f28e2b"),
        (traj_term, "trajectory", "#76b7b2"),
    ]:
        ax.bar(cases, vals, bottom=bottom, label=name, color=color)
        bottom += vals
    ax.set_title("Task4 Penalty Breakdown")
    ax.set_ylabel("Penalty Contribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "task4_penalty_breakdown.png", dpi=180)
    plt.close(fig)

    # quality score
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(cases, qscore, color=colors)
    ax.set_title("Task4 Quality Score")
    ax.set_xlabel("Case")
    ax.set_ylabel("Quality Score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "task4_quality_score.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
