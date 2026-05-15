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
                "smooth_jump_ratio": float(r["smooth_jump_ratio"]),
                "epi_dist_px": float(r["epi_dist_px"]),
                "reproj_err_px": float(r["reproj_err_px"]),
                "compose_rot_err_deg": float(r["compose_rot_err_deg"]),
                "quality_score": float(r["quality_score"]),
            })
    return rows


def main() -> None:
    rows = _load_rows()
    OUT.mkdir(parents=True, exist_ok=True)

    cases = [r["case"] for r in rows]
    labels = [r["label"] for r in rows]
    colors = ["tab:red" if v == "bad" else "tab:green" for v in labels]

    smooth = np.array([r["smooth_jump_ratio"] for r in rows], dtype=float)
    epi = np.array([r["epi_dist_px"] for r in rows], dtype=float)
    reproj = np.array([r["reproj_err_px"] for r in rows], dtype=float)
    compose = np.array([r["compose_rot_err_deg"] for r in rows], dtype=float)
    qscore = np.array([r["quality_score"] for r in rows], dtype=float)

    # individual metrics
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.flatten()
    x = np.arange(len(rows))
    items = [
        (smooth, "smooth_jump_ratio"),
        (epi, "epi_dist_px"),
        (reproj, "reproj_err_px"),
        (compose, "compose_rot_err_deg"),
    ]
    for ax, (vals, title) in zip(axes, items):
        ax.bar(x, vals, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(cases)
    fig.suptitle("Task4 Individual Metrics (Current 4 Categories)")
    fig.tight_layout()
    fig.savefig(OUT / "task4_individual_metrics.png", dpi=180)
    plt.close(fig)

    # penalty breakdown
    smooth_term = 0.20 * np.log1p(smooth / 3.0)
    epi_term = 0.30 * np.log1p(epi / 2.0)
    reproj_term = 0.40 * np.log1p(4.0 / np.maximum(reproj, 1e-6))
    compose_term = 0.10 * np.log1p(8.0 / np.maximum(compose, 1e-6))

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bottom = np.zeros(len(rows), dtype=float)
    for vals, name, color in [
        (smooth_term, "smooth", "#4e79a7"),
        (epi_term, "epi", "#f28e2b"),
        (reproj_term, "reproj", "#e15759"),
        (compose_term, "compose", "#76b7b2"),
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
