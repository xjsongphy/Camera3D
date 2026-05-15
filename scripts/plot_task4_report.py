from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs" / "lab1" / "task4" / "case_metrics.csv"
GEOMETRY_CSV_PATH = ROOT / "outputs" / "lab1" / "task4_geometry_fps16" / "case_metrics.csv"
OUT = ROOT / "docs" / "lab1" / "report_assets" / "task4"


def _load_rows():
    """Load and merge trajectory metrics with geometry metrics."""
    rows = {}

    # Load trajectory metrics
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            case = r["case"]
            rows[case] = {
                "case": case,
                "label": r["label"],
                "zigzag_score": float(r["zigzag_score"]),
                "smooth_jump_ratio": float(r["smooth_jump_ratio"]),
                "traj_smoothness": float(r["traj_smoothness"]),
                "rot_accel_jump_ratio": float(r["rot_accel_jump_ratio"]),
            }

    # Load geometry metrics and merge
    if GEOMETRY_CSV_PATH.exists():
        with GEOMETRY_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                case = r["case"]
                if case in rows:
                    rows[case]["epi_dist_px"] = float(r["epi_dist_px"])
                    rows[case]["reproj_err_px"] = float(r["reproj_err_px"])

    return list(rows.values())


def main() -> None:
    rows = _load_rows()
    OUT.mkdir(parents=True, exist_ok=True)

    cases = [r["case"] for r in rows]
    labels = [r["label"] for r in rows]
    colors = ["tab:red" if v == "bad" else "tab:green" for v in labels]

    zigzag = np.array([r["zigzag_score"] for r in rows], dtype=float)
    smooth = np.array([r["smooth_jump_ratio"] for r in rows], dtype=float)
    rot_smooth = np.array([r["rot_accel_jump_ratio"] for r in rows], dtype=float)
    epi_dist = np.array([r.get("epi_dist_px", np.nan) for r in rows], dtype=float)
    reproj_err = np.array([r.get("reproj_err_px", np.nan) for r in rows], dtype=float)

    # individual metrics - 5 metrics in 3 rows (2+2+1 layout)
    fig, axes = plt.subplots(3, 2, figsize=(9, 9), sharex=True,
                             gridspec_kw={'wspace': 0.2, 'hspace': 0.35})
    x = np.arange(len(rows))

    # Row 1: zigzag_score, smooth_jump_ratio
    items_row1 = [
        (axes[0, 0], zigzag, "zigzag_score"),
        (axes[0, 1], smooth, "smooth_jump_ratio"),
    ]

    # Row 2: rot_accel_jump_ratio, epi_dist_px
    items_row2 = [
        (axes[1, 0], rot_smooth, "rot_accel_jump_ratio"),
        (axes[1, 1], epi_dist, "epi_dist_px"),
    ]

    # Row 3: reproj_err_px (spans two columns)
    items_row3 = [
        (axes[2, 0], reproj_err, "reproj_err_px"),
    ]

    for ax, vals, title in items_row1 + items_row2 + items_row3:
        ax.bar(x, vals, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(cases)

    # Hide the unused subplot (axes[2, 1])
    axes[2, 1].set_visible(False)

    # Let reproj_err_px span both columns in row 3
    axes[2, 0].set_position([0.125, 0.11, 0.775, 0.18])

    fig.suptitle("Task4 Pose Quality Metrics")
    fig.tight_layout()
    fig.savefig(OUT / "task4_individual_metrics.png", dpi=180)
    plt.close(fig)

    print(f"Saved: {OUT / 'task4_individual_metrics.png'}")


if __name__ == "__main__":
    main()
