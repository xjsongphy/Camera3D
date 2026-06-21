#!/usr/bin/env python3
"""Training-dynamics figures for the lab3 report (fps4 formal run).

Reads the per-step loss_curve CSVs that the lab3 pipeline already exports under
logs/. Produces:
  chart_loss_curves.png   3-panel: 3DGS test-PSNR, Nerfacto loss, NeuS loss
Pure matplotlib, no nerfstudio/venv.
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "lab3" / "report_assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "outputs" / "lab3" / "20260620_200014_dormitory_fps4" / "logs"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def series(rows, col):
    return [(int(r["step"]), float(r[col])) for r in rows if r[col].strip() != ""]


def main():
    dgs = load(LOG / "3dgs_train_loss_curve.csv")
    nerf = load(LOG / "nerf_train_loss_curve.csv")
    neus = load(LOG / "neus_train_loss_curve.csv")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 3.5))

    # 3DGS: test PSNR vs step (this doubles as the per-iteration quality curve)
    psnr = series(dgs, "test/loss_viewpoint - psnr")
    if psnr:
        xs, ys = zip(*psnr)
        a1.plot(xs, ys, color="#d1495b", lw=1.2)
        for it in (7000, 30000):
            v = dict(psnr).get(it)
            if v is not None:
                a1.scatter([it], [v], color="#d1495b", zorder=3)
                a1.annotate(f"{v:.2f}", (it, v), textcoords="offset points",
                            xytext=(4, 5), fontsize=8, color="#d1495b")
    a1.set_title("3DGS · held-out PSNR"); a1.set_xlabel("iteration"); a1.set_ylabel("test PSNR (dB)")
    a1.grid(linestyle=":", alpha=0.4)

    # Nerfacto: train + eval loss
    for col, c, lab in [("Train Loss", "#00798c", "train"),
                        ("Eval Loss", "#d1495b", "eval")]:
        s = series(nerf, col)
        if s:
            a2.plot([x for x, _ in s], [y for _, y in s], color=c, lw=1.0, label=lab)
    a2.set_yscale("log"); a2.set_title("Nerfacto · loss"); a2.set_xlabel("iteration"); a2.set_ylabel("loss (log)")
    a2.legend(frameon=False, fontsize=8); a2.grid(linestyle=":", alpha=0.4)

    # NeuS: train + eval loss (+ eikonal)
    for col, c, lab in [("Train Loss", "#00798c", "train"),
                        ("Eval Loss", "#d1495b", "eval")]:
        s = series(neus, col)
        if s:
            a3.plot([x for x, _ in s], [y for _, y in s], color=c, lw=1.0, label=lab)
    a3.set_yscale("log"); a3.set_title("NeuS · loss"); a3.set_xlabel("iteration"); a3.set_ylabel("loss (log)")
    a3.legend(frameon=False, fontsize=8); a3.grid(linestyle=":", alpha=0.4)

    for a in (a1, a2, a3):
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "chart_loss_curves.png", dpi=150)
    print("wrote", OUT / "chart_loss_curves.png")


if __name__ == "__main__":
    main()
