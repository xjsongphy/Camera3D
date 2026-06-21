#!/usr/bin/env python3
"""Render quantitative charts (matplotlib) for the lab3 dormitory report."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "docs" / "lab3" / "report_assets" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# Formal runs only (fps2/4/8): same preprocessing + hyperparams, only fps differs.
# fps5 (test run, different protocol) is excluded from the sweep.
FPS = [2, 4, 8]
DGS_PSNR = [25.79, 27.70, 18.03]
NERF_PSNR = [20.65, 18.47, 12.23]
DGS_SSIM = [0.835, 0.868, 0.722]
NERF_SSIM = [0.749, 0.688, 0.597]
DGS_LPIPS = [0.368, 0.353, 0.562]
NERF_LPIPS = [0.452, 0.506, 0.709]
TRAIN_N = [278, 554, 1107]

C_DGS = "#d1495b"
C_NERF = "#00798c"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def chart_psnr():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(FPS, DGS_PSNR, "-o", color=C_DGS, label="3DGS")
    ax.plot(FPS, NERF_PSNR, "-s", color=C_NERF, label="Nerfacto")
    for x, y in zip(FPS, DGS_PSNR):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=C_DGS)
    for x, y in zip(FPS, NERF_PSNR):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8, color=C_NERF)
    ax.set_xlabel("video sampling fps"); ax.set_ylabel("PSNR (dB) ↑")
    ax.set_xticks(FPS); ax.set_title("Held-out PSNR vs. sampling fps")
    ax.legend(frameon=False); style(ax); fig.tight_layout()
    fig.savefig(OUT / "chart_psnr_fps.png", dpi=150); plt.close(fig)


def chart_ssim_lpips():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.3))
    a1.plot(FPS, DGS_SSIM, "-o", color=C_DGS, label="3DGS")
    a1.plot(FPS, NERF_SSIM, "-s", color=C_NERF, label="Nerfacto")
    a1.set_xticks(FPS); a1.set_xlabel("fps"); a1.set_ylabel("SSIM ↑"); a1.set_title("SSIM"); style(a1); a1.legend(frameon=False)
    a2.plot(FPS, DGS_LPIPS, "-o", color=C_DGS, label="3DGS")
    a2.plot(FPS, NERF_LPIPS, "-s", color=C_NERF, label="Nerfacto")
    a2.set_xticks(FPS); a2.set_xlabel("fps"); a2.set_ylabel("LPIPS ↓"); a2.set_title("LPIPS"); style(a2); a2.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "chart_ssim_lpips_fps.png", dpi=150); plt.close(fig)


def chart_efficiency():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.38
    x = np.arange(len(FPS))
    dgs_t = [2704.6, 3032.0, 3169.9]
    nerf_t = [2892.5, 3016.2, np.nan]
    ax.bar(x - width/2, dgs_t, width, color=C_DGS, label="3DGS train")
    ax.bar(x + width/2, nerf_t, width, color=C_NERF, label="Nerfacto train")
    ax.set_xticks(x); ax.set_xticklabels([f"{f}fps\n({n}img)" for f, n in zip(FPS, TRAIN_N)])
    ax.set_ylabel("training time (s)"); ax.set_title("Training time vs. sampling fps")
    ax.legend(frameon=False); style(ax); fig.tight_layout()
    fig.savefig(OUT / "chart_efficiency_fps.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    chart_psnr(); chart_ssim_lpips(); chart_efficiency()
    print("charts written to", OUT)
