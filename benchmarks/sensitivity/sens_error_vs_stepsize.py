#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from derivkit.derivative_kit import DerivativeKit
from common.style import (
    apply_plot_style,
    _DEFAULT_COLORS,
    _DEFAULT_FONTSIZE,
    _DEFAULT_LINEWIDTH,
)

# --- palette / style aliases ---
blue_color = _DEFAULT_COLORS["blue"]
red_color  = _DEFAULT_COLORS["red"]
DEFAULT_LINEWIDTH = _DEFAULT_LINEWIDTH
DEFAULT_FONTSIZE  = _DEFAULT_FONTSIZE

# ===== Problem setup (same linear model as your demos) =====
X0      = 0.30
ORDER   = 1
TRUE_A  = 1.700
TRUE_B  = -0.2

# Estimator knobs
N_POINTS = 25
BASE_ABS = 1e-3
RIDGE    = 1e-8

# Sweep over explicit step sizes (spacing)
SPACING_GRID = np.linspace(0.05, 0.55, 11)

# MC per spacing to estimate error robustly
DRAWS       = 400
SIGMA_NOISE = 0.01
SEED_MASTER = 123

def f_clean(x, a=TRUE_A, b=TRUE_B):
    x = np.asarray(x, float)
    return a * x + b

def one_draw_estimate(spacing: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    def g(xx):
        xx = np.asarray(xx, float)
        return f_clean(xx) + rng.normal(0.0, SIGMA_NOISE, size=xx.shape)
    dk = DerivativeKit(g, X0)
    val = dk.adaptive.differentiate(
        ORDER,
        n_points=int(N_POINTS),
        spacing=float(spacing),   # <<< explicit step size under test
        base_abs=BASE_ABS,
        ridge=RIDGE,
    )
    return float(np.asarray(val).ravel()[0])

def main():
    apply_plot_style(base=blue_color)

    rmse = []
    bias = []

    master = np.random.default_rng(SEED_MASTER)
    seeds = master.integers(1_000_000_000, size=DRAWS, dtype=np.int64)

    for sp in SPACING_GRID:
        ests = [one_draw_estimate(sp, int(s)) for s in seeds]
        ests = np.asarray(ests, float)
        err  = ests - TRUE_A
        rmse.append(np.sqrt(np.mean(err**2)))
        bias.append(np.mean(err))

    rmse = np.asarray(rmse, float)
    bias = np.asarray(bias, float)

    # ---- plot: error vs step size ----
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    ax.plot(SPACING_GRID, rmse, lw=DEFAULT_LINEWIDTH, color=blue_color, marker="o", label="RMSE")
    ax.axhline(0.0, ls="--", lw=DEFAULT_LINEWIDTH, color=red_color, label="zero bias")

    ax.set_xlabel("spacing", fontsize=DEFAULT_FONTSIZE)
    ax.set_ylabel("error (RMSE of slope)", fontsize=DEFAULT_FONTSIZE)
    ax.set_title("Estimator sensitivity: error vs step size", fontsize=DEFAULT_FONTSIZE+1)
    ax.legend(frameon=True, loc="best")

    outdir = Path("../demos/plots"); outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "error_vs_stepsize.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {outfile}")
    # print bias too
    for sp, r, b in zip(SPACING_GRID, rmse, bias):
        print(f"spacing={sp:.3f}  RMSE={r:.6g}  bias={b:.6g}")

if __name__ == "__main__":
    main()
