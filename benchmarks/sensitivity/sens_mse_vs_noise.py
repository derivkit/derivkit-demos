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

# Estimator knobs (keep simple/standard)
N_POINTS = 25
BASE_ABS = 1e-3
RIDGE    = 1e-8

# Monte Carlo sweep
DRAWS        = 500
NOISE_GRID   = np.geomspace(5e-4, 5e-2, 10)  # sweep sigma_noise
SEED_MASTER  = 42

def f_clean(x, a=TRUE_A, b=TRUE_B):
    x = np.asarray(x, float)
    return a * x + b

def one_draw_estimate(sigma: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    def g(xx):
        xx = np.asarray(xx, float)
        return f_clean(xx) + rng.normal(0.0, sigma, size=xx.shape)
    dk = DerivativeKit(g, X0)
    val = dk.adaptive.differentiate(
        ORDER,
        n_points=int(N_POINTS),
        spacing="auto",        # <<< default spacing showcased
        base_abs=BASE_ABS,
        ridge=RIDGE,
    )
    return float(np.asarray(val).ravel()[0])

def main():
    apply_plot_style(base=blue_color)

    rmse = []
    bias = []

    # fixed master RNG to get reproducible per-noise seeds
    master = np.random.default_rng(SEED_MASTER)

    for sigma in NOISE_GRID:
        ests = []
        for _ in range(DRAWS):
            seed = int(master.integers(1_000_000_000))
            ests.append(one_draw_estimate(float(sigma), seed))
        ests = np.asarray(ests, float)
        err  = ests - TRUE_A
        rmse.append(np.sqrt(np.mean(err**2)))
        bias.append(np.mean(err))

    rmse = np.asarray(rmse, float)
    bias = np.asarray(bias, float)

    # ---- plot: MSE (RMSE^2) vs noise level ----
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    ax.plot(NOISE_GRID, rmse**2, lw=DEFAULT_LINEWIDTH, color=blue_color, marker=None, label="MSE")
    ax.set_xscale("log"); ax.set_yscale("log")

    ax.set_xlabel(r"noise level $\sigma_{\mathrm{noise}}$", fontsize=DEFAULT_FONTSIZE)
    ax.set_ylabel("MSE", fontsize=DEFAULT_FONTSIZE)
    ax.set_title("Estimator sensitivity: MSE vs noise (default spacing)", fontsize=DEFAULT_FONTSIZE+1)
    ax.legend(frameon=True, loc="best")

    outdir = Path("../demos/plots"); outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "mse_vs_noise.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {outfile}")
    # also print RMSE and bias summaries
    for s, r, b in zip(NOISE_GRID, rmse, bias):
        print(f"sigma={s:.4g}  RMSE={r:.6g}  bias={b:.6g}")

if __name__ == "__main__":
    main()
