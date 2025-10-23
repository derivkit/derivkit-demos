#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- DerivKit ----
from derivkit.derivative_kit import DerivativeKit

# ---- style (no fallback) ----
from common.style import apply_plot_style, _DEFAULT_LINEWIDTH, _DEFAULT_FONTSIZE, _DEFAULT_COLORS
from scripts.plot_style import DEFAULT_LINEWIDTH

# --- palette aliases ---
blue_color = _DEFAULT_COLORS["blue"]
red_color  = _DEFAULT_COLORS["red"]
DEFAULT_LINEWIDTH = _DEFAULT_LINEWIDTH
DEFAULT_FONTSIZE = _DEFAULT_FONTSIZE

# =========================
# Shared defaults
# =========================
X0 = 0.30
ORDER = 1
TRUE_A = 1.700
TRUE_B = -0.2
SIGMA_NOISE = 0.01

# Estimator defaults
N_POINTS = 25
SPACING = 0.25
BASE_ABS = 1e-3
RIDGE = 1e-8

# MC defaults (kept for parity with original file, unused here)
MC_DRAWS = 500
MC_SEED = 42
MC_SIGMA = 0.02

# Viz defaults (kept for parity with original file, unused here)
N_POINTS_VIZ = 25
DELTA_VIS = 0.01


# ---------- model + noise ----------
def f_clean(x, a=TRUE_A, b=TRUE_B):
    x = np.asarray(x, float)
    return a * x + b

def make_noisy(f, sigma=SIGMA_NOISE, seed=42):
    rng = np.random.default_rng(seed)
    def g(x):
        x = np.asarray(x, float)
        return f(x) + rng.normal(0.0, sigma, size=x.shape)
    return g


# ---------- derivative helper ----------
def slope_default(
    f,
    x0,
    order=ORDER,
    *,
    n_points=N_POINTS,
    spacing="auto",            # <<< use default (no user-defined grid)
    base_abs=BASE_ABS,
    ridge=RIDGE,
):
    dk = DerivativeKit(f, x0)
    val = dk.adaptive.differentiate(
        order,
        n_points=int(n_points),
        spacing=spacing,        # <<< "auto"
        base_abs=base_abs,
        ridge=ridge,
    )
    return float(np.asarray(val).ravel()[0])


def main():
    apply_plot_style(base=blue_color)

    # same noisy target used in the original script
    f = make_noisy(lambda x: f_clean(x, a=TRUE_A, b=TRUE_B), sigma=SIGMA_NOISE, seed=42)

    # ---------- STABILITY: \hat d vs spacing (same plot structure; internal spacing is auto) ----------
    sp_grid = np.linspace(max(0.05, SPACING*0.4), SPACING*1.8, 12)
    d_grid = []
    for _sp in sp_grid:
        d_grid.append(
            slope_default(
                f, X0, order=ORDER, n_points=N_POINTS,
                spacing="auto",               # <<< force default spacing
                base_abs=BASE_ABS, ridge=RIDGE
            )
        )
    d_grid = np.asarray(d_grid, float)

    # figure/axes (standalone)
    fig, ax_stab = plt.subplots(1, 1, constrained_layout=True)

    ax_stab.plot(sp_grid, d_grid, marker=None, lw=DEFAULT_LINEWIDTH, color=blue_color, label=r"$\hat d$")
    ax_stab.axhline(TRUE_A, ls="--", lw=DEFAULT_LINEWIDTH, color=red_color, label="true $a$")
    ax_stab.axvline(SPACING, ls=":", lw=DEFAULT_LINEWIDTH, color=blue_color, label="chosen spacing")

    ax_stab.set_xlabel("spacing", fontsize=DEFAULT_FONTSIZE-5)
    ax_stab.set_ylabel(r"$\hat d$", fontsize=DEFAULT_FONTSIZE-5)
    ax_stab.tick_params(axis="both", labelsize=8)
    ax_stab.set_title("Stability vs spacing", fontsize=DEFAULT_FONTSIZE-5, pad=4.0)
    ax_stab.legend(fontsize=6, frameon=True, loc="best")

    # save & show
    outdir = Path("../demos/plots"); outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "stability_vs_spacing.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
