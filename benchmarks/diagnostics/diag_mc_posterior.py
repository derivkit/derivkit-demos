"""Adaptive Fit Derivative Demo — MC standalone"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from derivkit.derivative_kit import DerivativeKit
from common.style import (
    apply_plot_style,
    _DEFAULT_COLORS,
    _DEFAULT_FONTSIZE,
    _DEFAULT_LINEWIDTH,
    _DEFAULT_MARKERSIZE,
)

# utils
from common.formatters import format_value_with_uncertainty
from common.monte_carlo import tiny_mcmc_slope, tiny_mcmc_draws_2d, hpd_density_levels
from common.utils import add_gaussian_noise, random_generator

# palette aliases
blue_color = _DEFAULT_COLORS["blue"]
red_color = _DEFAULT_COLORS["red"]

# defaults
x0 = 0.30
order = 1
true_a = 1.700
true_b = -0.2
sigma_noise = 0.01

# estimator defaults
n_points = 25
spacing = 0.25
base_abs = 1e-3
ridge = 1e-8

# mc defaults
mc_draws = 500
mc_seed = 42
mc_sigma = 0.02

# viz defaults
n_points_viz = 25
delta_vis = 0.01


# ---------- model (clean) ----------
def f_clean(x, a=true_a, b=true_b):
    """simple linear model."""
    x = np.asarray(x, float)
    return a * x + b


def slope_estimator(f_clean_fn, sigma: float, rng: np.random.Generator) -> float:
    """Estimate slope at x0 using derivkit, adding gaussian noise to evaluations.

    The estimator gets the clean function, a noise level, and an rng.
    It defines a tiny local callable that adds noise to outputs (required for derivkit),
    then runs the adaptive derivative at x0.
    """
    def g(xx):
        yy = f_clean_fn(xx)
        return add_gaussian_noise(yy, sigma, rng)

    dk = DerivativeKit(g, x0)
    val = dk.adaptive.differentiate(
        order,
        n_points=int(n_points),
        spacing=spacing,
        base_abs=base_abs,
        ridge=ridge,
    )
    return float(np.asarray(val).ravel()[0])


def main():
    apply_plot_style(base=blue_color)

    # replicate the same RNG and single-noise draw used in the demo for d_default and f_noisy(x0)
    rng_main = random_generator(42)

    def f_noisy(xx):
        xx = np.asarray(xx, float)
        return f_clean(xx) + rng_main.normal(0.0, sigma_noise, size=xx.shape)

    # derivative + simple absolute error vs truth (same as demo)
    d_default = slope_estimator(f_clean, sigma_noise, rng_main)
    err_default = abs(d_default - true_a)

    # monte carlo stats + draws (exact same API and params as inset)
    mc_mean, mc_std = tiny_mcmc_slope(
        noiseless_function=f_clean,
        estimator=slope_estimator,
        draws=mc_draws,
        seed=mc_seed,
        sigma=mc_sigma,
    )
    d_draws, y0_draws = tiny_mcmc_draws_2d(
        x0=x0,
        f_clean=f_clean,
        estimator=slope_estimator,
        draws=mc_draws,
        seed=mc_seed,
        sigma=mc_sigma,
    )

    # build the same 2D histogram grid
    bins_d = max(20, mc_draws // 12)
    bins_y = max(20, mc_draws // 12)
    h, d_edges, y_edges = np.histogram2d(
        d_draws, y0_draws, bins=[bins_d, bins_y], density=True
    )

    # optional smoothing (identical behavior)
    try:
        from scipy.ndimage import gaussian_filter
        h = gaussian_filter(h, sigma=1.0, mode="nearest")
    except Exception:
        pass

    # HPD levels: return sorted levels (68%, 95%) — same helper function
    raw_levels = hpd_density_levels(h, d_edges, y_edges, masses=(0.68, 0.95))
    levels = np.asarray(raw_levels, float)
    levels.sort()

    d_centers = 0.5 * (d_edges[:-1] + d_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    d, y = np.meshgrid(d_centers, y_centers, indexing="ij")

    # ---- Standalone figure (same visuals as inset) ----
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    # HPD contours with the same styles and ordering
    ax.contour(
        d,
        y,
        h,
        levels=levels,
        colors=[blue_color, blue_color],
        linewidths=[_DEFAULT_LINEWIDTH, _DEFAULT_LINEWIDTH],
        linestyles=["-.", ":"],  # 95%, 68%
    )

    # same reference lines/points as inset
    h_true = ax.axvline(
        true_a,
        ls="--",
        lw=_DEFAULT_LINEWIDTH,
        color=blue_color,
        label=fr"$a={true_a:.3f}$ (true)",
    )
    h_est = ax.scatter(
        [d_default],
        [f_noisy(x0)],
        s=28,
        marker="x",
        edgecolors=blue_color,
        facecolor="none",
        label=fr"$\hat d={d_default:.3f}$",
    )

    # identical labels and tiny font tweaks
    ax.set_xlabel("slope $d$", fontsize=_DEFAULT_FONTSIZE - 5)
    ax.set_ylabel("$f\\,(x_0)$", fontsize=_DEFAULT_FONTSIZE - 5)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(
        f"mc ($n={mc_draws}$, seed={mc_seed}): slope {format_value_with_uncertainty(mc_mean, mc_std)}",
        fontsize=_DEFAULT_FONTSIZE - 5,
        pad=4.5,
    )

    # legend proxies & layout—kept exactly
    proxies = [
        Line2D(
            [],
            [],
            color=blue_color,
            lw=_DEFAULT_LINEWIDTH,
            linestyle="-.",
            label="$2\\,\\sigma$",
        ),
        Line2D(
            [],
            [],
            color=blue_color,
            lw=_DEFAULT_LINEWIDTH,
            linestyle=":",
            label="$1\\,\\sigma$",
        ),
    ]
    ax.legend(
        handles=proxies + [h_true, h_est],
        loc="lower left",
        fontsize=6,
        frameon=True,
        handlelength=2.8,
        ncol=2,
    )

    # save & show (mirrors your save pattern)
    outdir = Path("../demos/plots")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "adaptive_fit_mc.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"saved: {outfile}")
    print(
        f"mc (seed {mc_seed}, n={mc_draws}): "
        f"{format_value_with_uncertainty(mc_mean, mc_std)}"
    )
    print(
        f"reference (from single noisy eval): "
        f"d_default={d_default:.6g}, f_noisy(x0)={f_noisy(x0):.6g}"
    )


if __name__ == "__main__":
    main()
