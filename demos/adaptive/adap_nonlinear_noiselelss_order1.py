"""Adaptive Fit Derivative Demo — nonlinear, noise-free, order=1."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from derivkit.derivative_kit import DerivativeKit
from common.style import apply_plot_style, DEFAULT_COLORS
from common.formatters import format_value_with_uncertainty
from common.file import resolve_outdir, save_fig

blue = DEFAULT_COLORS["blue"]
red  = DEFAULT_COLORS["red"]

# ----- demo params -----
x0 = 0.30
order = 1

# estimator params (kept explicit; plotting uses global style)
n_points = 30
spacing  = 0.2
base_abs = 1e-2
ridge    = 1e-8

# viz (for smooth curve & sample display only)
n_points_viz = 25
delta_vis = 0.01


def f_clean(x):
    x = np.asarray(x, float)
    return np.sin(6.0 * x) + 0.4 * np.cos(2.0 * x) + 0.2 * x**2 - 0.1 * x

def fprime_true(x):
    x = np.asarray(x, float)
    return 6.0 * np.cos(6.0 * x) - 0.8 * np.sin(2.0 * x) + 0.4 * x - 0.1


def slope_estimator_clean(f_clean_fn) -> float:
    """Estimate slope at x0 using DerivKit on the clean (noise-free) function."""
    dk = DerivativeKit(f_clean_fn, x0)
    val = dk.adaptive.differentiate(
        order,
        n_points=int(n_points),
        spacing=spacing,
        base_abs=base_abs,
        ridge=ridge,
    )
    return float(np.asarray(val).ravel()[0])


def main():
    # one-line global style (sets linewidth, fontsize, markersize, etc.)
    apply_plot_style(base=blue)

    # true slope and adaptive estimate (noise-free)
    true_slope  = float(fprime_true(x0))
    d_est       = slope_estimator_clean(f_clean)
    err_default = abs(d_est - true_slope)

    # sample grid (for display only)
    w_plot = (n_points_viz - 1) * delta_vis
    xx    = np.linspace(x0 - w_plot, x0 + w_plot, 800)
    x_vis = np.linspace(x0 - w_plot, x0 + w_plot, n_points_viz)
    y_vis = f_clean(x_vis)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)

    # true function (dashed red)
    ax.plot(xx, f_clean(xx), "--", color=red, label=r"true $f(x)$")

    # samples (hollow blue)
    ax.scatter(
        x_vis, y_vis,
        facecolor="none", edgecolors=blue,
        label="samples", zorder=3,
    )

    # mark x0 (hollow red)
    ax.scatter(
        [x0], [f_clean(x0)],
        marker="o", facecolor="none", edgecolors=red,
        label=r"$x_0$", zorder=4,
    )

    # tangent line from adaptive derivative (blue)
    ax.plot(
        xx, f_clean(x0) + d_est * (xx - x0),
        color=blue, label=r"adaptive linear fit @ $x_0$",
    )

    ax.set_title("adaptive derivative on a nonlinear function (noise-free)")
    ax.set_xlabel(r"local coordinate $x$")
    ax.set_ylabel(r"function $f\,(x)$")
    ax.legend(frameon=True, loc="lower right")

    # summary box (uses global text sizes)
    info_text = "\n".join([
        r"$f(x)=\sin(6x)+0.4\,\cos(2x)+0.2\,x^{2}-0.1\,x$",
        rf"true slope: $f'(x_0) = {true_slope:.3f}$ at $x_0={x0:.2f}$",
        rf"estimated: $a_\mathrm{{est}} = {format_value_with_uncertainty(d_est, err_default)}$",
    ])
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes, ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    # save & show (consistent with repo utils)
    outdir  = resolve_outdir(None, file=__file__, default_rel="../plots")
    outfile = save_fig(fig, outdir, stem="adaptive_demo_nonlinear_nonoise_order1", ext="png")
    plt.show()
    print(f"saved: {outfile}")
    print(f"true slope @ x0: {true_slope:.6g}")
    print(f"adaptive slope @ x0: {d_est:.6g} (err={err_default:.6g})")


if __name__ == "__main__":
    main()
