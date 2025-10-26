"""Adaptive Fit Derivative Demo — nonlinear, noise-free, order=2."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from derivkit.derivative_kit import DerivativeKit
from common.style import apply_plot_style, DEFAULT_COLORS
from common.formatters import format_value_with_uncertainty
from common.file import resolve_outdir, save_fig

blue = DEFAULT_COLORS["blue"]
red  = DEFAULT_COLORS["red"]

# Demo params
x0, order = 0.30, 2   # second derivative

# Visualization range (for smooth lines; NOT the adaptive grid)
n_points_viz, delta_vis = 25, 0.01

# --- Nonlinear target (analytic derivatives available) ---
def f_clean(x):
    x = np.asarray(x, float)
    return np.sin(6.0 * x) + 0.4 * np.cos(2.0 * x) + 0.2 * x**2 - 0.1 * x

def fprime_true(x):
    x = np.asarray(x, float)
    return 6.0 * np.cos(6.0 * x) - 0.8 * np.sin(2.0 * x) + 0.4 * x - 0.1

def f2_true(x):
    x = np.asarray(x, float)
    return -36.0 * np.sin(6.0 * x) - 1.6 * np.cos(2.0 * x) + 0.4

def main():
    apply_plot_style(base=blue)

    dk = DerivativeKit(f_clean, x0)

    # Estimate f''(x0) with defaults only
    d2_est = float(np.asarray(dk.adaptive.differentiate(order)).ravel()[0])
    true_d2 = float(f2_true(x0))
    err2 = abs(d2_est - true_d2)

    # Build a local quadratic using known f and f' plus estimated f''
    w_plot = (n_points_viz - 1) * delta_vis
    xx = np.linspace(x0 - w_plot, x0 + w_plot, 800)
    x_vis = np.linspace(x0 - w_plot, x0 + w_plot, n_points_viz)
    y_vis = f_clean(x_vis)

    f0 = f_clean(x0)
    f1 = fprime_true(x0)
    quad = f0 + f1 * (xx - x0) + 0.5 * d2_est * (xx - x0) ** 2

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)

    # True function (dashed red)
    ax.plot(xx, f_clean(xx), "--", color=red, label=r"true $f(x)$")

    # Samples (hollow blue, display only)
    ax.scatter(x_vis, y_vis, facecolor="none", edgecolors=blue, label="samples", zorder=3)

    # Mark x0 (hollow red)
    ax.scatter([x0], [f0], facecolor="none", edgecolors=red, label=r"$x_0$", zorder=4)

    # Quadratic approx using estimated f''(x0)
    ax.plot(xx, quad, color=blue, label=rf"quadratic approx @ $x_0$ (uses $\hat f''$)")

    ax.set_title("adaptive second derivative on a nonlinear function (noise-free)")
    ax.set_xlabel(r"local coordinate $x$")
    ax.set_ylabel(r"$f(x)$")
    ax.legend(frameon=True, loc="lower right")

    # Summary box
    info = "\n".join([
        r"$f(x)=\sin(6x)+0.4\,\cos(2x)+0.2\,x^{2}-0.1\,x$",
        rf"true $f''(x_0) = {true_d2:.3f}$ at $x_0={x0:.2f}$",
        rf"estimated $f''(x_0) = {format_value_with_uncertainty(d2_est, err2)}$",
    ])
    ax.text(0.02, 0.58, info, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"))

    outdir  = resolve_outdir(None, file=__file__, default_rel="../plots")
    outfile = save_fig(fig, outdir, stem="adaptive_demo_nonlinear_nonoise_order2", ext="png")
    plt.show()
    print(f"saved: {outfile}")
    print(f"true f''(x0): {true_d2:.6g}")
    print(f"adaptive f''(x0): {d2_est:.6g} (err={err2:.6g})")

if __name__ == "__main__":
    main()
