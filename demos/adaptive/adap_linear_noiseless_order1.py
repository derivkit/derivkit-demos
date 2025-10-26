"""Adaptive Fit Derivative Demo — linear, noise-free, order=1."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from derivkit.derivative_kit import DerivativeKit
from common.style import apply_plot_style, DEFAULT_COLORS
from common.formatters import format_value_with_uncertainty
from common.file import resolve_outdir, save_fig
from adap_common import f_clean

blue = DEFAULT_COLORS["blue"]
red  = DEFAULT_COLORS["red"]

x0, order = 0.30, 1
true_a, true_b = 1.700, -0.2

# viz range (just for the smooth line)
n_points_viz, delta_vis = 25, 0.01


def main():
    apply_plot_style(base=blue)  # <- sets linewidth/fontsize/markersize/etc.

    dk = DerivativeKit(f_clean, x0)
    d_est = float(np.asarray(dk.adaptive.differentiate(order)).ravel()[0])
    err = abs(d_est - true_a)

    # cosmetic grid for the curve (NOT the adaptive grid)
    w_plot = (n_points_viz - 1) * delta_vis
    xx = np.linspace(x0 - w_plot, x0 + w_plot, 800)
    x_vis = np.linspace(x0 - w_plot, x0 + w_plot, n_points_viz)
    y_vis = f_clean(x_vis, true_a, true_b)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)

    ax.plot(xx, f_clean(xx, true_a, true_b), "--", color=red, label=r"true $f(x)$")

    # These are the hollow scatter points to represent samples
    ax.scatter(x_vis, y_vis, facecolor="none", edgecolors=blue, label="samples", zorder=3)

    # Here I highlight x0
    ax.scatter([x0], [f_clean(x0, true_a, true_b)], facecolor="none", edgecolors=red, label=r"$x_0$", zorder=4)

    # This is tangent line
    ax.plot(xx, f_clean(x0, true_a, true_b) + d_est * (xx - x0), color=blue, label=r"adaptive fit @ $x_0$")

    ax.set_title("adaptive derivative on a linear function (noise-free)")
    ax.set_xlabel(r"local coordinate $x$")
    ax.set_ylabel(r"$f(x)$")
    ax.legend(frameon=True, loc="lower right")

    # This is the summary box to be printed in the plot as text
    info = "\n".join([
        rf"$f(x)=a\,x+b \;=\; {true_a:.3f}\,x{true_b:+.3f}$",
        rf"true slope: $a = {true_a:.3f}$",
        rf"estimated: $a_\mathrm{{est}} = {format_value_with_uncertainty(d_est, err)}$",
    ])
    ax.text(0.02, 0.98, info, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"))

    outdir = resolve_outdir(None, file=__file__, default_rel="../plots")
    stem = "adaptive_demo_linear_nonoise_order1"
    outfile = save_fig(fig, outdir, stem=stem, ext="png")
    plt.show()
    print(f"saved: {outfile}")
    print(f"true slope: {true_a:.6g}")
    print(f"adaptive slope @ x0: {d_est:.6g} (err={err:.6g})")

if __name__ == "__main__":
    main()
