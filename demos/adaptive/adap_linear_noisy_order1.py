"""Adaptive Fit Derivative Demo — linear, noisy, order=1."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from derivkit.derivative_kit import DerivativeKit
from common.style import apply_plot_style, DEFAULT_COLORS
from common.formatters import format_value_with_uncertainty
from common.utils import add_gaussian_noise, random_generator, resolve_outdir, save_fig

# palette
blue_color = DEFAULT_COLORS["blue"]
red_color  = DEFAULT_COLORS["red"]

# model / demo params
x0 = 0.30
order = 1
true_a = 1.700
true_b = -0.2
sigma_noise = 0.01

# estimator (keep as-is; plotting is what we’re simplifying)
n_points = 25
spacing  = 0.25
base_abs = 1e-3
ridge    = 1e-8

# viz (for smooth curve & sample display only)
n_points_viz = 25
delta_vis    = 0.01


# ---------- model (clean) ----------
def f_clean(x, a=true_a, b=true_b):
    x = np.asarray(x, float)
    return a * x + b


def slope_estimator(f_clean_fn, sigma: float, rng: np.random.Generator) -> float:
    """Estimate slope at x0 using derivkit, adding gaussian noise to evaluations."""
    def g(xx):
        return add_gaussian_noise(f_clean_fn(xx), sigma, rng)

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
    # one-line global style; sets linewidth/fontsize/markersize/etc.
    apply_plot_style(base=blue_color)

    rng_main = random_generator(42)

    def f_noisy(xx):
        xx = np.asarray(xx, float)
        return f_clean(xx) + rng_main.normal(0.0, sigma_noise, size=xx.shape)

    # derivative + abs error
    d_default = slope_estimator(f_clean, sigma_noise, rng_main)
    err_default = abs(d_default - true_a)

    # sample grid for plotting (not the adaptive grid)
    w_plot = (n_points_viz - 1) * delta_vis
    xx = np.linspace(x0 - w_plot, x0 + w_plot, 800)
    x_vis = np.linspace(x0 - w_plot, x0 + w_plot, n_points_viz)
    y_vis = f_noisy(x_vis)
    idx0 = int(np.argmin(np.abs(x_vis - x0)))
    x_vis_circles = np.delete(x_vis, idx0)
    y_vis_circles = np.delete(y_vis, idx0)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)

    # true function (dashed, red)
    ax.plot(xx, f_clean(xx), "--", color=red_color, label=r"true $f(x)$")

    # noisy samples (hollow blue)
    ax.scatter(
        x_vis_circles, y_vis_circles,
        facecolor="none", edgecolors=blue_color,
        label="samples", zorder=3,
    )

    # mark x0 (hollow red)
    ax.scatter(
        [x0], [f_noisy(x0)],
        marker="o", facecolor="none", edgecolors=red_color,
        label=r"evaluation point $x_0$", zorder=4,
    )

    # tangent line from adaptive derivative (blue)
    ax.plot(
        xx, f_noisy(x0) + d_default * (xx - x0),
        color=blue_color, label=r"adaptive linear fit @ $x_0$",
    )

    ax.set_title("adaptive derivative on a noisy linear function")
    ax.set_xlabel(r"local coordinate $x$")
    ax.set_ylabel(r"function $f\,(x)$")
    ax.legend(frameon=True, loc="lower right")

    # summary box (no explicit fontsize; uses global)
    info_text = "\n".join([
        rf"$f(x)=a\,x+b \;=\; {true_a:.3f}\,x{true_b:+.3f}$",
        rf"true slope: $a = {true_a:.3f}$",
        rf"estimated: $a_\mathrm{{est}} = {format_value_with_uncertainty(d_default, err_default)}$",
    ])
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes, ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )

    # save & show
    outdir = resolve_outdir(None, file=__file__, default_rel="../plots")
    outfile = save_fig(fig, outdir, stem="adaptive_demo_linear_noisy_order1", ext="png")
    plt.show()
    print(f"saved: {outfile}")
    print(f"true slope: {true_a:.6g}")
    print(f"adaptive slope @ x0: {d_default:.6g} (err={err_default:.6g})")


if __name__ == "__main__":
    main()
