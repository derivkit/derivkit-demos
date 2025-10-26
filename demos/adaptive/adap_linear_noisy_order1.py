"""Adaptive Fit Derivative Demo — linear, noisy, order=1."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common.style import apply_plot_style, DEFAULT_COLORS
from common.formatters import format_value_with_uncertainty
from common.noise import random_generator, make_noisy_function
from common.file import resolve_outdir, save_fig
from adap_common import f_clean, slope_estimator

# palette
blue_color = DEFAULT_COLORS["blue"]
red_color  = DEFAULT_COLORS["red"]

# model / demo params
x0 = 0.30
order = 1
true_a = 1.700
true_b = -0.2
sigma_noise = 0.05

# estimator
n_points = 27
spacing = 5
base_abs = 1e-3
ridge = 1e-8
rng_main = random_generator(42)

# viz (for smooth curve & sample display only)
n_points_viz = 25
delta_vis = 0.01

f_noisy = make_noisy_function(f_clean, sigma_noise, rng_main, f_args=(true_a, true_b))

def main():
    # one-line global style; sets linewidth/fontsize/markersize/etc.
    apply_plot_style(base=blue_color)
    rng_main = random_generator(42)

    def f_noisy(xx):
        xx = np.asarray(xx, float)
        return f_clean(xx, true_a, true_b) + rng_main.normal(0.0, sigma_noise, size=xx.shape)

    # derivative + abs error
    d_default = slope_estimator(
        f_clean,
        x0=x0,
        order=order,
        n_points=n_points,
        spacing=spacing,
        base_abs=base_abs,
        ridge=ridge,
        sigma=sigma_noise,
        rng=rng_main,
        f_args=(true_a, true_b),
    )

    err_default = abs(d_default - true_a)

    # sample grid for plotting (not the adaptive grid)
    # span around x0 for visualization only
    plot_half_span = (n_points_viz - 1) * delta_vis

    # a dense grid for drawing smooth curves
    x_dense = np.linspace(x0 - plot_half_span, x0 + plot_half_span, 800)

    # a sparse set of sample points to display as markers
    x_samples = np.linspace(x0 - plot_half_span, x0 + plot_half_span, n_points_viz)
    y_samples = f_noisy(x_samples)

    # find and omit the point closest to x0 (keeps x0 highlighted separately)
    x0_nearest_idx = int(np.argmin(np.abs(x_samples - x0)))
    x_samples_no_x0 = np.delete(x_samples, x0_nearest_idx)
    y_samples_no_x0 = np.delete(y_samples, x0_nearest_idx)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)

    # true function (dashed, red)
    ax.plot(x_dense, f_clean(x_dense, true_a, true_b), "--", color=red_color, label=r"true $f(x)$")

    # noisy samples (hollow blue)
    ax.scatter(
        x_samples_no_x0, y_samples_no_x0,
        facecolor="none", edgecolors=blue_color,
        label="noisy samples", zorder=3,
    )

    # mark x0 (hollow red)
    ax.scatter(
        [x0], [f_noisy(x0)],
        marker="o", facecolor="none", edgecolors=red_color,
        label=r"evaluation point $x_0$", zorder=4,
    )

    # tangent line from adaptive derivative (blue)
    ax.plot(
        x_dense, f_noisy(x0) + d_default * (x_dense - x0),
        color=blue_color, label=r"adaptive fit at $x_0$",
    )

    ax.set_title(r"adaptive derivative on a noisy linear model")
    ax.set_xlabel(r"local coordinate $x$")
    ax.set_ylabel(r"observed value $y$")
    ax.legend(frameon=True, loc="lower right")

    # info box
    txt = rf"""
    data model
      $y=f(x)+\varepsilon,\ \ \varepsilon\sim\mathcal{{N}}(0,\sigma^2)$
      $f(x)=a x + b$
      $a={true_a:6.3f},\ \ b={true_b:+6.3f},\ \ \sigma={sigma_noise:5.3g}$

    estimate
      $\widehat{{a}}_\mathrm{{AF}}={format_value_with_uncertainty(d_default, err_default)}$
    """
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top", fontsize=12.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0, edgecolor="none"))

    # save & show
    outdir = resolve_outdir(None, file=__file__, default_rel="../plots")
    outfile = save_fig(fig, outdir, stem="adaptive_demo_linear_noisy_order1", ext="png")
    plt.show()

    # print results
    print(f"saved: {outfile}")
    print(f"true slope: {true_a:.6g}")
    print(f"adaptive slope @ x0: {d_default:.6g} (err={err_default:.6g})")

if __name__ == "__main__":
    main()
