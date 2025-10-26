"""Adaptive Fit Derivative Demo — nonlinear, noisy, order=2."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from derivkit.derivative_kit import DerivativeKit
from common.style import apply_plot_style, DEFAULT_COLORS
from common.formatters import format_value_with_uncertainty
from common.noise import add_gaussian_noise, random_generator
from common.file import resolve_outdir, save_fig

blue = DEFAULT_COLORS["blue"]
red  = DEFAULT_COLORS["red"]

# --- demo params ---
x0 = 0.30
sigma_noise = 0.05          # per-evaluation Gaussian noise
replicates = 15              # average this many noisy evals per node (variance ↓ by ~1/replicates)

# --- robust adaptive settings (small bump over defaults) ---
# Widen the half-width, add a few nodes, keep tiny ridge for stability.
ORDER = 2
N_POINTS = 25
SPACING  = 0.25
BASE_ABS = 1e-3
RIDGE    = 1e-8

# viz range (NOT the adaptive grid)
n_points_viz, delta_vis = 25, 0.01

# --- nonlinear target with analytic derivatives (for ground truth) ---
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
    # Global style only (linewidth, fontsize, markersize, etc.)
    apply_plot_style(base=blue)

    rng = random_generator(42)

    # Average R noisy evals per node to tame variance
    def f_noisy_avg(xx):
        xx = np.asarray(xx, float)
        acc = np.zeros_like(xx, dtype=float)
        for _ in range(replicates):
            acc += add_gaussian_noise(f_clean(xx), sigma_noise, rng)
        return acc / float(replicates)

    # DerivativeKit on averaged-noisy function at x0 (robust settings)
    dk = DerivativeKit(f_noisy_avg, x0)
    d2_est = float(np.asarray(dk.adaptive.differentiate(
        ORDER,
        n_points=N_POINTS,
        spacing=SPACING,
        base_abs=BASE_ABS,
        ridge=RIDGE,
    )).ravel()[0])

    # truths & error
    d2_true = float(f2_true(x0))
    err2    = abs(d2_est - d2_true)

    # display grid (cosmetic)
    w_plot = (n_points_viz - 1) * delta_vis
    xx    = np.linspace(x0 - w_plot, x0 + w_plot, 800)
    x_vis = np.linspace(x0 - w_plot, x0 + w_plot, n_points_viz)

    # show noisy samples (single noisy draw per point for display)
    y_vis = add_gaussian_noise(f_clean(x_vis), sigma_noise, rng)

    # local quadratic using estimated f'' and *estimated* f'(x0) for realism
    # (we also estimate f'(x0) with the same stabilized settings)
    d1_est = float(np.asarray(dk.adaptive.differentiate(
        1,
        n_points=N_POINTS,
        spacing=SPACING,
        base_abs=BASE_ABS,
        ridge=RIDGE,
    )).ravel()[0])

    # anchor at a single (noisy) observation at x0 for visual consistency
    f0_noisy = float(add_gaussian_noise(f_clean(x0), sigma_noise, rng))
    quad = f0_noisy + d1_est * (xx - x0) + 0.5 * d2_est * (xx - x0) ** 2

    # ---- plot ----
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)

    # true function (dashed red)
    ax.plot(xx, f_clean(xx), "--", color=red, label=r"true $f(x)$")

    # noisy samples (hollow blue)
    ax.scatter(x_vis, y_vis, facecolor="none", edgecolors=blue, label="samples", zorder=3)

    # mark x0 (noisy observation, hollow red)
    ax.scatter([x0], [f0_noisy], facecolor="none", edgecolors=red, label=r"$x_0$", zorder=4)

    # quadratic approximation using stabilized estimates
    ax.plot(xx, quad, color=blue, label=rf"quadratic approx @ $x_0$ (uses $\hat f', \hat f''$)")

    ax.set_title("adaptive second derivative on a nonlinear function (noisy, stabilized)")
    ax.set_xlabel(r"local coordinate $x$")
    ax.set_ylabel(r"$f(x)$")
    ax.legend(frameon=True, loc="lower right")

    # summary box (focus on f'')
    info = "\n".join([
        r"$f(x)=\sin(6x)+0.4\,\cos(2x)+0.2\,x^{2}-0.1\,x$",
        rf"true $f''(x_0) = {d2_true:.3f}$ at $x_0={x0:.2f}$",
        rf"estimated $f''(x_0) = {format_value_with_uncertainty(d2_est, err2)}$",
        # rf"(n_points={N_POINTS}, spacing={SPACING:.2f}, ridge={RIDGE:.0e}, reps={replicates})",
    ])
    ax.text(0., -0.2, info, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor=blue))

    # save & print
    outdir  = resolve_outdir(None, file=__file__, default_rel="../plots")
    outfile = save_fig(fig, outdir, stem="adaptive_demo_nonlinear_noisy_order2", ext="png")
    plt.show()
    print(f"saved: {outfile}")
    print(f"true f''(x0): {d2_true:.6g}")
    print(f"adaptive f''(x0): {d2_est:.6g} (err={err2:.6g})")

if __name__ == "__main__":
    main()
