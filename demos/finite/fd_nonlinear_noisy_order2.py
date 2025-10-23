#!/usr/bin/env python3
""""Finite-Difference Derivative Demo — nonlinear, noisy, order=2 (stencil-only, no replicates)."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from derivkit.derivative_kit import DerivativeKit
from common.style import apply_plot_style, DEFAULT_COLORS
from common.formatters import format_value_with_uncertainty
from common.utils import add_gaussian_noise, random_generator, resolve_outdir, save_fig

# palette
blue = DEFAULT_COLORS["blue"]
red  = DEFAULT_COLORS["red"]

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

# --- demo params ---
x0 = 0.30
sigma_noise = 0.05               # per-evaluation noise std (single draw)

# Finite-difference settings
order = 2
num_points = 5                   # (5,2) is supported by your FD class
h = 0.1                         # larger step reduces noise amplification ~ 1/h^2

# For the smooth red curve only (NOT a sampling grid)
plot_halfwidth = 0.24
n_curve = 800

def main():
    apply_plot_style(base=blue)
    rng = random_generator(42)

    # Single-draw noisy evaluator for FD calls (no replicates)
    def f_eval(x):
        return f_clean(x) + rng.normal(0.0, sigma_noise)

    # ---- Derivatives via DerivativeKit's finite difference ----
    dk = DerivativeKit(f_eval, x0)

    # 2nd derivative estimate (target of this demo)
    d2_est = float(np.asarray(
        dk.finite.differentiate(order=2, stepsize=h, num_points=num_points)
    ).ravel()[0])

    # Also estimate the first derivative to build a local quadratic without using ground truth
    d1_est = float(np.asarray(
        dk.finite.differentiate(order=1, stepsize=h, num_points=num_points)
    ).ravel()[0])

    # Ground truths at x0
    true_d1 = float(fprime_true(x0))
    true_d2 = float(f2_true(x0))

    # Simple absolute errors (shown as ± using format_value_with_uncertainty)
    err1 = abs(d1_est - true_d1)
    err2 = abs(d2_est - true_d2)

    # Get the actual stencil nodes your FD used (for the chosen stepsize)
    offsets, _ = dk.finite.get_finite_difference_tables(h)  # dicts
    x_nodes = x0 + np.array(offsets[num_points], float) * h

    # One noisy observation at x0 for visual anchoring
    fx0_noisy = float(add_gaussian_noise(f_clean(x0), sigma_noise, rng))

    # Smooth curve for the true function (visual only)
    xx = np.linspace(x0 - plot_halfwidth, x0 + plot_halfwidth, n_curve)

    # ---- figure ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)

    # true function (solid red)
    ax.plot(xx, f_clean(xx), "-", color=red, label=r"true $f(x)$")

    # stencil nodes only (hollow blue); single noisy draw for display
    y_nodes = f_clean(x_nodes) + rng.normal(0.0, sigma_noise, size=x_nodes.shape)
    ax.scatter(x_nodes, y_nodes, facecolor="none", edgecolors=blue,
               label=fr"FD stencil ({num_points}-point)", zorder=3)

    # mark x0 (noisy obs, hollow red)
    ax.scatter([x0], [fx0_noisy], facecolor="none", edgecolors=red, label=r"$x_0$", zorder=4)

    # Local quadratic around x0 using the *estimated* derivatives (self-contained)
    quad = fx0_noisy + d1_est * (xx - x0) + 0.5 * d2_est * (xx - x0) ** 2
    ax.plot(xx, quad, color=blue, label=rf"quadratic approx @ $x_0$ (uses $\hat f', \hat f''$)")

    ax.set_title("finite difference on a noisy nonlinear function — second derivative")
    ax.set_xlabel(r"local coordinate $x$")
    ax.set_ylabel(r"$f(x)$")
    ax.legend(frameon=True, loc="lower right")

    # summary box
    info = "\n".join([
        rf"true $f'(x_0) = {true_d1:.3f}$,  estimated $f'(x_0) = {format_value_with_uncertainty(d1_est, err1)}$",
        rf"true $f''(x_0) = {true_d2:.3f}$, estimated $f''(x_0) = {format_value_with_uncertainty(d2_est, err2)}$",
        rf"(stencil={num_points}-point, h={h:.2f}, $\sigma_\mathrm{{noise}}={sigma_noise:.3f}$)",
    ])
    ax.text(0.02, 0.98, info, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"))

    # save & show
    outdir  = resolve_outdir(None, file=__file__, default_rel="../plots")
    outfile = save_fig(fig, outdir, stem="fd_demo_nonlinear_noisy_order2", ext="png")
    plt.show()
    print(f"saved: {outfile}")
    print(f"true f'(x0): {true_d1:.6g},  FD f'(x0): {d1_est:.6g} (err={err1:.6g})")
    print(f"true f''(x0): {true_d2:.6g}, FD f''(x0): {d2_est:.6g} (err={err2:.6g})")

if __name__ == "__main__":
    main()
