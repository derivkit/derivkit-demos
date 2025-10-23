""""Finite-Difference Derivative Demo — linear, noisy, order=1."""

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

# --- demo params ---
x0 = 0.30
true_a, true_b = 1.700, -0.2            # f(x) = a x + b
sigma_noise = 0.01                       # per-evaluation noise std (single draw)

# Finite-difference settings
order = 1
num_points = 5                           # 3,5,7,9 supported (per your class)
h = 0.1                                  # stepsize

# For the smooth red curve only (NOT a sample grid)
plot_halfwidth = 0.24
n_curve = 800

def f_clean(x, a=true_a, b=true_b):
    x = np.asarray(x, float)
    return a * x + b

def main():
    apply_plot_style(base=blue)

    rng = random_generator(42)

    # Single-draw noisy evaluator for FD calls (no replicates)
    def f_eval(x):
        return f_clean(x) + rng.normal(0.0, sigma_noise)

    # Derivative via DerivativeKit's finite difference
    dk = DerivativeKit(f_eval, x0)
    d_est = float(np.asarray(
        dk.finite.differentiate(order=order, stepsize=h, num_points=num_points)
    ).ravel()[0])
    err = abs(d_est - true_a)

    # --- get stencil nodes from the finite-diff class (no hard-coded map) ---
    offsets, _ = dk.finite.get_finite_difference_tables(
        h)  # dicts: offsets[num_points], coeffs_table[(num_points, order)]
    x_nodes = x0 + np.array(offsets[num_points], float) * h

    # One noisy observation at x0 for anchoring the tangent visually
    fx0_noisy = float(add_gaussian_noise(f_clean(x0), sigma_noise, rng))

    # Smooth curve for the true function (purely visual; not a sampling grid)
    xx = np.linspace(x0 - plot_halfwidth, x0 + plot_halfwidth, n_curve)

    # ---- figure ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)

    # true function (dashed red)
    ax.plot(xx, f_clean(xx), "--", color=red, label=r"true $f(x)$")

    # stencil nodes only (hollow blue); single noisy draw for display
    y_nodes = f_clean(x_nodes) + rng.normal(0.0, sigma_noise, size=x_nodes.shape)
    ax.scatter(x_nodes, y_nodes, facecolor="none", edgecolors=blue,
               label=fr"FD stencil ({num_points}-point)", zorder=3)

    # mark x0 (noisy obs, hollow red)
    ax.scatter([x0], [fx0_noisy], facecolor="none", edgecolors=red, label=r"$x_0$", zorder=4)

    # tangent line using FD estimate
    ax.plot(xx, fx0_noisy + d_est * (xx - x0), color=blue, label=r"finite-diff fit @ $x_0$")

    ax.set_title("finite difference on a noisy linear function")
    ax.set_xlabel(r"local coordinate $x$")
    ax.set_ylabel(r"$f(x)$")
    ax.legend(frameon=True, loc="lower right")

    # summary box
    info = "\n".join([
        rf"$f(x)=a\,x+b \;=\; {true_a:.3f}\,x{true_b:+.3f}$",
        rf"true slope: $a = {true_a:.3f}$",
        rf"estimated (FD): $a_\mathrm{{est}} = {format_value_with_uncertainty(d_est, err)}$",
        rf"(stencil={num_points}-point, h={h:.3f})",
    ])
    ax.text(0.02, 0.98, info, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"))

    # save & show
    outdir  = resolve_outdir(None, file=__file__, default_rel="../plots")
    outfile = save_fig(fig, outdir, stem="fd_demo_linear_noisy_order1", ext="png")
    plt.show()
    print(f"saved: {outfile}")
    print(f"true slope: {true_a:.6g}")
    print(f"FD slope @ x0: {d_est:.6g} (err={err:.6g})")

if __name__ == "__main__":
    main()
