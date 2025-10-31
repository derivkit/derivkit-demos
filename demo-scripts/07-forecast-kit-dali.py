"""DerivKit — ForecastKit DALI (1D) Demo.

This demo compares the *exact* Gaussian log-likelihood to two local
approximations built from ForecastKit tensors:
  • Fisher (quadratic) using the Fisher information F,
  • Doublet-DALI using (F, G, H; fisher and doublet-DALI tensors) to include cubic and quartic terms.

Functions
---------
Model (one parameter → one observable):
    o(x) = 100 · exp(x²)

Likelihoods (with data variance σ_o²):
    exact:    log L(x) = -½ · [(o(x) - o(x₀)) / σ_o]²
    fisher:   log L(x) ≈ -½ · F · (x - x₀)²
    doublet:  log L(x) ≈ -½ · F · Δx² - ½ · G · Δx³ - ⅛ · H · Δx⁴,
              where Δx = x - x₀ and (F, G, H) are the 1D tensors.

What it does
------------
- Sets fiducial parameter x₀ and data covariance C = [[σ_o²]].
- Builds Fisher F matrix and DALI tensors (G, H) via :class:`ForecastKit` at x₀.
- Evaluates:
    • exact log-likelihood on a dense grid,
    • fisher on the same dense grid,
    • doublet-DALI on a sparser window (plotted as markers).
- Plots all three for visual comparison and prints a sanity check at x = x₀.

Defaults (exact numbers used here)
----------------------------------
- x₀ = 0.1
- C = [[1.0]]  ⇒  σ_o = 1.0
- dense grid:   linspace(-1, 1, 1000)
- sparse grid:  linspace(-0.2, 0.2, 100)
- axes limits:  xlim = (-0.3, 0.6),  ylim = (-0.65, 0.05)
- output file:  plots/dali_plot.pdf

Usage
-----
  $ python demo-scripts/07-forecast-kit-dali.py --method adaptive
  $ python demo-scripts/07-forecast-kit-dali.py --method adaptive --plot

Notes
-----
- If ``method`` is omitted, the **adaptive** backend is used; you can pass
  ``--method finite`` (or any registered backend).
- In 1D, F has shape (1,1), G → (1,1,1), H → (1,1,1,1); the demo reads them
  as scalars F[0,0], G[0,0,0], H[0,0,0,0].
- All three curves meet at x = x₀ by construction. With x₀ ≠ 0 the cubic
  term is generally nonzero, so DALI skews relative to Fisher and typically
  tracks the exact curve better away from x₀.
"""


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from derivkit.forecast_kit import ForecastKit
from utils.style import apply_plot_style, DEFAULT_COLORS


def test_model_1d(param_list) -> np.ndarray:
    """Returns one observable with a quadratic dependence on one parameter.

    Args:
        param_list: list of one parameter [x].

    Returns:
        np.ndarray: one observable [o].
    """
    x = float(param_list[0])
    obs = 1e2 * np.exp(x**2)
    return np.array([obs], dtype=float)


def loglike_1d_exact(sigma_o: float, fiducial_x: float, x: float) -> float:
    """Exact Gaussian log-likelihood for one observable with variance sigma_o^2.

    Args:
        sigma_o: standard deviation of the observable.
        fiducial_x: fiducial parameter value x0.
        x: parameter value at which to evaluate the log-likelihood.

    Returns:
        float: log-likelihood value.
    """
    delta_o = test_model_1d([x]) - test_model_1d([fiducial_x])
    return float(-0.5 * (delta_o[0] / sigma_o) ** 2)


def loglike_1d_approx(tensors: list[np.ndarray], fiducial_x: float, x: float) -> float:
    """Approximate log-likelihood using Fisher + doublet-DALI terms.

    Args:
        tensors: list of tensors [F, G, H] where
                 F is (1,1), G is (1,1,1), H is (1,1,1,1) in 1D.
        fiducial_x: fiducial parameter value x0.
        x: parameter value at which to evaluate the log-likelihood.

    Returns:
        float: approximate log-likelihood value.
    """
    dx = x - fiducial_x
    logl = 0.0
    if len(tensors) >= 1:
        f = np.asarray(tensors[0])
        logl += -0.5 * float(f[0, 0]) * dx**2
    if len(tensors) >= 3:
        g = np.asarray(tensors[1])
        h = np.asarray(tensors[2])
        logl += -0.5 * float(g[0, 0, 0]) * dx**3
        logl += -0.125 * float(h[0, 0, 0, 0]) * dx**4
    return float(logl)


def parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="forecastkit dali 1d (exact numbers)")
    ap.add_argument("--plot", action="store_true", help="show the figure instead of saving")
    ap.add_argument(
        "--method",
        type=str,
        default="adaptive",
        help='derivative backend (e.g., "adaptive", "finite")',
    )
    ap.add_argument("--dpi", type=int, default=150, help="figure dpi (default: 150)")
    return ap.parse_args()


def main() -> None:
    """Runs the ForecastKit DALI 1D demo."""
    args = parse_args()

    # exact numbers you provided
    observables = test_model_1d
    fiducial_values = [0.1]
    covmat = np.array([[1.0]], dtype=float)

    fiducial_x = float(fiducial_values[0])
    sigma_o = float(np.sqrt(covmat[0, 0]))

    xgrid = np.linspace(-1.0, 1.0, 1000)
    xgrid_sparse = np.linspace(-0.2, 0.2, 100)

    # forecast kit tensors
    fk = ForecastKit(function=observables, theta0=np.array(fiducial_values, dtype=float), cov=covmat)
    fisher_matrix = fk.fisher(method=args.method)  # (1, 1)
    dali_g, dali_h = fk.dali(method=args.method)  # (1,1,1), (1,1,1,1)
    tensors = [fisher_matrix, dali_g, dali_h]

    # likelihoods
    exact_like = [loglike_1d_exact(sigma_o, fiducial_x, x) for x in xgrid]
    fisher_like = [loglike_1d_approx([fisher_matrix], fiducial_x, x) for x in xgrid]
    dali_like_sparse = [loglike_1d_approx(tensors, fiducial_x, x) for x in xgrid_sparse]

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    blue = DEFAULT_COLORS["blue"]
    red = DEFAULT_COLORS["red"]
    yellow = DEFAULT_COLORS["yellow"]

    ax.plot(xgrid, exact_like, label="Exact Likelihood", linewidth=3, color=red)
    ax.plot(xgrid, fisher_like, label="Fisher Matrix", linewidth=3, linestyle="-", color=yellow)
    ax.plot(xgrid_sparse, dali_like_sparse, label="Doublet DALI", markersize=6,
            color=blue, linestyle="--", linewidth=3)

    ax.set_title(r"$\mathrm{observable}= 100 \cdot e^{x^2}$", fontsize=20)
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$\mathrm{log}(P)$", fontsize=20)
    ax.set_xlim(-0.3, 0.6)
    ax.set_ylim(-0.65, 0.05)
    ax.legend(fontsize=16, framealpha=1.0)
    ax.minorticks_off()
    plt.tight_layout()

    out = Path("plots/dali_plot_1d.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")  # PDF
    plt.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")  # PNG
    print(f"saved: {out} and {out.with_suffix('.png')}")

    if args.plot:
        plt.show()
    else:
        plt.close(fig)

    # sanity at x0
    exact_x0 = loglike_1d_exact(sigma_o, fiducial_x, fiducial_x)
    fisher_x0 = loglike_1d_approx([fisher_matrix], fiducial_x, fiducial_x)
    dali_x0 = loglike_1d_approx(tensors, fiducial_x, fiducial_x)
    print("\nsanity at fiducial x0:")
    print(f"  exact   : {exact_x0:.6e}")
    print(f"  fisher  : {fisher_x0:.6e}")
    print(f"  dali    : {dali_x0:.6e}")


if __name__ == "__main__":
    main()
