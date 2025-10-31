"""DerivKit — ForecastKit Fisher Information Demo.

This demo shows how to use :class:`ForecastKit` to compute the Fisher
information matrix for a simple 2-parameter model (x and y) with 2 observables
 (o1 and o2). We compare the Fisher matrix computed via ForecastKit to
the analytic Fisher matrix derived from the model's Jacobian.

Functions:
---------
    Model mapping parameters to observables (the parameters (θ1, θ2) determine the observables (o1, o2)):
        o1 = θ1 + θ2
        o2 = θ1² + 2θ2²
    Jacobian of the model (tells us how the observables change when the parameters change):
        J(θ1, θ2) = [[1, 1],
                  [2θ1, 4θ2]]

What it does
------------
- Defines a model function mapping parameters (θ1, θ2) to observables (o1, o2).
- Computes the Jacobian of the model analytically.
- Computes the Fisher matrix numerically using ForecastKit.
- Compares the numeric Fisher matrix to the analytic Fisher matrix.
- Computes the parameter covariance matrix (inverse of the Fisher matrix).
- Extracts and prints the 1-sigma and 2-sigma uncertainties on the parameters x and y.
- Optionally plots the Fisher ellipses corresponding to the parameter uncertainties.

Usage
-----
    $ python demo-scripts/05-forecast-kit-fisher.py --method adaptive
    $ python demo-scripts/05-forecast-kit-fisher.py --method adaptive --plot

Notes:
-----
Assuming the data covariance matrix is the identity, the Fisher matrix is
computed as J-transpose times J. The inverse of the Fisher matrix gives the
parameter covariance, from which we can read off the 1-sigma and 2-sigma
uncertainties on θ1 and θ2.
In other words, the observables are what the model predicts, and the Fisher
matrix quantifies how precisely we can estimate the parameters that produce
those observables.

- If ``method`` is omitted, the **adaptive** backend is used.
- You can force a backend via ``method="finite"`` (or another registered one).
- To list available methods at runtime, use
  ``from derivkit.derivative_kit import available_methods``.

Requirements
------------
- ``derivkit`` importable in your environment.
- ``numpy`` for numerical computations.
- ``matplotlib`` for plotting (if --plot is used).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from derivkit import ForecastKit
from utils.style import apply_plot_style, DEFAULT_COLORS


# --- constants for pretty printing/labels ---
# 1D confidence levels for marginal errors
PCT_1D_1SIG = 0.682689492  # 68.27%
PCT_1D_2SIG = 0.954499736  # 95.45%

# 2D confidence levels used for Fisher ellipses
PCT_2D_1SIG = 0.683  # 68.3%
PCT_2D_2SIG = 0.955  # 95.5%


def plot_fisher_ellipse(theta0: np.ndarray,
                        fisher_matrix: np.ndarray,
                        *,
                        level: float = PCT_2D_1SIG,
                        ax=None,
                        label: str | None = None,
                        lw: float = 2.0,
                        ls: str = "-",
                        color: str | None = None) -> plt.Axes:
    """Plots the Fisher ellipse for a 2×2 Fisher matrix `F2` at parameter point `theta0`.

    Args:
        theta0: Center of the ellipse (2D parameter point).
        fisher_matrix: Fisher matrix (shape 2x2).
        level: Confidence level for the ellipse (0.393, 0.683, 0.955, 0.997).
        ax: Matplotlib Axes to plot on (creates new if None).
        label: Label for the ellipse (for legend).
        lw: Line width for the ellipse.
        ls: Line style for the ellipse.
        color: Line color for the ellipse (default uses style blue).

    Returns:
        Matplotlib Axes with the ellipse plotted.
    """
    f_mat = np.asarray(fisher_matrix, dtype=float)
    if f_mat.shape != (2, 2):
        raise ValueError(f"Expected a 2×2 Fisher; got shape {f_mat.shape}.")
    if not np.allclose(f_mat, f_mat.T, rtol=1e-10, atol=1e-12):
        f_mat = 0.5 * (f_mat + f_mat.T)  # symmetrize

    # 2 dof chi2 quantiles mapping
    level_to_k2 = {0.393: 1.00, 0.683: 2.30, 0.955: 6.17, 0.997: 11.8}
    k2 = level_to_k2.get(level, 2.30)

    param_cov = np.linalg.pinv(f_mat, rcond=1e-12)  # covariance
    vals, vecs = np.linalg.eigh(param_cov)  # ascending eigenvalues
    t = np.linspace(0, 2*np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])
    ellipse = vecs @ np.diag(np.sqrt(vals * k2)) @ circle

    if not color:
        color = DEFAULT_COLORS["blue"]

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(theta0[0] + ellipse[0], theta0[1] + ellipse[1],
            label=label, linewidth=lw, linestyle=ls, color=color)
    ax.scatter([theta0[0]], [theta0[1]], s=25, color=color)
    ax.set_xlabel(r"parameter $\theta_1$", fontsize=13)
    ax.set_ylabel(r"parameter $\theta_2$", fontsize=13)
    ax.set_aspect("equal", adjustable="box")
    if label:
        ax.legend(frameon=False, fontsize=13)
    return ax


def model(param_list) -> np.ndarray:
    """Simple model: maps parameters(θ1, θ2) to observables (o1, o2) .

    o1 = θ1 + θ2
    o2 = θ1^2 + 2θ2^2

    Args:
        param_list: List or array of parameters [θ1, θ2].

    Returns:
        Array of observables [o1, o2].
    """
    theta1 = float(param_list[0])
    theta2 = float(param_list[1])
    obs1 = theta1 + theta2
    obs2 = theta1 * theta1 + 2.0 * theta2 * theta2
    return np.array([obs1, obs2], dtype=float)


def jacobian_analytic(theta1: float, theta2: float) -> np.ndarray:
    """Analytic Jacobian of `my_function`.

    Args:
        theta1: Parameter theta1.
        theta2: Parameter theta2.

    Returns:
        Jacobian matrix J (shape 2x2).
    """
    return np.array([[1.0, 1.0],
                     [2.0 * theta1, 4.0 * theta2]], dtype=float)


def fisher_analytic(theta1: float, theta2: float, cov: np.ndarray) -> np.ndarray:
    """Analytic Fisher matrix for the model at (x, y) with data covariance `cov`.

    Args:
        theta1: Parameter theta1.
        theta2: Parameter theta2.
        cov: Data covariance matrix (shape 2x2).

    Returns:
        Fisher information matrix (shape 2x2).
    """
    jac = jacobian_analytic(theta1, theta2)
    cov_inv = np.linalg.inv(cov)
    return jac.T @ cov_inv @ jac


def array_delta(name: str, num: np.ndarray, ref: np.ndarray) -> None:
    """Computes and prints the difference between two arrays.

    Args:
        name: Name of the array (for printing).
        num: Numeric array.
        ref: Reference (analytic) array.

    Returns:
        None
    """
    d = num - ref
    print(f"\n{name} (numeric):\n{num}")
    print(f"\n{name} (analytic):\n{ref}")
    print(f"\n $\\Delta$ = numeric - analytic:\n{d}")
    print(f"$\\max |\\Delta|$ = {np.max(np.abs(d)):.3e}, $|| \\Delta||_2$  = {np.linalg.norm(d):.3e}")


def main() -> None:
    """Runs the DerivKit ForecastKit Fisher demo."""
    ap = argparse.ArgumentParser(description="DerivKit Fisher demo")
    ap.add_argument("--method", default="adaptive", choices=["adaptive", "finite"],
                    help="Derivative backend for ForecastKit (default: adaptive).")
    ap.add_argument("--rcond", type=float, default=1e-12,
                    help="rcond for pseudo-inverse of Fisher (default: 1e-12).")
    ap.add_argument("--plot", action="store_true",
                    help="Show 1σ (68.3%) and 2σ (95.5%) ellipses from F^{-1}.")
    args = ap.parse_args()

    # Inputs
    theta0 = np.array([1.0, 2.0], dtype=float)  # (x=1, y=2)
    cov = np.array([[1.0, 0.0],
                    [0.0, 1.0]], dtype=float)  # identity covariance

    print("=== Fisher demo ===")
    print("fiducial $\\theta_0$ =", theta0)
    print("covariance $C$ =\n", cov)
    print(f"backend method = {args.method}")

    # ForecastKit Fisher
    fk = ForecastKit(function=model, theta0=theta0, cov=cov)
    fish_num = fk.fisher(method=args.method)
    fish_ref = fisher_analytic(theta0[0], theta0[1], cov)
    array_delta("Fisher", fish_num, fish_ref)

    # Parameter covariance and 1sigma / 2sigma (marginal) errors
    fish_inv = np.linalg.pinv(fish_num, rcond=args.rcond)
    sigma_1 = np.sqrt(np.diag(fish_inv))     # 1σ marginal
    sigma_2 = 2.0 * sigma_1              # 2σ marginal
    print("\nParameter covariance (F^{-1}):\n", fish_inv)
    print(f"\n1σ ({PCT_1D_1SIG*100:.2f}%) errors on (x, y):", sigma_1)
    print(f"2σ ({PCT_1D_2SIG*100:.2f}%) errors on (x, y):", sigma_2)

    if args.plot:
        apply_plot_style()
        fig, ax = plt.subplots()
        plot_fisher_ellipse(theta0, fish_num, level=PCT_2D_1SIG,
                            ax=ax, label="1σ (68.3%)", ls="--",  lw=2.0)
        plot_fisher_ellipse(theta0, fish_num, level=PCT_2D_2SIG,
                            ax=ax, label="2σ (95.5%)", ls="-", lw=2.0)
        title = "Fisher $1 \\sigma \\: (68.3\\%)$ and $2 \\sigma \\: (95.5\\%)$ contours"
        ax.set_title(title, fontsize=15)
        plt.tight_layout()

        # --- save as PDF and PNG ---
        from pathlib import Path
        out = Path("plots/fisher_ellipses_2d.pdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"saved: {out} and {out.with_suffix('.png')}")

        plt.show()

        print("Done. Another ellipse, another insight.")


if __name__ == "__main__":
    main()
