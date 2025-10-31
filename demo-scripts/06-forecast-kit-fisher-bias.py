"""DerivKit — ForecastKit Fisher Bias Demo.

This demo shows how to use :class:`ForecastKit` to compute the *Fisher bias*
and the corresponding parameter shift Δθ caused by a systematic offset in the
observables.

Functions:
---------
We reuse the same 2-parameter model and utilities from the Fisher demo:
  Model:    o1 = θ1 + θ2
            o2 = θ1² + 2 θ2²
  Jacobian: J(θ1, θ2) = [[1, 1],
                        [2θ1, 4θ2]]

What it does
------------
- Defines the model and its analytic Jacobian.
- Sets fiducial parameters θ0 and (by default) identity data covariance C.
- Creates a toy *systematic* offset Δd in the observables:
    data_with = model(θ0) + Δd
    data_without = model(θ0)
  so that Δν = data_with - data_without = Δd.
- Computes:
    • Fisher matrix F with ForecastKit (same backend flag as Fisher demo),
    • Fisher bias vector b and parameter shift Δθ via ForecastKit,
    • Analytic Δθ_analytic = F^{-1} Jᵀ C^{-1} Δν for cross-check.
- Prints numeric vs analytic comparisons.
- Optionally plots 1σ/2σ Fisher ellipses and the bias arrow θ0 → θ0 + Δθ.

Usage
-----
  $ python demo-scripts/06-forecast-kit-fisher-bias.py --method adaptive
  $ python demo-scripts/06-forecast-kit-fisher-bias.py --plot
  $ python demo-scripts/06-forecast-kit-fisher-bias.py --sys 1.5 -3.6

Notes:
-----
- If ``method`` is omitted, the **adaptive** backend is used.
- The analytic first-order bias formula for small Δν is:
      Δθ ≈ F^{-1} Jᵀ C^{-1} Δν
  where F = Jᵀ C^{-1} J at θ0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from derivkit import ForecastKit
from utils.style import DEFAULT_COLORS, apply_plot_style

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
                        color: str | None = None):
    """Plots the Fisher ellipse for a 2×2 Fisher matrix `F2` at parameter point `theta0`.

    Args:
        theta0: Center of the ellipse (2D parameter point).
        fisher_matrix: Fisher matrix (shape 2x2).
        level: Confidence level for the ellipse (0.393, 0.683, 0.955, 0.997).
        ax: Matplotlib Axes to plot on (creates new if None).
        label: Label for the ellipse (for legend).
        lw: Line width for the ellipse.
        ls: Line style for the ellipse.
        color: Color of the ellipse.

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

    param_cov = np.linalg.pinv(f_mat, rcond=1e-12)
    vals, vecs = np.linalg.eigh(param_cov)  # ascending eigenvalues

    t = np.linspace(0, 2*np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])
    ellipse = vecs @ np.diag(np.sqrt(vals * k2)) @ circle

    if not color:
        color= "#3b9ab2"
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


def analytic_bias_delta_theta(theta0: np.ndarray, cov: np.ndarray, delta_nu: np.ndarray) -> np.ndarray:
    """First-order analytic delta_theta from Fisher bias formula.

    Args:
        theta0: Fiducial parameter point (array of shape 2).
        cov: Data covariance matrix (shape 2x2).
        delta_nu: Systematic offset in observables (shape 2).

    Returns:
        Analytic delta_theta (shape 2)
    """
    jac = jacobian_analytic(theta0[0], theta0[1])
    cov_inv = np.linalg.inv(cov)
    fish = jac.T @ cov_inv @ jac
    return np.linalg.pinv(fish, rcond=1e-12) @ (jac.T @ cov_inv @ delta_nu)


def main() -> None:
    """Runs the ForecastKit Fisher bias demo."""
    ap = argparse.ArgumentParser(description="DerivKit Fisher Bias demo")
    ap.add_argument("--method", default="adaptive", choices=["adaptive", "finite"],
                    help="Derivative backend for ForecastKit (default: adaptive).")
    ap.add_argument("--rcond", type=float, default=1e-12,
                    help="rcond for pseudo-inverse (default: 1e-12).")
    ap.add_argument("--plot", action="store_true",
                    help="Plot 1σ/2σ ellipses and the bias arrow.")
    ap.add_argument("--sys", nargs=2, type=float, metavar=("d1", "d2"),
                    default=(0.5, -0.2),
                    help="Systematic offset Δd applied to observables (default: 0.05, -0.02).")
    args = ap.parse_args()

    # Fiducial point and covariance
    theta0 = np.array([1.0, 2.0], dtype=float)
    cov = np.array([[1.0, 0.0],
                    [0.0, 1.0]], dtype=float)  # identity

    # Observables with/without systematics
    d_without = model(theta0)
    sys_offset = np.array(args.sys, dtype=float)
    d_with = d_without + sys_offset

    print("=== Fisher Bias demo ===")
    print("fiducial θ0 =", theta0)
    print("covariance C =\n", cov)
    print(f"backend method = {args.method}")
    print("model(θ0) =", d_without)
    print("systematic Δd =", sys_offset)

    # ForecastKit pipeline
    fk = ForecastKit(function=model, theta0=theta0, cov=cov)

    fish_num = fk.fisher(method=args.method)
    fish_ref = fisher_analytic(theta0[0], theta0[1], cov)
    array_delta("Fisher", fish_num, fish_ref)

    delta_nu = fk.delta_nu(data_with=d_with, data_without=d_without)
    print("\nΔν =", delta_nu)

    # Fisher bias + parameter shift from ForecastKit
    bias_vec, dtheta_num = fk.fisher_bias(
        fisher_matrix=fish_num,
        delta_nu=delta_nu,
        method=args.method,
        rcond=args.rcond,
    )
    # Note: Your LikelihoodExpansion.build_fisher_bias returns (bias, Δθ).
    # If your API differs, adjust unpacking above accordingly.

    # Analytic Δθ for cross-check
    dtheta_ref = analytic_bias_delta_theta(theta0, cov, delta_nu)
    array_delta("Δθ (parameter shift)", dtheta_num, dtheta_ref)

    # Report marginal σ from F^{-1} and the shift in σ-units
    f_inv = np.linalg.pinv(fish_num, rcond=args.rcond)
    sigma = np.sqrt(np.diag(f_inv))
    z_units = dtheta_num / sigma
    print("\nParameter covariance (F^{-1}):\n", f_inv)
    print("marginal 1σ:", sigma)
    print("Δθ in σ-units:", z_units)

    if args.plot:
        # Setting up our custom style
        apply_plot_style()
        red = DEFAULT_COLORS["red"]
        blue = DEFAULT_COLORS["blue"]

        fig, ax = plt.subplots()

        theta_biased = theta0 + dtheta_num

        # Plot fiducial ellipse
        plot_fisher_ellipse(theta0, fish_num, level=PCT_2D_1SIG,
                            ax=ax, label="fiducial 1σ", ls="-", lw=2.0, color=red)

        # Plot biased ellipse (same Fisher)
        plot_fisher_ellipse(theta_biased, fish_num, level=PCT_2D_1SIG,
                            ax=ax, label="biased 1σ", ls="-", lw=2.0)

        ax.set_aspect("equal", adjustable="box")
        ax.annotate("", xy=theta_biased, xytext=theta0,
                    arrowprops=dict(arrowstyle="->", lw=2.0, color=blue))
        ax.scatter(*theta_biased, s=25, color=blue)

        ax.set_title("Fisher bias: 1σ ellipses (fiducial vs biased)", fontsize=15)
        plt.tight_layout()

        # --- save figure both as PDF and PNG ---
        out = Path("plots/fisher_bias_2d.pdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"saved: {out} and {out.with_suffix('.png')}")
        plt.show()

        print("Done. Small steps in θ, big steps for understanding.")

if __name__ == "__main__":
    main()
