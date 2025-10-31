#!/usr/bin/env python3
"""DerivKit — Fisher Information Demo (toy 2-param, 2-observable)

We forecast for a 2-parameter model with 2 observables and compare the Fisher
matrix computed via ForecastKit to the analytic Fisher (from Jᵀ C⁻¹ J).

Model:
    o1(x,y) = x + y
    o2(x,y) = x^2 + 2 y^2

Jacobian:
    J(x,y) = [[ 1,  1],
              [2x, 4y]]

With covariance C = I, Fisher = Jᵀ J.

Usage:
    $ python demo-scripts/05-forecasting-kit-fisher.py --method adaptive
    $ python demo-scripts/05-forecasting-kit-fisher.py --method finite --plot
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from derivkit import ForecastKit


# ---- Inline Fisher ellipse plotting (2x2 Fisher only) ----
def plot_fisher_ellipse(theta0: np.ndarray,
                        F2: np.ndarray,
                        *,
                        level: float = 0.683,
                        ax=None,
                        label: str | None = None,
                        lw: float = 2.0):
    """
    Plot the Gaussian 2D confidence ellipse implied by a 2×2 Fisher matrix F2 centered at theta0.
    level: confidence in 2D (0.393, 0.683, 0.955, 0.997 → k^2≈1.00, 2.30, 6.17, 11.8).
    """
    F2 = np.asarray(F2, dtype=float)
    if F2.shape != (2, 2):
        raise ValueError(f"Expected a 2×2 Fisher; got shape {F2.shape}.")
    if not np.allclose(F2, F2.T, rtol=1e-10, atol=1e-12):
        F2 = 0.5 * (F2 + F2.T)  # symmetrize

    # 2 dof chi2 quantiles
    level_to_k2 = {0.393: 1.00, 0.683: 2.30, 0.955: 6.17, 0.997: 11.8}
    k2 = level_to_k2.get(level, 2.30)

    C2 = np.linalg.pinv(F2, rcond=1e-12)  # covariance
    vals, vecs = np.linalg.eigh(C2)       # ascending eigenvalues
    t = np.linspace(0, 2*np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])
    ellipse = vecs @ np.diag(np.sqrt(vals * k2)) @ circle

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(theta0[0] + ellipse[0], theta0[1] + ellipse[1], label=label, linewidth=lw)
    ax.scatter([theta0[0]], [theta0[1]], s=25)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_aspect("equal", adjustable="box")
    if label:
        ax.legend(frameon=False)
    return ax


# --- Model mapping parameters -> observables (shape (2,)) ---
def myf(param_list) -> np.ndarray:
    x = float(param_list[0])
    y = float(param_list[1])
    obs1 = x + y
    obs2 = x * x + 2.0 * y * y
    return np.array([obs1, obs2], dtype=float)


# --- Analytic Jacobian and Fisher ---
def jacobian_analytic(x: float, y: float) -> np.ndarray:
    return np.array([[1.0, 1.0],
                     [2.0 * x, 4.0 * y]], dtype=float)


def fisher_analytic(x: float, y: float, cov: np.ndarray) -> np.ndarray:
    J = jacobian_analytic(x, y)
    Cinv = np.linalg.inv(cov)
    return J.T @ Cinv @ J


def brief_delta(name: str, num: np.ndarray, ref: np.ndarray) -> None:
    d = num - ref
    print(f"\n{name} (numeric):\n{num}")
    print(f"\n{name} (analytic):\n{ref}")
    print(f"\nΔ = numeric - analytic:\n{d}")
    print(f"max|Δ| = {np.max(np.abs(d)):.3e},  ||Δ||₂ = {np.linalg.norm(d):.3e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="DerivKit Fisher demo")
    ap.add_argument("--method", default="adaptive", choices=["adaptive", "finite"],
                    help="Derivative backend for ForecastKit (default: adaptive).")
    ap.add_argument("--rcond", type=float, default=1e-12,
                    help="rcond for pseudo-inverse of Fisher (default: 1e-12).")
    ap.add_argument("--plot", action="store_true",
                    help="Show 68% ellipse from F^{-1}.")
    args = ap.parse_args()

    # Inputs
    theta0 = np.array([1.0, 2.0], dtype=float)  # (x=1, y=2)
    cov = np.array([[1.0, 0.0],
                    [0.0, 1.0]], dtype=float)   # identity covariance

    print("=== Fisher demo ===")
    print("fiducial θ0 =", theta0)
    print("covariance C =\n", cov)
    print(f"backend method = {args.method}")

    # ForecastKit Fisher
    fk = ForecastKit(function=myf, theta0=theta0, cov=cov)
    F_num = fk.fisher(method=args.method)
    F_ref = fisher_analytic(theta0[0], theta0[1], cov)
    brief_delta("Fisher", F_num, F_ref)

    # Parameter covariance and 1σ
    Finv = np.linalg.pinv(F_num, rcond=args.rcond)
    sigma = np.sqrt(np.diag(Finv))
    print("\nParameter covariance (F^{-1}):\n", Finv)
    print("\n1σ errors on (x, y):", sigma)

    if args.plot:
        plot_fisher_ellipse(theta0, F_num, level=0.683, label="68%")
        plt.title("Fisher 68% ellipse")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
