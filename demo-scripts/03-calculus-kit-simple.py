"""DerivKit — CalculusKit Demo (gradient, Jacobian, Hessian).

This script shows how to use :class:`CalculusKit` to compute the gradient and
Hessian of a scalar-valued function and the Jacobian of a vector-valued function,
then compare against analytic results at a chosen point.

Functions
---------
- Scalar: Rosenbrock f(x,y) = (a - x)^2 + b (y - x^2)^2  (with analytic ∇f, H)
- Vector: g(x1,x2) = [ x1^2, sin(x2), x1*x2 ]  (with analytic J)

What it does
------------
- Sets x0 = [0.7, -1.2].
- Computes:
  • ∇f and H with **adaptive** derivatives,
  • J for the vector function with **adaptive** derivatives.
- Prints numeric vs analytic results and their deltas (max|Δ| and ||Δ||₂).

Notes:
-----
- You can pass through backend controls (e.g., ``method="finite"``) via the
  CalculusKit methods’ kwargs.
- For non-analytic targets, compare against a finite-difference baseline instead.

Usage
-----
    $ python demo-scripts/03-calculus-kit-simple.py

Requirements
------------
- ``derivkit`` importable in your environment.
"""

from __future__ import annotations

import numpy as np

from derivkit.calculus_kit import CalculusKit


def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """Scalar-valued Rosenbrock: f(x,y) = (a - x)^2 + b (y - x^2)^2."""
    x1, x2 = float(x[0]), float(x[1])
    return (a - x1) ** 2 + b * (x2 - x1 ** 2) ** 2

def rosenbrock_grad_analytic(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    """Scalar-valued Rosenbrock gradient analytic."""
    x1, x2 = float(x[0]), float(x[1])
    dfdx = 2 * (x1 - a) - 4 * b * x1 * (x2 - x1 ** 2)
    dfdy = 2 * b * (x2 - x1 ** 2)
    return np.array([dfdx, dfdy], dtype=float)

def rosenbrock_hess_analytic(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    """Scalar-valued Rosenbrock Hessian analytic."""
    x1, x2 = float(x[0]), float(x[1])
    dxx = 2 - 4 * b * x2 + 12 * b * x1 ** 2
    dxy = -4 * b * x1
    dyy = 2 * b
    return np.array([[dxx, dxy],
                     [dxy, dyy]], dtype=float)

def vec_func(x: np.ndarray) -> np.ndarray:
    """Vector-valued example: g(x1,x2) = [ x1^2,  sin(x2),  x1*x2 ]."""
    x1, x2 = float(x[0]), float(x[1])
    return np.array([x1**2, np.sin(x2), x1 * x2], dtype=float)

def vec_func_jac_analytic(x: np.ndarray) -> np.ndarray:
    """Jacobian of vec_fun: shape (3 outputs, 2 params)."""
    x1, x2 = float(x[0]), float(x[1])
    # rows are outputs, columns are partials wrt [x1, x2]
    return np.array([
        [2 * x1, 0.0],          # d(x1^2)/dx
        [0.0,    np.cos(x2)],   # d(sin x2)/dx
        [x2,     x1],           # d(x1*x2)/dx
    ], dtype=float)

def pretty_print(name: str, arr: np.ndarray) -> None:
    """Pretty-print an array with a name."""
    print(f"{name}:\n{np.array(arr, dtype=float)}\n")


def show_delta(name: str, a: np.ndarray, b: np.ndarray) -> None:
    """Computes and prints the difference between two arrays."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    diff = a - b
    eps = 1e-15
    denom = np.maximum(1.0, np.abs(b)) + eps
    rel_elem = np.abs(diff) / denom
    rel_max = np.max(rel_elem)
    rel_rms = np.sqrt(np.mean(rel_elem**2))
    print(f"{name} delta (num - analytic):")
    print(diff)
    print(f"  max rel err = {rel_max:.3e},  rms rel err = {rel_rms:.3e}")
    print(f"  max|Δ| = {np.max(np.abs(diff)):.3e},  ||Δ||₂ = {np.linalg.norm(diff):.3e}\n")


def main() -> None:
    """Runs the CalculusKit demo for gradient, Hessian, and Jacobian."""
    # Evaluation point
    x0 = np.array([0.7, -1.2], dtype=float)
    print("=== CalculusKit demo at x0 =", x0, "===\n")

    # Instantiate CalculusKit for each function
    calc_rosen = CalculusKit(rosenbrock, x0=x0)
    calc_vec   = CalculusKit(vec_func, x0=x0)

    # --- Gradient & Hessian (scalar-valued) ---
    grad_num = calc_rosen.gradient(method="adaptive")
    hess_num = calc_rosen.hessian(method="adaptive")

    grad_ref = rosenbrock_grad_analytic(x0)
    hess_ref = rosenbrock_hess_analytic(x0)

    pretty_print("∇f (numeric)", grad_num)
    pretty_print("∇f (analytic)", grad_ref)
    show_delta("∇f", grad_num, grad_ref)

    pretty_print("H (numeric)", hess_num)
    pretty_print("H (analytic)", hess_ref)
    show_delta("H", hess_num, hess_ref)

    # --- Jacobian (vector-valued) ---
    jac_num = calc_vec.jacobian(method="adaptive")
    jac_ref = vec_func_jac_analytic(x0)

    pretty_print("J (numeric)", jac_num)
    pretty_print("J (analytic)", jac_ref)
    show_delta("J", jac_num, jac_ref)

    print("Done.")

if __name__ == "__main__":
    main()
