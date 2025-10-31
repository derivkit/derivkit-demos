"""DerivKit — CalculusKit Demo (advanced analytic: gradient, Hessian, Jacobian).

This demo uses a more complex scalar function (exp/trig/poly/log mix) and a
richer vector function, both with closed-form derivatives, to validate
CalculusKit’s adaptive derivatives against analytic truth.

Functions
---------
Scalar:
    f(x1, x2) = exp(x1)*sin(x2) + 0.5*x1^2*x2^3 - log(1 + x1^2 + x2^2)

Vector:
    g(x1, x2) = [
        exp(x1)*cos(x2) + x1*x2**2,
        x1**2 * x2 + sin(x1*x2),
        log(1 + x1**2 * x2**2) + cosh(x1) - sinh(x2)
    ]

What it does
------------
- Sets x0 = [0.7, -1.2].
- Computes:
  • ∇f and H with **adaptive** derivatives,
  • J for the vector function with **adaptive** derivatives.
- Prints numeric vs analytic results and their deltas.

Usage
-----
    $ python demo-scripts/03-calculus-kit-advanced.py
"""

from __future__ import annotations

import numpy as np

from derivkit.calculus_kit import CalculusKit


def f_scalar_function(x: np.ndarray) -> float:
    """Scalar function f: R^2 -> R with mixed nonlinearity."""
    x1, x2 = float(x[0]), float(x[1])
    return (
        np.exp(x1) * np.sin(x2)
        + 0.5 * x1**2 * x2**3
        - np.log(1.0 + x1**2 + x2**2)
    )


def f_grad_analytic(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of f."""
    x1, x2 = float(x[0]), float(x[1])
    denom = 1.0 + x1**2 + x2**2
    dfdx1 = np.exp(x1) * np.sin(x2) + x1 * x2**3 - 2.0 * x1 / denom
    dfdx2 = np.exp(x1) * np.cos(x2) + 1.5 * x1**2 * x2**2 - 2.0 * x2 / denom
    return np.array([dfdx1, dfdx2], dtype=float)


def f_hess_analytic(x: np.ndarray) -> np.ndarray:
    """Analytic Hessian of f."""
    x1, x2 = float(x[0]), float(x[1])
    q = x1**2 + x2**2
    denom = 1.0 + q
    denom2 = denom**2

    dxx = np.exp(x1) * np.sin(x2) + x2**3 - 2.0 / denom + 4.0 * x1**2 / denom2
    dxy = np.exp(x1) * np.cos(x2) + 3.0 * x1 * x2**2 + 4.0 * x1 * x2 / denom2
    dyy = -np.exp(x1) * np.sin(x2) + 3.0 * x1**2 * x2 - 2.0 / denom + 4.0 * x2**2 / denom2

    return np.array([[dxx, dxy], [dxy, dyy]], dtype=float)


def g_vector_function(x: np.ndarray) -> np.ndarray:
    """Vector function g: R^2 -> R^3 with mixed nonlinearity."""
    x1, x2 = float(x[0]), float(x[1])
    return np.array(
        [
            np.exp(x1) * np.cos(x2) + x1 * x2**2,
            x1**2 * x2 + np.sin(x1 * x2),
            np.log(1.0 + x1**2 * x2**2) + np.cosh(x1) - np.sinh(x2),
        ],
        dtype=float,
    )


def g_jac_analytic(x: np.ndarray) -> np.ndarray:
    """Analytic Jacobian of g; shape (3 outputs, 2 params)."""
    x1, x2 = float(x[0]), float(x[1])

    # Row 1: h1 = exp(x1)*cos(x2) + x1*x2^2
    dh1_dx1 = np.exp(x1) * np.cos(x2) + x2**2
    dh1_dx2 = -np.exp(x1) * np.sin(x2) + 2.0 * x1 * x2

    # Row 2: h2 = x1^2 * x2 + sin(x1*x2)
    dh2_dx1 = 2.0 * x1 * x2 + np.cos(x1 * x2) * x2
    dh2_dx2 = x1**2 + np.cos(x1 * x2) * x1

    # Row 3: h3 = log(1 + x1^2 x2^2) + cosh(x1) - sinh(x2)
    r = 1.0 + x1**2 * x2**2
    dh3_dx1 = (2.0 * x1 * x2**2) / r + np.sinh(x1)
    dh3_dx2 = (2.0 * x2 * x1**2) / r - np.cosh(x2)

    return np.array(
        [
            [dh1_dx1, dh1_dx2],
            [dh2_dx1, dh2_dx2],
            [dh3_dx1, dh3_dx2],
        ],
        dtype=float,
    )


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
    """Runs the CalculusKit advanced demo with analytic gradient, Hessian, Jacobian."""
    x0 = np.array([0.7, -1.2], dtype=float)
    print("=== CalculusKit advanced demo at x0 =", x0, "===\n")

    calc_f = CalculusKit(f_scalar_function, x0=x0)
    calc_g = CalculusKit(g_vector_function, x0=x0)

    # Scalar: gradient & Hessian
    grad_num = calc_f.gradient(method="adaptive")
    hess_num = calc_f.hessian(method="adaptive")

    grad_ref = f_grad_analytic(x0)
    hess_ref = f_hess_analytic(x0)

    pretty_print("∇f (numeric)", grad_num)
    pretty_print("∇f (analytic)", grad_ref)
    show_delta("∇f", grad_num, grad_ref)

    pretty_print("H (numeric)", hess_num)
    pretty_print("H (analytic)", hess_ref)
    show_delta("H", hess_num, hess_ref)

    # Vector: Jacobian
    jac_num = calc_g.jacobian(method="adaptive")
    jac_ref = g_jac_analytic(x0)

    pretty_print("J (numeric)", jac_num)
    pretty_print("J (analytic)", jac_ref)
    show_delta("J", jac_num, jac_ref)

    print("Done. ∇ guides, H bends, J translates.")

if __name__ == "__main__":
    main()
