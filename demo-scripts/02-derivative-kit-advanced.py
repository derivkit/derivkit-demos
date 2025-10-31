"""DerivKit — Advanced Analytic Function Demo.

This script compares :class:`DerivativeKit`’s **adaptive** (polynomial-fit)
and **finite-difference** backends on a more oscillatory function, so you can
see how node choices and spacing affect stability. We evaluate at x0 = 0.3.

Function
--------
    f(x) = exp(-x^2) * sin(3x) + 0.1 * x^3

What it does
------------
- Defines f(x) and sets x0 = 0.3.
- Computes 1st/2nd/3rd derivatives with:
  • **adaptive** using a user-supplied uniform grid for d¹ (via `grid=("offsets", np.arange(...))`),
  • **adaptive** using the default Chebyshev grid with `spacing=0.05` for d², and
  • **adaptive** with `spacing=1/8` (half-width) and `n_workers=2` for d³.
- Also computes a **5-point central** finite-difference baseline for d¹–d³.
- Prints numerical results alongside the analytic derivatives for immediate comparison.

Notes
-----
- When you pass `grid=("offsets", ...)`, `n_points` and `spacing` are ignored for that call.
- `spacing` denotes the **half-width** of the symmetric sampling window around `x0`
  when building default Chebyshev offsets.
- For oscillatory functions, keep the window moderate and the point count modest;
  a small `ridge` can stabilize higher orders.

Usage
-----
    $ python demo-scripts/02-demo_derivative-kit_advanced.py

Requirements
------------
- ``derivkit`` importable in your environment.
"""


from __future__ import annotations

import numpy as np

from derivkit.derivative_kit import DerivativeKit


def damped_sine_plus_cubic(x: float) -> float:
    """Returns a damped sine plus cubic function for the demo."""
    return np.exp(-x * x) * np.sin(3.0 * x) + 0.1 * x**3


def truth_1_2_3(x: float) -> tuple[float, float, float]:
    """Returns the analytic 1st, 2nd, and 3rd derivatives at x."""
    ex = np.exp(-x * x)
    s3 = np.sin(3.0 * x)
    c3 = np.cos(3.0 * x)
    d1 = ex * (3.0 * c3 - 2.0 * x * s3) + 0.3 * x * x
    d2 = ex * ((4.0 * x * x - 11.0) * s3 - 12.0 * x * c3) + 0.6 * x
    d3 = ex * ((-8.0 * x**3 + 66.0 * x) * s3 + (36.0 * x * x - 45.0) * c3) + 0.6
    return d1, d2, d3


def main() -> None:
    """Runs the DerivativeKit demo for a more advanced function."""
    title = "f(x) = exp(-x^2) * sin(3x) + 0.1 * x^3"
    x0 = 0.3
    dk = DerivativeKit(function=damped_sine_plus_cubic, x0=x0)

    # --- Order 1: user-supplied grid built from arange in angle-space ---
    h = 0.16  # we use tighter window for uniform spacing
    step = 0.02
    t1_offsets = np.arange(-h, h, step)

    d1_ad = dk.differentiate(
        order=1,
        grid=("offsets", t1_offsets),
    )

    # --- Order 2: Adaptive with a user-supplied grid (offsets) ---
    d2_ad = dk.differentiate(
        order=2,
        n_points=19,
        spacing=0.05,
        base_abs=1e-3,
    )

    # --- Order 3: Adaptive with spacing as a *literal fraction* (1/8) ---
    d3_ad = dk.differentiate(
        order=3,
        n_points=19,
        spacing=1/8,
        n_workers=2,
    )

    # Finite-difference baseline (5-point central) for orders 1..3
    d1_fd = dk.differentiate(method="finite", order=1, num_points=3)
    d2_fd = dk.differentiate(method="finite", order=2, num_points=7)
    d3_fd = dk.differentiate(method="finite", order=3, stepsize=5e-4)

    # Analytic truth
    t1, t2, t3 = truth_1_2_3(x0)

    # Pretty print
    thick = "=" * 62
    thin = "-" * 62
    print(thick)
    print(title)
    print(f"x0 = {x0:.4f}")
    print(thin)
    print("Order   Analytic          Adaptive          Finite")
    print(thin)
    print(f"  1   {t1:>12.6f}   {d1_ad:>12.6f}   {d1_fd:>12.6f}")
    print(f"  2   {t2:>12.6f}   {d2_ad:>12.6f}   {d2_fd:>12.6f}")
    print(f"  3   {t3:>12.6f}   {d3_ad:>12.6f}   {d3_fd:>12.6f}")
    print(thick)


if __name__ == "__main__":
    main()
