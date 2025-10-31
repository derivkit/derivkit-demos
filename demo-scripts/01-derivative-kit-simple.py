"""DerivKit — Simple Analytic Function Demo.

This tiny script shows how to use :class:`DerivativeKit` to compute the
first and second derivatives of a scalar function at a single point.
We use f(x) = x^2 so the analytic derivatives are trivial and we can
compare numerics to truth.

Function
--------
    f(x)  = x^2
    f'(x) = 2x
    f''(x)= 2

What it does
------------
- Defines f(x) and sets x0 = 1.0.
- Computes 1st/2nd derivatives with:
  • the default **adaptive** method, and
  • explicit **finite** differences (baseline).
- Prints numerical and analytic values side by side.

Notes
-----
- If ``method`` is omitted, the **adaptive** backend is used.
- You can force a backend via ``method="finite"`` (or another registered one).
- To list available methods at runtime, use
  ``from derivkit.derivative_kit import available_methods``.

Usage
-----
    $ python demo-scripts/01-derivative-kit-simple.py

Requirements
------------
- ``derivkit`` importable in your environment.
"""


from __future__ import annotations

from derivkit.derivative_kit import DerivativeKit


def simple_function(x: float) -> float:
    """Simple function: f(x) = x^2."""
    return x * x  # x^2


def main() -> None:
    """Runs the DerivativeKit demo for f(x) = x^2."""
    x0 = 1.0
    dk = DerivativeKit(function=simple_function, x0=x0)

    # Numerical derivatives
    d1_ad = dk.differentiate(order=1)
    d2_ad = dk.differentiate(order=2)
    d1_fd = dk.differentiate(method="finite", order=1)
    d2_fd = dk.differentiate(method="finite", order=2)

    # Analytic values
    d1_true = 2.0 * x0
    d2_true = 2.0

    # Print results
    print(f"f(x) = x^2 at x0 = {x0:.4f}\n")
    print("Method       f'(x0)        f''(x0)")
    print("------------------------------------")
    print(f"Analytic   : {d1_true:>12.6f}   {d2_true:>12.6f}")
    print(f"Adaptive   : {d1_ad:>12.6f}   {d2_ad:>12.6f}")
    print(f"Finite     : {d1_fd:>12.6f}   {d2_fd:>12.6f}")

    print("Done. DerivKit — the joy of change, expressed precisely.")

if __name__ == "__main__":
    main()
