"""DerivKit — Tabulated Derivatives Demo (simple).

We build a noisy tabulated sin(x) table, then estimate f'(x0) at x0 = 0.7
with two methods using DerivativeKit in tabulated mode:
  - finite differences + Ridders extrapolation
  - adaptive fit

Underlying function (noise-free)
--------------------------------
    f(x) = sin(x)
    f'(x) = cos(x)

Tabulated data
--------------
    y_tab = sin(x_tab) + Normal(0, sigma^2)

What it does
------------
- Builds a noisy tabulated dataset with a fixed random seed (reproducible).
- Sets x0 = 0.7.
- Computes f'(x0) from the noisy table using:
  - finite differences + Ridders extrapolation
  - adaptive fit
- Prints estimates alongside the analytic derivative cos(x0) for comparison.

Usage
-----
    $ python demo-scripts/08-derivative-kit-tabulated-simple.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make repo root importable so DerivKit imports work when running from demo-scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derivkit.derivative_kit import DerivativeKit
from derivkit.derivatives.tabulated_model.one_d import Tabulated1DModel


def main() -> None:
    rng = np.random.default_rng(42)

    n_tab = 70
    x_tab = np.linspace(0.0, 2.0 * np.pi, n_tab)

    y_noise_sigma = 0.05  # a 5% noise level
    y_noisy = np.sin(x_tab) + rng.normal(0.0, y_noise_sigma, size=x_tab.shape)

    model = Tabulated1DModel(x_tab, y_noisy, extrapolate=True)

    x0 = 0.7  # point to differentiate at
    truth = float(np.cos(x0))  # analytic derivative at x0

    dk = DerivativeKit(function=model, x0=x0)

    d_fr = float(
        np.asarray(
            dk.differentiate(method="finite", order=1, extrapolation="ridders")
        ).reshape(-1)[0]
    )
    d_ad = float(
        np.asarray(
            dk.differentiate(method="adaptive", order=1, n_points=27, spacing=0.25)
        ).reshape(-1)[0]
    )

    e_fr = abs(d_fr - truth)
    e_ad = abs(d_ad - truth)

    print("DerivKit tabulated derivative demo (single point)")
    print(f"x0 = {x0:.3f}")
    print(f"truth  f'(x0)=cos(x0)      = {truth:+.6f}")
    print(f"finite (Ridders) estimate  = {d_fr:+.6f}   |err| = {e_fr:.3e}")
    print(f"adaptive estimate          = {d_ad:+.6f}   |err| = {e_ad:.3e}")
    print(f"winner (smaller |err|): {'adaptive' if e_ad < e_fr else 'finite (Ridders)'}")


if __name__ == "__main__":
    main()
