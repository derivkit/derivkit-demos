#!/usr/bin/env python
"""
Demonstrate derivkit fit-quality checks with good and bad scenarios.

Each case runs AdaptiveFitDerivative.differentiate(...) with settings chosen to:
- produce a healthy fit (no warnings),
- trigger Chebyshev cap error,
- trigger ill-conditioning / large residuals (narrow spacing),
- trigger instability via noise,
- trigger "not enough points" for explicit grid,
- exercise domain transforms (signed-log and sqrt) and print diagnostics.
"""

from __future__ import annotations

import numpy as np

from derivkit.adaptive.adaptive_fit import AdaptiveFitDerivative

# from derivkit.adaptive.transforms import signed_log_to_physical  # just to prove import works

np.set_printoptions(precision=5, suppress=True)
rng = np.random.default_rng(42)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------
def f_smooth(x: float | np.ndarray) -> np.ndarray:
    """Very smooth, well-behaved function: exp(x)*sin(x). Returns 1D array."""
    x = np.asarray(x, float)
    y = np.exp(x) * np.sin(x)
    return y if y.ndim else np.array([y])


def f_noisy(x: float | np.ndarray) -> np.ndarray:
    """Same as f_smooth but with small additive noise to stress the fit."""
    x = np.asarray(x, float)
    pure = np.exp(x) * np.sin(x)
    noise = 5e-4 * rng.standard_normal(size=pure.shape)
    y = pure + noise
    return y if y.ndim else np.array([y])


def f_sqrt(x: float | np.ndarray) -> np.ndarray:
    """Sqrt on [0, +inf), triggers sqrt pullback around x0=0 with domain=(0, None)."""
    x = np.asarray(x, float)
    # clamp negatives to NaN to make violations obvious if they occur
    y = np.where(x >= 0.0, np.sqrt(x), np.nan)
    return y if y.ndim else np.array([y])


# ---------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------
def case_good_smooth():
    banner("CASE 1: good fit on smooth function (should be healthy)")
    x0 = 0.2
    afd = AdaptiveFitDerivative(f_smooth, x0)
    d1, diag = afd.differentiate(
        1,
        n_points=15,
        spacing="2%",
        diagnostics=True,
        ridge=1e-8,
        domain=None,
    )
    print(f"Derivative (order=1) at x0={x0:.3f}: {d1:.8f}")
    # key metrics preview
    fq = diag.get("fit_quality", {})
    print("fit_quality:", {k: v for k, v in fq.items() if k != "thresholds"})


def case_cheb_cap_error():
    banner("CASE 2: too many points for default Chebyshev (should raise)")
    x0 = 0.0
    afd = AdaptiveFitDerivative(f_smooth, x0)
    try:
        afd.differentiate(
            1,
            n_points=40,  # > 30 → your cap
            spacing="auto",
            diagnostics=True,
        )
    except Exception as e:
        print("Expected error:", e)


def case_ill_conditioned():
    banner("CASE 3: ill-conditioned (narrow spacing, many points) — expect warnings")
    x0 = 0.0
    afd = AdaptiveFitDerivative(f_smooth, x0)
    # keep under the cap but very tight spacing to push cond #
    d1 = afd.differentiate(
        1,
        n_points=29,
        spacing=1e-9,  # super tight → Vandermonde cond skyrockets
        diagnostics=False,
        ridge=0.0,
    )
    print("Derivative (order=1):", d1)


def case_noisy_instability():
    banner("CASE 4: noisy data → large residuals / LOO — expect warnings")
    x0 = -0.3
    afd = AdaptiveFitDerivative(f_noisy, x0)
    d1, diag = afd.differentiate(
        1,
        n_points=17,
        spacing="1%",
        diagnostics=True,
        ridge=0.0,
    )
    print(f"Derivative (order=1) at x0={x0:.3f}: {d1:.8f}")
    print("suggestions:", diag.get("fit_suggestions"))


def case_not_enough_points_explicit_grid():
    banner("CASE 5: explicit offsets with too few points for order=3 (should raise)")
    x0 = 0.1
    afd = AdaptiveFitDerivative(f_smooth, x0)
    # order=3 ⇒ min_pts = 2*3+1 = 7 required; we only give 3 offsets (0 auto-inserted)
    t = np.array([-1e-3, 1e-3, 2e-3])
    try:
        afd.differentiate(
            3,
            grid=("offsets", t),
            diagnostics=False,
        )
    except Exception as e:
        print("Expected error:", e)


def case_signed_log_transform():
    banner("CASE 6: domain=(0, None) with x0 away from 0; symmetric grid would cross 0")
    x0 = 0.05
    afd = AdaptiveFitDerivative(f_smooth, x0)
    # Using a spacing that would normally dip negative forces the signed-log path.
    d1, diag = afd.differentiate(
        1,
        n_points=13,
        spacing=0.1,  # would place points < 0 without transform
        domain=(0.0, None),
        diagnostics=True,
    )
    print(f"Derivative (order=1) at x0={x0:.3f}: {d1:.8f}")
    print(
        "mode used:", diag["meta"]["mode"] if "meta" in diag else "<meta not attached>"
    )


def case_sqrt_transform_boundary():
    banner("CASE 7: sqrt-domain at boundary x0=0 with domain=(0, None)")
    x0 = 0.0
    afd = AdaptiveFitDerivative(f_sqrt, x0)
    d1, diag = afd.differentiate(
        1,
        n_points=11,
        spacing="auto",
        domain=(0.0, None),
        diagnostics=True,
    )
    print(f"Derivative (order=1) at x0={x0:.1f}: {d1:.8f}")
    print(
        "mode used:", diag["meta"]["mode"] if "meta" in diag else "<meta not attached>"
    )


def main():
    case_good_smooth()
    case_cheb_cap_error()
    case_ill_conditioned()
    case_noisy_instability()
    case_not_enough_points_explicit_grid()
    case_signed_log_transform()
    case_sqrt_transform_boundary()


if __name__ == "__main__":
    main()
