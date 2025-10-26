"""Common functions for adaptive demos."""

from __future__ import annotations
from typing import Callable, Any
import numpy as np

from derivkit.derivative_kit import DerivativeKit
from common.noise import make_noisy_function

__all__ = [
    "f_clean",
    "slope_estimator",
]


def f_clean(x: float | np.ndarray, a: float, b: float) -> np.ndarray:
    """Simple linear function f(x) = a*x + b."""
    x = np.asarray(x, float)
    return a * x + b


def slope_estimator(
    f_clean_fn: Callable[..., np.ndarray],
    x0: float,
    order: int,
    n_points: int,
    spacing: float,
    base_abs: float,
    ridge: float,
    sigma: float,
    rng: np.random.Generator,
    *f_args: Any,
    **f_kwargs: Any,
) -> float:
    """Estimate derivative at x0 using DerivKit's adaptive method on a noisy wrapper.

    Notes
    -----
    - `f_clean_fn` can be any callable f(x, *f_args, **f_kwargs).
    - Noise is injected only through the wrapper used by DerivKit; your plotting
      can use an independent noisy function if you like.
    """
    if f_kwargs is None:
        f_kwargs = {}
    g = make_noisy_function(f_clean_fn, sigma, rng, *f_args, **f_kwargs)
    dk = DerivativeKit(g, x0)
    val = dk.adaptive.differentiate(
        order,
        n_points=int(n_points),
        spacing=spacing,
        base_abs=base_abs,
        ridge=ridge,
    )
    return float(np.asarray(val).ravel()[0])