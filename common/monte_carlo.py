"""Monte Carlo helpers for simple slope estimation and 2D HPD contours."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from common.utils import add_gaussian_noise, random_generator

__all__ = [
    "tiny_mcmc_slope",
    "tiny_mcmc_draws_2d",
    "hpd_density_levels",
]


def tiny_mcmc_slope(
    noiseless_function: Callable[[ArrayLike], NDArray[np.float64]],
    estimator: Callable[[Callable[[ArrayLike], NDArray[np.float64]], float, np.random.Generator], float],
    draws: int,
    seed: int,
    sigma: float,
) -> Tuple[float, float]:
    """Estimate mean and standard deviation of a slope via simple Monte Carlo.

    On each draw, the estimator receives the clean function, a noise level, and
    a fresh random generator. The estimator should evaluate the function at the
    points it needs, add Gaussian noise with `add_gaussian_noise`, compute a
    slope estimate, and return that scalar. This function aggregates the slope
    estimates and returns their mean and (population) standard deviation.

    Args:
        noiseless_function: Deterministic function mapping `x` to `y`.
        estimator: Callable with signature `(f_clean, sigma, rng) -> float` that
            returns a slope estimate for one noisy realization.
        draws: Number of Monte Carlo draws.
        seed: Random seed for reproducibility.
        sigma: Standard deviation of the additive Gaussian noise.

    Returns:
        Tuple `(mean, std)` of the slope estimates across all draws. `std` is
        computed with `ddof=0` (population standard deviation).
    """
    rng = random_generator(seed)
    estimates: list[float] = []
    for _ in range(int(draws)):
        child_rng = random_generator(int(rng.integers(1_000_000_000)))
        estimates.append(float(estimator(noiseless_function, sigma, child_rng)))
    arr = np.asarray(estimates, float)
    return float(arr.mean()), float(arr.std(ddof=0))


def tiny_mcmc_draws_2d(
    *,
    x0: float,
    f_clean: Callable[[ArrayLike], NDArray[np.float64]],
    estimator: Callable[[Callable[[ArrayLike], NDArray[np.float64]], float, np.random.Generator], float],
    draws: int,
    seed: int,
    sigma: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Draw slope samples and noisy values at a single point for 2D density plots.

    For each draw, this function:
      1. calls the estimator `(f_clean, sigma, rng)` to produce one slope sample,
      2. evaluates `f_clean(x0)` and adds Gaussian noise with the same draw's `rng`.

    This produces aligned arrays of slope samples and `y(x0)` samples, which you
    can bin into a 2D density and plot HPD contours with `hpd_density_levels`.

    Args:
        x0: Point at which to record the noisy function value per draw.
        f_clean: Deterministic function mapping `x` to `y`.
        estimator: Callable with signature `(f_clean, sigma, rng) -> float`.
        draws: Number of Monte Carlo draws.
        seed: Random seed for reproducibility.
        sigma: Standard deviation of the additive Gaussian noise.

    Returns:
        Tuple `(d_draws, y0_draws)`:
            - `d_draws`: array of slope samples, shape `(draws,)`.
            - `y0_draws`: array of `y(x0)` samples (noisy), shape `(draws,)`.
    """
    rng = random_generator(seed)
    d_list: list[float] = []
    y0_list: list[float] = []

    for _ in range(int(draws)):
        child_rng = random_generator(int(rng.integers(1_000_000_000)))
        d_list.append(float(estimator(f_clean, sigma, child_rng)))

        y0_clean = f_clean(np.asarray(x0, float))
        y0_noisy = add_gaussian_noise(y0_clean, sigma, child_rng)
        y0_list.append(float(np.asarray(y0_noisy, float)))

    return np.asarray(d_list, float), np.asarray(y0_list, float)


def hpd_density_levels(
    h: NDArray[np.float64],
    d_edges: NDArray[np.float64],
    y_edges: NDArray[np.float64],
    masses: tuple[float, float] = (0.68, 0.95),
) -> list[float]:
    """Compute HPD (highest posterior density) thresholds for a 2D density grid.

    Given a 2D density array on a rectangular grid, this finds density levels
    `t_k` such that the set `{(d, y): h(d, y) >= t_k}` contains each requested
    probability mass. Levels are returned in the same units as `h`.

    The algorithm sorts grid cells by density (descending), accumulates total
    probability by summing `density * cell_area`, and records the density at
    which each target mass is first reached.

    Args:
        h: 2D array of non-negative densities with shape `(n_d, n_y)`.
        d_edges: Bin edges along the d-axis; length must be `n_d + 1`.
        y_edges: Bin edges along the y-axis; length must be `n_y + 1`.
        masses: Target probability masses in `(0, 1]`, e.g., `(0.68, 0.95)`.

    Returns:
        List of density thresholds corresponding to `masses` (same order).
    """
    dx = np.diff(d_edges)[:, None]
    dy = np.diff(y_edges)[None, :]
    area = dx * dy

    h_flat = h.ravel()
    a_flat = area.ravel()
    order = np.argsort(h_flat)[-1::-1]  # descending
    h_sorted = h_flat[order]
    a_sorted = a_flat[order]

    mass_cum = np.cumsum(h_sorted * a_sorted)
    total_mass = mass_cum[-1] if mass_cum.size else 1.0
    if total_mass <= 0 or not np.isfinite(total_mass):
        return []
    mass_cum /= total_mass

    levels: list[float] = []
    for m in masses:
        k = np.searchsorted(mass_cum, float(m), side="left")
        k = min(max(k, 0), len(h_sorted) - 1)
        levels.append(float(h_sorted[k]))
    return levels
