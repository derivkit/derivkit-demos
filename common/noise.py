from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Mapping, Optional
from functools import partial


import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.figure as mpl_figure

__all__ = [
    "add_gaussian_noise",
    "make_noisy_function",
    "random_generator",
]


def add_gaussian_noise(
    y: ArrayLike,
    sigma: float | ArrayLike,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Add element-wise Gaussian noise to an array-like input.

    The function treats the input ``y`` as the clean data and adds random noise
    drawn from a normal distribution with mean zero and standard deviation
    ``sigma``. The output keeps the same shape as the input and is always
    returned as ``float64``.

    Args:
        y: Array-like clean data to which noise will be added.
        sigma: Standard deviation of the Gaussian noise. Can be a scalar or an
            array broadcastable to the shape of ``y``.
        rng: NumPy random number generator used to draw noise samples.

    Returns:
        A NumPy array with Gaussian noise added element-wise to ``y``.

    Example:
        >>> rng = random_generator(0)
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> noisy = add_gaussian_noise(data, sigma=0.1, rng=rng)
        >>> noisy.shape
        (3,)
    """
    y_arr = np.asarray(y, float)
    sig_arr = np.asarray(sigma, float)
    noise = rng.normal(0.0, sig_arr, size=y_arr.shape)
    return (y_arr + noise).astype(np.float64, copy=False)


def _noisy_eval(
    x: float | np.ndarray,
    *,
    func: Callable[..., np.ndarray],
    sigma: float | np.ndarray,
    rng: np.random.Generator,
    f_args: tuple[Any, ...],
    f_kwargs: dict[str, Any],
) -> np.ndarray:
    """Top-level helper used by make_noisy_function (kept public-free)."""
    x = np.asarray(x, float)
    y = func(x, *f_args, **f_kwargs)
    return add_gaussian_noise(y, sigma=sigma, rng=rng)


def make_noisy_function(
    func: Callable[..., np.ndarray],
    sigma: float | np.ndarray,
    rng: np.random.Generator,
    *f_args: Any,
    **kwargs: Any,
) -> Callable[[float | np.ndarray], np.ndarray]:
    """Returns a noisy wrapper around a clean function.

    The returned function takes a single argument `x` (float or array-like) and
    calls `func(x, *f_args, **f_kwargs)`, adding Gaussian noise with standard
    deviation `sigma` to the output.

    Args:
        func:
            The clean function to wrap. It should accept `x` as its first argument,
            followed by any additional positional and keyword arguments.
        sigma:
            Standard deviation of the Gaussian noise to add to the output.
        rng: np.random.Generator
            Random number generator used to draw noise samples.
        *f_args:
            Additional positional arguments to pass to `func`.
        **kwargs:
            Additional keyword arguments to pass to `func`.

    Returns:
            A function that takes `x` and returns the noisy evaluation.
    """
    # Allow passing args both positionally and via keyword f_args=(...)
    extra_f_args = kwargs.pop("f_args", ())
    if extra_f_args:
        if isinstance(extra_f_args, (list, tuple)):
            f_args = tuple(f_args) + tuple(extra_f_args)
        else:
            f_args = tuple(f_args) + (extra_f_args,)

    # Allow explicit f_kwargs dict; any remaining kwargs are treated as model kwargs too
    f_kwargs = kwargs.pop("f_kwargs", None)
    if f_kwargs is None:
        f_kwargs = {}
    if kwargs:
        for k, v in kwargs.items():
            f_kwargs.setdefault(k, v)

    return partial(
        _noisy_eval,
        func=func,
        sigma=np.asarray(sigma, float),
        rng=rng,
        f_args=tuple(f_args),
        f_kwargs=dict(f_kwargs),
    )


def random_generator(seed: int | None = None) -> np.random.Generator:
    """Create a reproducible NumPy random number generator.

    This function wraps ``np.random.default_rng`` and ensures that the input
    seed (if given) is cast to an integer. It is mainly used to provide a
    consistent interface for generating independent random streams across
    Monte Carlo draws.

    Args:
        seed: Optional integer seed for reproducibility. If ``None``, a random
            seed will be drawn from system entropy.

    Returns:
        A NumPy ``Generator`` instance that can be used to produce random draws.

    Example:
        >>> rng = random_generator(42)
        >>> rng.normal(size=3).shape
        (3,)
    """
    return np.random.default_rng(None if seed is None else int(seed))
