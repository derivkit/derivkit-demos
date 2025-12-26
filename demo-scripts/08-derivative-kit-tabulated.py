"""DerivKit — Tabulated Derivatives Demo.

Short summary:
    Differentiates a known function using DerivativeKit in *tabulated mode*
    from a *noisy* (x_tab, y_tab) table and compares:
      - truth (analytic)
      - finite difference + Ridders extrapolation
      - adaptive fit

Functions
---------
- f(x) = sin(x)
- f'(x) = cos(x)

What it does
------------
1) Creates a noisy tabulated table on a uniform x-grid (noise in y; optional jitter in x).
   This way we emulate real case scenario tabulated data with measurement errors.
2) Evaluates dy/dx at many x0 points using:
     - finite differences + Ridders
     - adaptive fit
3) Produces three plots:
   - noisy function table vs truth
   - derivative estimates (with internal errors when supported) vs truth
   - internal error vs x
4) Prints RMSE vs truth.

Usage
-----
    $ python demo-scripts/08-derivative-kit-tabulated.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Make repo root importable so `utils.style` works when running from demo-scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derivkit.derivative_kit import DerivativeKit
from derivkit.tabulated_model.one_d import Tabulated1DModel
from utils.style import DEFAULT_COLORS, apply_plot_style


def sin_func(x: np.ndarray) -> np.ndarray:
    """Returns sin(x).

    Args:
        x: input array.

    Returns:

    """
    return np.sin(x)


def df_truth(x: np.ndarray) -> np.ndarray:
    """Returns the analytic derivative cos(x).

    Args:
        x: input array.

    Returns:
        Derivative array.
    """
    return np.cos(x)


def ensure_plots_dir() -> Path:
    """Ensures the plots/ directory exists and returns its Path.

    Args:
        None.

    Returns:
        Path to plots/ directory.
    """
    outdir = Path("plots")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Colmputes the root mean square error (RMSE), mean absolute error (MAE),
    and maximum absolute error (MaxAE) between estimate and truth, ignoring NaNs.

    RMSE is computed as sqrt(mean((estimate - truth)^2)).
    MAE is computed as mean(|estimate - truth|).
    MaxAE is computed as max(|estimate - truth|).

    Args:
        estimate: estimated array.
        truth: ground truth array.

    Returns:
        Dictionary with keys "rmse", "mae", "maxae".
    """
    m = np.isfinite(estimate) & np.isfinite(truth)
    if not np.any(m):
        return {"rmse": float("nan"), "mae": float("nan"), "maxae": float("nan")}
    err = estimate[m] - truth[m]
    ae = np.abs(err)
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(ae)),
        "maxae": float(np.max(ae)),
    }

def tex_sci(x: float, sig: int = 2) -> str:
    """Return TeX string like 1.23\\times 10^{4} (or 0).

    Args:
        x: input float number.
        sig: number of significant digits in mantissa.

    Returns:
        TeX-formatted scientific notation string.
    """
    if not np.isfinite(x) or x == 0.0:
        return r"0"

    sgn = "-" if x < 0 else ""
    ax = abs(x)

    exp = int(np.floor(np.log10(ax)))
    mant = ax / (10.0 ** exp)

    # If exponent is 0, don't show ×10^{0}
    if exp == 0:
        return rf"{sgn}{mant:.{sig}f}"

    return rf"{sgn}{mant:.{sig}f}\times10^{{{exp}}}"


def compute_method_over_grid(
    x0_grid: np.ndarray,
    model: Tabulated1DModel,
    method_name: str,
    **extra_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute derivative at each x0 by looping over points.

    Args:
        x0_grid: array of x0 points where to compute the derivative.
        model: Tabulated1DModel instance to differentiate.
        method_name: derivative method name (e.g., "finite", "adaptive").
        extra_kwargs: extra kwargs to pass to differentiate().

    Returns:
        slopes, errs arrays of shape (N,). If return_error is unsupported,
        errs will be NaN.
    """
    slopes = np.empty_like(x0_grid, dtype=float)
    errs = np.full_like(x0_grid, np.nan, dtype=float)

    for i, x0 in enumerate(x0_grid):
        dk = DerivativeKit(function=model, x0=float(x0))
        try:
            out = dk.differentiate(
                method=method_name,
                order=1,
                return_error=True,
                **extra_kwargs,
            )
            val, err = out
            slopes[i] = np.asarray(val).reshape(-1)[0]
            errs[i] = np.asarray(err).reshape(-1)[0]
        except TypeError:
            try:
                val = dk.differentiate(
                    method=method_name,
                    order=1,
                    **extra_kwargs,
                )
                slopes[i] = np.asarray(val).reshape(-1)[0]
                errs[i] = np.nan
            except Exception as exc2:
                print(f"[warning] {method_name} failed at i={i} (x0={x0}): {exc2}")
                slopes[i] = np.nan
                errs[i] = np.nan
        except Exception as exc:
            print(f"[warning] {method_name} failed at i={i} (x0={x0}): {exc}")
            slopes[i] = np.nan
            errs[i] = np.nan

    return slopes, errs


def main() -> None:
    """Plots tabulated derivative estimates from noisy data."""
    apply_plot_style()

    rng = np.random.default_rng(42)

    # Here we build the noisy tabulated model
    n_tab = 70  # number of tabulated points
    x_tab = np.linspace(0.0, 2.0 * np.pi, n_tab)
    y_clean = sin_func(x_tab)

    y_noise_sigma = 0.05  # noise level in y
    x_jitter_sigma = 0  # noise level in x; set to 0 to disable x-jitter

    y_noisy = y_clean + rng.normal(0.0, y_noise_sigma, size=y_clean.shape)

    if x_jitter_sigma > 0:
        x_noisy = x_tab + rng.normal(0.0, x_jitter_sigma, size=x_tab.shape)
        srt = np.argsort(x_noisy)
        x_noisy = x_noisy[srt]
        y_noisy = y_noisy[srt]
    else:
        x_noisy = x_tab.copy()

    model_noisy = Tabulated1DModel(x_noisy, y_noisy, extrapolate=True)

    # Derivative evaluation grid (interior)
    n_eval = 50  # number of derivative eval points
    x0 = np.linspace(x_tab.min() + 0.15, x_tab.max() - 0.15, n_eval)  # avoid edges
    truth = df_truth(x0)

    # Compute: finite+Ridders and adaptive on noisy table
    slopes_fr, errs_fr = compute_method_over_grid(
        x0, model_noisy, "finite", extrapolation="ridders"
    )
    slopes_ad, errs_ad = compute_method_over_grid(
        x0, model_noisy, "adaptive", n_points=27, spacing=0.25
    )

    m_fr = metrics(slopes_fr, truth)
    m_ad = metrics(slopes_ad, truth)

    print("\nDerivative accuracy vs truth:")
    print(f"finite (Ridders): RMSE={m_fr['rmse']:.3e}  MAE={m_fr['mae']:.3e}  MaxAE={m_fr['maxae']:.3e}")
    print(f"adaptive : RMSE={m_ad['rmse']:.3e}  MAE={m_ad['mae']:.3e}  MaxAE={m_ad['maxae']:.3e}")

    # Plotting
    outdir = ensure_plots_dir()
    base = "tabulated_derivatives"

    # User-specified colors:
    c_truth = DEFAULT_COLORS["red"]
    c_finite = DEFAULT_COLORS["yellow"]
    c_adaptive = DEFAULT_COLORS["blue"]

    # Common "open circle" marker kwargs
    oc_finite = dict(marker="o", mfc="none", mec=c_finite, mew=1.2)
    oc_adapt = dict(marker="o", mfc="none", mec=c_adaptive, mew=1.2)
    oc_table = dict(marker="o", mfc="none", mec=c_finite, mew=1.2)

    # Plot 1: noisy function table vs truth
    x_dense = np.linspace(x_tab.min(), x_tab.max(), 800)  # dense grid for truth
    y_dense = sin_func(x_dense)  # truth values on a dense grid

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x_dense, y_dense, lw=2.0, label="truth sin(x)", color=c_truth)
    ax.plot(
        x_noisy,
        y_noisy,
        linestyle="--",
        linewidth=1.,
        ms=5.0,
        label=rf"$y=\sin(x)+\mathcal{{N}}(0,{y_noise_sigma:g}^2)$",
        color=c_finite,
        **oc_table,
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("Exact function and noisy samples (Gaussian jitter)")
    ax.legend(frameon=False, ncol=1, loc=1, fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / f"{base}_function.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot 2: derivative estimates with internal error bars
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x0, truth, lw=2.5, label="truth $\cos(x)$", color=c_truth)

    ax.errorbar(
        x0,
        slopes_fr,
        yerr=errs_fr,
        linestyle="--",
        linewidth=1.,
        capsize=2,
        label="finite (Ridders)",
        color=c_finite,
        markersize=5,
        **oc_finite,
    )

    ax.errorbar(
        x0,
        slopes_ad,
        yerr=errs_ad,
        linestyle="--",
        linewidth=1.,
        capsize=2,
        label="adaptive",
        color=c_adaptive,
        markersize=5,
        **oc_adapt,
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$f'(x)$")
    ax.set_title("Noisy tabulated derivative estimates vs truth")
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(outdir / f"{base}_derivative_errorbars.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot 3: internal error vs x
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Build legend labels with TeX math (multiline, no 'e' notation)
    lab_fr = (
            r"$\mathrm{finite\ (Ridders)}:$"
            "\n"
            + rf"$\mathrm{{RMSE}}={tex_sci(m_fr['rmse'])}$"
              "\n"
            + rf"$\mathrm{{MAE}}={tex_sci(m_fr['mae'])}$"
    )

    lab_ad = (
            r"$\mathrm{adaptive}:$"
            "\n"
            + rf"$\mathrm{{RMSE}}={tex_sci(m_ad['rmse'])}$"
              "\n"
            + rf"$\mathrm{{MAE}}={tex_sci(m_ad['mae'])}$"
    )

    ax.semilogy(
        x0,
        errs_fr,
        linestyle="--",
        linewidth=1.0,
        label=lab_fr,
        color=c_finite,
        markersize=5,
        **oc_finite,
    )
    ax.semilogy(
        x0,
        errs_ad,
        linestyle="--",
        linewidth=1.0,
        label=lab_ad,
        color=c_adaptive,
        markersize=5,
        **oc_adapt,
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\widehat{\sigma}_{f'}(x)$")  # or keep "internal error estimate"
    ax.set_title("Derivative internal error vs noisy tabulated data")
    ax.legend(frameon=True, ncol=1, fontsize=12, loc=4)
    fig.tight_layout()
    fig.savefig(outdir / f"{base}_internal_error.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
