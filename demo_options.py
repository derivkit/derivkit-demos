#!/usr/bin/env python3
"""
Adaptive-fit demo using DerivKit (minimal, new API):

A) default spacing (spacing="auto" or a float half-width) built around x0
B) user-defined spacing via grid=("absolute", x_abs)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---- DerivKit ----
from derivkit.derivative_kit import DerivativeKit

# ---- optional local style (graceful fallback) ----
try:
    from scripts.plot_style import (
        apply_plot_style,
        DEFAULT_LINEWIDTH,
        DEFAULT_COLORS,
        MID_GRAY,
    )
except Exception:

    def apply_plot_style():
        pass

    DEFAULT_LINEWIDTH = 1.6
    MID_GRAY = "#666666"
    DEFAULT_COLORS = {
        "adaptive": "#3b82f6",
        "adaptive_lite": "#60a5fa",
        "forecasts": "#10b981",
        "samples": "#10b981",  # alias if your palette lacks "samples"
        "central": "#a855f7",
        "x0": "#a855f7",
        "red": "#dc2626",
        "danger": "#dc2626",
    }

# --- robust color aliases (works with any palette keys) ---
SAMPLES_COLOR = DEFAULT_COLORS.get("samples", DEFAULT_COLORS.get("forecasts", "#10b981"))
ADAPTIVE_COLOR = DEFAULT_COLORS.get("adaptive", "#3b82f6")
CENTRAL_RED = DEFAULT_COLORS.get("danger", DEFAULT_COLORS.get("red", "#dc2626"))


# ---------- target & noise ----------
def f_clean(x, a=1.7, b=-0.2):
    x = np.asarray(x, float)
    return a * x + b  # true slope is 'a'


def make_noisy(f, sigma=0.02, seed=42):
    rng = np.random.default_rng(seed)

    def g(x):
        x = np.asarray(x, float)
        return f(x) + rng.normal(0.0, sigma, size=x.shape)

    return g


# ---------- DerivKit helpers (NEW API) ----------
def slope_default(f, x0, order=1, *, n_points=25, spacing="auto", base_abs=1e-3):
    """
    Derivative with DerivKit's default/adaptive spacing (new API).
    """
    dk = DerivativeKit(f, x0)
    val = dk.adaptive.differentiate(
        order,
        n_points=int(n_points),
        spacing=spacing,     # "auto" | float | "2%"
        base_abs=base_abs,   # floor for small-x regions (useful near x0≈0)
        # domain=None, ridge=0.0, diagnostics=False ...
    )
    return float(np.asarray(val).ravel()[0])


def slope_user_grid(f, x0, order=1, *, x_abs):
    """
    Derivative on a user-provided ABSOLUTE x grid.
    Use grid=("absolute", x_abs) per new API.
    """
    x_abs = np.asarray(x_abs, float)
    dk = DerivativeKit(f, x0)
    val = dk.adaptive.differentiate(
        order,
        grid=("absolute", x_abs),  # n_points ignored when grid is provided
    )
    return float(np.asarray(val).ravel()[0])


# ---------- formatting helper: value ± error with smart rounding ----------
def format_pm(val: float, err: float) -> str:
    """
    Format 'val ± err' choosing decimals based on the error magnitude.
    Uses 1 significant figure for the error (2 if leading digit is 1),
    and rounds the value to the same decimal place.
    """
    err = abs(float(err))
    if err == 0 or not np.isfinite(err):
        return f"{val:.6g} ± 0"
    leading = int(f"{err:.1e}".split("e")[0].replace(".", "").lstrip("0")[:1] or "1")
    sig_err = 2 if leading == 1 else 1
    exp = int(np.floor(np.log10(err)))
    decimals = max(0, -(exp) + (sig_err - 1))
    val_r = round(val, decimals)
    err_r = round(err, decimals)
    fmt = f"{{:.{decimals}f}} ± {{:.{decimals}f}}"
    return fmt.format(val_r, err_r)


def main():
    apply_plot_style()

    # --- knobs (wider spacing + a few more points) ---
    x0 = 0.30
    n_points = 25  # Chebyshev cap in this branch is ~30
    noise = 0.01
    order = 1
    true_a, true_b = 1.7, -0.2

    # use a *wider* default half-width for the adaptive grid to improve conditioning
    DEFAULT_SPACING = 0.30  # absolute half-width around x0
    BASE_ABS = 1e-3         # floor near small x0 (kept same)

    # noisy linear target
    f = make_noisy(lambda x: f_clean(x, a=true_a, b=true_b), sigma=noise, seed=42)

    # ----- build grids for plotting & user-defined case -----
    delta_vis = 0.01
    W_plot = (n_points - 1) * delta_vis
    xx = np.linspace(x0 - W_plot, x0 + W_plot, 800)

    # Symmetric custom grid with tighter points near x0, but WIDER overall and MORE nodes
    # (tanh gives central crowding, endpoints reach ±DEFAULT_SPACING)
    t_custom = DEFAULT_SPACING * np.tanh(np.linspace(-2.0, 2.0, n_points)).astype(float)
    x_custom = x0 + t_custom

    # ----- compute derivatives (use wider spacing for default) -----
    d_default = slope_default(
        f, x0, order=order, n_points=n_points, spacing=DEFAULT_SPACING, base_abs=BASE_ABS
    )
    d_usergrid = slope_user_grid(f, x0, order=order, x_abs=x_custom)

    true_slope = true_a

    # pseudo-uncertainty for demo (distance to truth; for real work use bootstrap/diagnostics)
    e_default = abs(d_default - true_slope)
    e_usergrid = abs(d_usergrid - true_slope)

    # ----- samples for visualization only -----
    x_vis = np.linspace(x0 - W_plot, x0 + W_plot, n_points)
    y_vis = f(x_vis)
    y_custom = f(x_custom)

    # ----- plotting -----
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 5.0), constrained_layout=True)

    # Panel A: default spacing (wider half-width)
    ax = axs[0]
    ax.plot(
        xx,
        f_clean(xx, a=true_a, b=true_b),
        "--",
        lw=DEFAULT_LINEWIDTH,
        color=MID_GRAY,
        label="true $f(x)$",
    )
    ax.scatter(x_vis, y_vis, s=80, color=SAMPLES_COLOR, label="samples (viz)", zorder=3)
    ax.scatter([x0], [f(x0)], s=120, color=CENTRAL_RED, edgecolor="none", label="$x_0$", zorder=4)
    ax.plot(
        xx,
        f(x0) + d_default * (xx - x0),
        lw=DEFAULT_LINEWIDTH + 0.6,
        color=ADAPTIVE_COLOR,
        label="slope @ $x_0$",
    )
    ax.set_title("A · default spacing (wider half-width)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.text(
        0.02,
        0.96,
        f"true slope ≈ {true_slope:.3f}\nest. = {format_pm(d_default, e_default)}",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(frameon=True, fontsize=9, loc="lower right")

    # Panel B: user-defined absolute grid
    ax = axs[1]
    ax.plot(
        xx,
        f_clean(xx, a=true_a, b=true_b),
        "--",
        lw=DEFAULT_LINEWIDTH,
        color=MID_GRAY,
        label="true $f(x)$",
    )
    ax.scatter(x_custom, y_custom, s=90, color=SAMPLES_COLOR, label="user grid samples", zorder=3)
    ax.scatter([x0], [f(x0)], s=120, color=CENTRAL_RED, edgecolor="none", label="$x_0$", zorder=4)
    ax.plot(
        xx,
        f(x0) + d_usergrid * (xx - x0),
        lw=DEFAULT_LINEWIDTH + 0.6,
        color=ADAPTIVE_COLOR,
        label="slope @ $x_0$",
    )
    ax.set_title("B · user-defined grid (absolute x)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.text(
        0.02,
        0.96,
        f"true slope ≈ {true_slope:.3f}\nest. = {format_pm(d_usergrid, e_usergrid)}",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(frameon=True, fontsize=9, loc="lower right")

    # save & print
    outdir = Path("plots")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "adaptive_fit_minimal.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {outfile}")
    print(f"\nA · default spacing:    {format_pm(d_default, e_default)}")
    print(f"B · user-defined grid:  {format_pm(d_usergrid, e_usergrid)}")


if __name__ == "__main__":
    main()
