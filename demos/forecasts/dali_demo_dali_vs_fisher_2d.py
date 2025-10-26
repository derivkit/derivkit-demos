#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact vs Fisher vs DALI — closed 1σ/2σ contours using Matplotlib's contour on a global grid.

We evaluate three log-likelihood surfaces on the *same* θ-grid:
  - Exact (via the model)
  - DALI (F, G, H tensors)
  - Fisher quadratic (F only)

No analytic ellipse; contours come purely from matplotlib.contour.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from derivkit.forecast_kit import ForecastKit
from common.style import apply_plot_style, DEFAULT_COLORS, DEFAULT_LINEWIDTH
from common.utils import resolve_outdir, save_fig

# Style / palette
blue = DEFAULT_COLORS["blue"]
yellow = DEFAULT_COLORS.get("yellow", "#F2C94C")
red = DEFAULT_COLORS["red"]

# User toggles
SHOW_1SIGMA = True    # Δχ² = 2.30 (2D)
SHOW_2SIGMA = True    # Δχ² = 6.17 (2D)
SHOW_EXACT = False
GRID_N = 500  # global θ-grid resolution per axis (bump if still jaggy)
BOX_MARGIN = 5.0  # how many semi-axes beyond Fisher’s 1σ to include in each dir


def sigma_suffix():
    if SHOW_1SIGMA and SHOW_2SIGMA: return "1and2sigma"
    if SHOW_1SIGMA: return "1sigma"
    if SHOW_2SIGMA: return "2sigma"
    return "nosigma"


# Model & likelihoods
def model_2d(theta_list):
    """
    theta = [θ1, θ2] ≡ [x, eps]
      o1 = (1 + eps) * 100 * exp(x^2)
      o2 = (1 + 0.3*eps) *  40 * exp(0.5*x)
    """
    x, eps = float(theta_list[0]), float(theta_list[1])
    o1 = (1.0 + eps)       * (1e2 * np.exp(x**2))
    o2 = (1.0 + 0.3 * eps) * (4e1 * np.exp(0.5 * x))
    return np.array([o1, o2])


def loglike_exact(cov: np.ndarray, theta0: np.ndarray, theta: np.ndarray) -> float:
    o = model_2d(theta)
    o0 = model_2d(theta0)
    r = o - o0
    Ci = np.linalg.pinv(cov)
    return -0.5 * float(r @ (Ci @ r))


def loglike_fisher(F: np.ndarray, dtheta: np.ndarray) -> float:
    return -0.5 * float(dtheta @ F @ dtheta)


def loglike_dali(F: np.ndarray, G: np.ndarray, H: np.ndarray, dtheta: np.ndarray) -> float:
    q = -0.5   * float(dtheta @ F @ dtheta)
    cubic = -0.5   * float(np.einsum('ijk,i,j,k',   G, dtheta, dtheta, dtheta))
    quart = -0.125 * float(np.einsum('ijkl,i,j,k,l', H, dtheta, dtheta, dtheta, dtheta))
    return q + cubic + quart


def main():
    apply_plot_style(base=blue)

    # Fiducial and observable covariance
    theta0 = np.array([0.10, 0.0])  # [θ1, θ2]
    covmat = np.diag([1.0, 1.0])  # 2 observables, unit cov

    # Build tensors once
    fk = ForecastKit(model_2d, theta0, covmat)
    F = fk.fisher()  # (2,2)
    G, H = fk.dali()  # (2,2,2), (2,2,2,2)

    # Exact max value at center and Δχ² levels
    L0 = loglike_exact(covmat, theta0, theta0)
    levels = []
    if SHOW_1SIGMA: levels.append(2.30)
    if SHOW_2SIGMA: levels.append(6.17)

    # --- Build one big θ-grid based on Fisher 1σ ellipse (just for box sizing) ---
    covariance = np.linalg.pinv(F)
    evals, evecs = np.linalg.eigh(covariance)
    evals = np.clip(evals, 0.0, None)
    # semi-axes for Δχ²=2.30 (1σ in 2D)
    r = np.sqrt(2.30 * np.maximum(evals, 0.0))
    # corners of an oriented rectangle that bounds the ellipse, then expand by BOX_MARGIN
    # We bound in lab frame by sampling unit circle directions; simple and robust:
    t = np.linspace(0, 2*np.pi, 720, endpoint=False)
    circ = np.vstack([np.cos(t), np.sin(t)])               # (2, m)
    ell  = (evecs @ (r[:, None] * circ))                  # (2, m)
    x_min, x_max = ell[0].min(), ell[0].max()
    y_min, y_max = ell[1].min(), ell[1].max()
    # margin factor
    x_half = BOX_MARGIN * max(abs(x_min), abs(x_max))
    y_half = BOX_MARGIN * max(abs(y_min), abs(y_max))

    x = np.linspace(theta0[0] - x_half, theta0[0] + x_half, GRID_N)
    y = np.linspace(theta0[1] - y_half, theta0[1] + y_half, GRID_N)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Flatten to evaluate quickly
    TH = np.stack([X.ravel(), Y.ravel()], axis=1)
    dTH = TH - theta0[None, :]

    # Compute fields
    # Exact
    Z_exact = np.empty(TH.shape[0], dtype=float)
    for i, th in enumerate(TH):
        Z_exact[i] = loglike_exact(covmat, theta0, th)
    Z_exact = Z_exact.reshape(X.shape)

    # DALI
    Z_dali = np.array([loglike_dali(F, G, H, dt) for dt in dTH], float).reshape(X.shape)

    # Fisher
    Z_fish = np.array([loglike_fisher(F, dt) for dt in dTH], float).reshape(X.shape)

    # Δχ² list is [2.30, 6.17] (1σ, 2σ). Convert to logL levels and sort ascending.
    L_levels = np.array([L0 - 0.5 * d for d in levels], dtype=float)
    L_levels = np.unique(L_levels)  # remove accidental duplicates
    L_levels.sort()  # must be strictly increasing for contour in matplotlib

    lws = [DEFAULT_LINEWIDTH, DEFAULT_LINEWIDTH]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))

    # Fisher (yellow)
    cs_fish  = ax.contour(X, Y, Z_fish,  levels=L_levels,
                          colors=[yellow, yellow], linestyles=["-", "-"],
                          linewidths=lws)

    # DALI (blue)
    cs_dali  = ax.contour(X, Y, Z_dali,  levels=L_levels,
                          colors=[blue, blue], linestyles=["-", "-"], linewidths=lws)

    if SHOW_EXACT:
        # Exact (red)
        cs_exact = ax.contour(X, Y, Z_exact, levels=L_levels,
                              colors=[red, red], linestyles=["--", "--"], linewidths=lws)

    # Fiducial
    ax.scatter([theta0[0]], [theta0[1]], marker="x", s=100, color=red, zorder=10)

    # Legend (fixed order)
    if SHOW_EXACT:
        handles = [
            Line2D([0],[0], color=red, lw=DEFAULT_LINEWIDTH, ls='-',  label="Exact"),
            Line2D([0],[0], color=blue, lw=DEFAULT_LINEWIDTH, ls='-.', label="Doublet DALI"),
            Line2D([0],[0], color=yellow, lw=DEFAULT_LINEWIDTH, ls='-', label="Fisher"),
            Line2D([0],[0], color=red, marker='x', s=100, label="Fiducial"),
        ]
    else:
        handles = [
            Line2D([0],[0], color=blue, lw=DEFAULT_LINEWIDTH, ls='-', label="Doublet DALI"),
            Line2D([0],[0], color=yellow, lw=DEFAULT_LINEWIDTH, ls='-', label="Fisher"),
            Line2D([0],[0], color=red, lw=0, marker='x', ms=10, label="Fiducial"),
        ]
    ax.legend(handles=handles, fontsize=12, framealpha=1.0, loc="best")

    ax.set_xlabel(r"parameter $\theta_1$", fontsize=18)
    ax.set_ylabel(r"parameter $\theta_2$", fontsize=18)
    ax.tick_params(labelsize=13)
    ax.minorticks_off()

    #ax.set_xlim(x.min(), x.max())
    #ax.set_ylim(y.min(), y.max())
    ax.set_xlim(-0.2, 0.4)
    ax.set_ylim(-0.075, 0.05)
    # lock equal scaling strictly on the data, not on figure space
    #ax.set_aspect('equal', adjustable='box')
    #ax.autoscale(enable=True, tight=True)

    outdir = resolve_outdir(None, file=__file__, default_rel="../plots")
    outfile = save_fig(fig, outdir, stem=f"dali_vs_fisher_2d_{sigma_suffix()}", ext="png")
    plt.show()
    print("saved:", outfile)

if __name__ == "__main__":
    main()
