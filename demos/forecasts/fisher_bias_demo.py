#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot unfilled Gaussian 68%/95% ellipses for unbiased vs biased posteriors.

Pick ONE way to provide the bias:
  MODE = "DIRECT"      -> provide bias_vec (Δθ) directly
  MODE = "SCORE"       -> provide Fisher F and score vector g = J^T C^{-1} Δν
  MODE = "FORECASTKIT" -> provide cov, data_with/without, model(theta)->obs, theta0
                           (uses your ForecastKit to compute Δν and Fisher bias)

Ellipses are drawn from the parameter covariance C = F^{-1} (or pinv).
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from common.utils import resolve_outdir, save_fig
from common.style import apply_plot_style, DEFAULT_COLORS

def _apply_style(): apply_plot_style(base=DEFAULT_COLORS["blue"])
BLUE   = DEFAULT_COLORS["blue"]
YELLOW = DEFAULT_COLORS.get("yellow", "#F2C94C")
RED    = DEFAULT_COLORS["red"]

# ==== TOGGLES FOR WHICH CONTOURS TO DRAW ====
SHOW_1SIGMA = True   # 68% (1σ)
SHOW_2SIGMA = True   # 95% (2σ)

# ==== CHOOSE YOUR MODE ====
MODE = "DIRECT"        # "DIRECT", "SCORE", or "FORECASTKIT"

# ==== USER INPUTS ====================================
theta0 = np.array([0.10, 0.00])      # center of unbiased ellipse (P>=2)
F = np.array([[4.0, 1.2],
              [1.2, 2.5]])           # Fisher (P×P)

labels = [r"$\theta_0$", r"$\theta_1$"]

# DIRECT mode bias
bias_vec = np.array([0.06, -0.02])
bias_scale = 5.0
bias_vec = bias_scale * bias_vec

# SCORE mode score vector g = J^T C^{-1} Δν
g = np.array([0.30, -0.10])
g_scale = 4.0
g = g_scale * g
g = np.array([0.0, 1.0]) * np.linalg.norm(g)  # point along θ1

# FORECASTKIT mode inputs
cov = np.array([[1.0]])
data_without = np.array([100.0])
eps = 0.10
data_with = (1.0 + eps) * data_without
def model(theta):
    a, b = float(theta[0]), float(theta[1])
    return np.array([a*50.0 + b*10.0])

# ======================================================

def ellipse_points_2d(Cov2x2: np.ndarray, center: np.ndarray, delta_chi2: float, n: int = 720):
    Cov2x2 = np.asarray(Cov2x2, float)
    center = np.asarray(center, float).ravel()
    evals, evecs = np.linalg.eigh(Cov2x2)
    evals = np.clip(evals, 0.0, None)
    radii = np.sqrt(delta_chi2) * np.sqrt(evals)
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    circ = np.vstack([np.cos(t), np.sin(t)])
    pts = (evecs @ (radii[:, None] * circ)).T + center
    return pts

def compute_bias_DIRECT(F, bias_vec):
    return np.asarray(bias_vec, float).ravel()

def compute_bias_SCORE(F, g):
    F = np.asarray(F, float)
    g = np.asarray(g, float).ravel()
    return (np.linalg.pinv(F) @ g).ravel()

def compute_bias_FORECASTKIT(F, theta0, cov, data_with, data_without, model):
    from derivkit.forecast_kit import ForecastKit
    fk = ForecastKit(model, theta0, cov)
    F_here = F if F is not None else fk.fisher()
    delta_nu = fk.delta_nu(data_with=data_with, data_without=data_without)
    bias = fk.fisher_bias(fisher_matrix=F_here, delta_nu=delta_nu)
    return np.asarray(bias, float).ravel(), np.asarray(F_here, float)

def plot_ellipses(theta0, bias_vec, F, labels, show_1sigma=True, show_2sigma=True):
    _apply_style()
    theta0 = np.asarray(theta0, float).ravel()
    bias_vec = np.asarray(bias_vec, float).ravel()
    F = np.asarray(F, float)

    if theta0.size < 2 or F.shape[0] < 2:
        raise ValueError("Need at least 2 parameters (2×2 Fisher) to draw ellipses.")

    C = np.linalg.pinv(F)
    idx = (0, 1)
    C2 = C[np.ix_(idx, idx)]
    c0 = theta0[list(idx)]
    cb = (theta0 + bias_vec)[list(idx)]

    dchi2_68, dchi2_95 = 2.30, 6.17

    plotted_segments = []

    E95_u = E95_b = None
    if show_2sigma:
        E95_u = ellipse_points_2d(C2, c0, dchi2_95)
        E95_b = ellipse_points_2d(C2, cb, dchi2_95)

    E68_u = E68_b = None
    if show_1sigma:
        E68_u = ellipse_points_2d(C2, c0, dchi2_68)
        E68_b = ellipse_points_2d(C2, cb, dchi2_68)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    if show_2sigma:
        ax.plot(E95_u[:,0], E95_u[:,1], ls="--", lw=1.6, color=RED,  label=r"$2\sigma$ (unbiased)")
        ax.plot(E95_b[:,0], E95_b[:,1], ls="--", lw=1.6, color=BLUE, label=r"$2\sigma$ (biased)")
        plotted_segments += [E95_u, E95_b]

    if show_1sigma:
        ax.plot(E68_u[:,0], E68_u[:,1], ls="-",  lw=1.9, color=RED,  label=r"$1\sigma$ (unbiased)")
        ax.plot(E68_b[:,0], E68_b[:,1], ls="-",  lw=1.9, color=BLUE, label=r"$1\sigma$ (biased)")
        plotted_segments += [E68_u, E68_b]

    ax.scatter([c0[0]], [c0[1]], marker="x", s=90, color=RED,  linewidths=2.0, label="fiducial", zorder=3)
    ax.scatter([cb[0]], [cb[1]], marker="x", s=90, color=BLUE, linewidths=2.0, label="biased",   zorder=3)
    ax.annotate("", xy=(cb[0], cb[1]), xytext=(c0[0], c0[1]),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=YELLOW), zorder=4)

    # Auto-limits
    Xs, Ys = [c0[0], cb[0]], [c0[1], cb[1]]
    for seg in plotted_segments:
        Xs.extend([seg[:,0].min(), seg[:,0].max()])
        Ys.extend([seg[:,1].min(), seg[:,1].max()])
    Xs, Ys = np.array(Xs), np.array(Ys)
    xmid, ymid = Xs.mean(), Ys.mean()
    xr, yr = (Xs.max()-Xs.min()), (Ys.max()-Ys.min())
    pad = 0.22
    xr = xr if xr > 0 else 1.0
    yr = yr if yr > 0 else 1.0
    ax.set_xlim(xmid - (1+pad)*xr/2, xmid + (1+pad)*xr/2)
    ax.set_ylim(ymid - (1+pad)*yr/2, ymid + (1+pad)*yr/2)

    ax.set_xlabel(labels[0] if len(labels) > 0 else r"$\theta_0$", fontsize=16)
    ax.set_ylabel(labels[1] if len(labels) > 1 else r"$\theta_1$", fontsize=16)
    ax.set_title("Fisher bias demo", fontsize=17)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12, framealpha=1.0, loc="best")
    ax.minorticks_off()

    # -------- filename suffix based on contour toggles --------
    if show_1sigma and show_2sigma:
        sigma_suffix = "1and2sigma"
    elif show_1sigma:
        sigma_suffix = "1sigma"
    elif show_2sigma:
        sigma_suffix = "2sigma"
    else:
        sigma_suffix = "nosigma"

    outdir = resolve_outdir(None, file=__file__, default_rel="../plots")
    stem = f"fisher_bias_demo_{sigma_suffix}"
    outfile = save_fig(fig, outdir, stem=stem, ext="png")
    print("Saved figure:", outfile)

    plt.show()


def main():
    mode = MODE.upper().strip()

    if mode == "DIRECT":
        bv = compute_bias_DIRECT(F, bias_vec)
        plot_ellipses(theta0, bv, F, labels, show_1sigma=SHOW_1SIGMA, show_2sigma=SHOW_2SIGMA)

    elif mode == "SCORE":
        bv = compute_bias_SCORE(F, g)
        plot_ellipses(theta0, bv, F, labels, show_1sigma=SHOW_1SIGMA, show_2sigma=SHOW_2SIGMA)

    elif mode == "FORECASTKIT":
        from derivkit.forecast_kit import ForecastKit
        fk = ForecastKit(model, theta0, cov)
        F_fk = fk.fisher()
        bv, F_used = compute_bias_FORECASTKIT(F_fk, theta0, cov, data_with, data_without, model)
        plot_ellipses(theta0, bv, F_used, labels, show_1sigma=SHOW_1SIGMA, show_2sigma=SHOW_2SIGMA)

    else:
        raise ValueError("MODE must be one of: DIRECT, SCORE, FORECASTKIT")

if __name__ == "__main__":
    main()
