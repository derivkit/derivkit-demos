"""DALI vs Fisher vs Exact Likelihood."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from derivkit.forecast_kit import ForecastKit
from common.style import apply_plot_style, DEFAULT_COLORS
from common.utils import resolve_outdir, save_fig

# Colors
blue   = DEFAULT_COLORS["blue"]
yellow = DEFAULT_COLORS.get("yellow")
red    = DEFAULT_COLORS["red"]

# Model & likelihoods
# Define a model: one observable with a quadratic dependence on one model parameter.
def Model1d(paramList):
    x = paramList[0]
    obs = 1e2 * np.exp(x**2)
    return np.array([obs])

# Define a function which returns the exact likelihood.
def logLike1dExact(sigma_o, fiducial_x, x):
    delta_o = Model1d([x]) - Model1d([fiducial_x])
    logLike = 0
    logLike += -0.5 * (delta_o / sigma_o) ** 2
    return logLike

# Define a function which returns an approximate likelihood using the Fisher and doublet-DALI terms.
def logLike1dApprox(tensors, fiducial_x, x):
    delta_x = x - fiducial_x
    logLike = 0
    if len(tensors) >= 1:
        F = tensors[0]
        logLike += -0.5 * F[0][0] * delta_x**2
    if len(tensors) >= 3:
        G = tensors[1]
        H = tensors[2]
        logLike += -0.5 * G[0][0][0] * delta_x**3
        logLike += -0.125 * H[0][0][0][0] * delta_x**4
    return logLike

# -----------------------------
# Setup forecasting objects
# -----------------------------
observables = Model1d
fiducial_values = [0.1]
covmat = np.array([[1.0]])

# Initialize the forecasting utility using these inputs.
fk = ForecastKit(observables, fiducial_values, covmat)

# Create a list of all three tensors: Fisher, doublet-DALI G, and doublet-DALI H.
fisher_matrix = fk.fisher()
DALI_G, DALI_H = fk.dali()
tensors = [fisher_matrix, DALI_G, DALI_H]

# -----------------------------
# Grids & evaluations
# -----------------------------
fiducial_x = fiducial_values[0]
sigma_o = np.sqrt(covmat[0][0])

xgrid = np.linspace(-1, 1, 1000)
xgrid_sparse = np.linspace(-0.2, 0.2, 100)

exactLike = np.array([logLike1dExact(sigma_o, fiducial_x, x) for x in xgrid])
fisherLike = np.array([logLike1dApprox([fisher_matrix], fiducial_x, x) for x in xgrid])
doubletDALILike = np.array([logLike1dApprox(tensors, fiducial_x, x) for x in xgrid_sparse])

# -----------------------------
# Plot
# -----------------------------
def main():
    apply_plot_style(base=blue)  # your style; no grid changes needed

    plt.figure(figsize=(8, 5))

    # fisher
    plt.plot(
        xgrid,
        fisherLike,
        label="Fisher Matrix",
        color=yellow,
    )
    # dali (sparse markers for clarity)
    #plt.scatter(
    #    xgrid_sparse,
    #    doubletDALILike,
    #    label="Doublet DALI",
    #    facecolor="none",
    #    edgecolors=blue,
    #)
    plt.plot(
        xgrid_sparse,
        doubletDALILike,
        label="Doublet DALI",
        color=blue,
    )


    # exact
    plt.plot(xgrid, exactLike, label="Exact Likelihood", color=red, linestyle="--")

    plt.title(r"$\mathrm{observable}= 100 \cdot e^{x^2}$", fontsize=20)
    plt.xlabel(r"parameter value $x$", fontsize=20)
    plt.ylabel(r"log-likelihood $\mathrm{log}(P)$", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(-0.3, 0.6)
    plt.ylim(-0.65, 0.05)
    plt.legend(fontsize=16, framealpha=1.0)
    plt.minorticks_off()

    outdir = resolve_outdir(None, file=__file__, default_rel="../plots")
    outfile = save_fig(plt.gcf(), outdir, stem="dali_vs_fisher_exact_1d", ext="png")
    plt.show()
    print(f"saved: {outfile}")

if __name__ == "__main__":
    main()
