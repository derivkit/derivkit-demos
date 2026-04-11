"""DerivKit — ForecastKit Fisher Sampling Demo (GetDist).

This demo shows how to use :class:`ForecastKit` to compute a Fisher matrix
for a simple 2-parameter model and then generate GetDist samples from the
corresponding Gaussian approximation.

We work with a toy model:
    o1 = a
    o2 = b
    o3 = a + 2b

Steps
-----
- Define a model mapping parameters (a, b) to observables.
- Compute the Fisher matrix numerically using ForecastKit.
- Convert the Fisher matrix into GetDist MCSamples via Gaussian sampling.
- Plot the resulting 2D parameter contours.

Usage
-----
    python -m scripts.forecast-kit-fisher

Notes
-----
- The Fisher matrix defines a Gaussian approximation to the likelihood:
      L(θ) ∝ exp[-1/2 (θ - θ0)^T F (θ - θ0)]
- Sampling from this Gaussian gives contours equivalent to the analytic Fisher ellipses.
- This uses DerivKit’s Fisher → GetDist sampling utility rather than the analytic Gaussian wrapper.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from derivkit import ForecastKit
from getdist import plots as getdist_plots


# Define a simple toy model
def model(theta):
    a, b = theta
    return np.array([a, b, a + 2.0 * b], dtype=float)


# Fiducial parameters and covariance
theta0 = np.array([1.0, 2.0], dtype=float)
cov = np.eye(3, dtype=float)

# Build ForecastKit and compute Fisher matrix
fk = ForecastKit(function=model, theta0=theta0, cov=cov)
fisher = fk.fisher(
    method="finite",
    stepsize=1e-2,
    num_points=5,
    extrapolation="ridders",
    levels=4,
)

# Draw GetDist MCSamples from the Fisher Gaussian
samples = fk.getdist_fisher_gaussian(
    fisher=fisher,
    names=["a", "b"],
    labels=[r"a", r"b"],
    label="Fisher (samples)",
)

# Plot Fisher contours in DerivKit colors
dk_red = "#f21901"
line_width = 1.5

plotter = getdist_plots.get_subplot_plotter(width_inch=3.6)
plotter.settings.linewidth_contour = line_width
plotter.settings.linewidth = line_width

plotter.triangle_plot(
    [samples],
    params=["a", "b"],
    filled=False,
    contour_colors=[dk_red],
)

print("Fisher matrix:\n", fisher)
print("Samples type:", type(samples))

# Save and/or show
out = Path("plots/forecast_kit_fisher_sampling_demo.pdf")
out.parent.mkdir(parents=True, exist_ok=True)
plotter.export(str(out))
print(f"saved: {out}")

plt.show()
