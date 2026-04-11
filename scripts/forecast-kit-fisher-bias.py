"""DerivKit — ForecastKit Fisher Bias Demo (GetDist).

This demo shows how to use :class:`ForecastKit` to compute the Fisher bias
and the corresponding parameter shift Δθ caused by a systematic offset in the
observables, then visualize the unbiased and biased Fisher Gaussians with
GetDist.

We work with a simple 2-parameter toy model:
    o1 = θ1
    o2 = θ2
    o3 = θ1 + 2 θ2

Steps
-----
- Define a model mapping parameters (θ1, θ2) to observables.
- Compute the Fisher matrix numerically using ForecastKit.
- Build a toy systematic offset Δd in the observables.
- Compute the Fisher bias vector and parameter shift Δθ.
- Build unbiased and biased Gaussian GetDist samples using the same Fisher matrix.
- Plot the corresponding contours.

Usage
-----
    python -m scripts.forecast-kit-fisher-bias

Notes
-----
- The first-order Fisher bias formula is:
      Δθ ≈ F^{-1} J^T C^{-1} Δν
  where:
      F = J^T C^{-1} J
      J = ∂m / ∂θ
      Δν = data_biased - data_unbiased
- This demo uses ForecastKit’s built-in Fisher and Fisher-bias utilities.
"""

from pathlib import Path

from getdist import plots as getdist_plots
from derivkit import ForecastKit
import matplotlib.pyplot as plt
import numpy as np


def model(theta0):
    """Simple toy model."""
    theta1, theta2 = theta0
    return np.array([theta1, theta2, theta1 + 2.0 * theta2], dtype=float)


def jacobian_analytic(theta1, theta2):
    """Analytic Jacobian of the toy model."""
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )


def fisher_analytic(theta1, theta2, cov):
    """Analytic Fisher matrix."""
    jac = jacobian_analytic(theta1, theta2)
    cov_inv = np.linalg.inv(cov)
    return jac.T @ cov_inv @ jac


def analytic_bias_delta_theta(theta0, cov, delta_nu):
    """Analytic first-order Fisher bias parameter shift."""
    jac = jacobian_analytic(theta0[0], theta0[1])
    cov_inv = np.linalg.inv(cov)
    fisher = jac.T @ cov_inv @ jac
    return np.linalg.pinv(fisher, rcond=1e-12) @ (jac.T @ cov_inv @ delta_nu)


def print_comparison(name, numeric, analytic):
    """Print numeric vs analytic arrays and their difference."""
    delta = numeric - analytic
    print(f"\n{name} (numeric):\n{numeric}")
    print(f"\n{name} (analytic):\n{analytic}")
    print(f"\nΔ = numeric - analytic:\n{delta}")
    print(f"max|Δ| = {np.max(np.abs(delta)):.3e}")
    print(f"||Δ||₂ = {np.linalg.norm(delta):.3e}")


# Fiducial parameters and covariance
theta0 = np.array([1.0, 2.0], dtype=float)
cov = np.eye(3, dtype=float)

# Unbiased ForecastKit at theta0
fk = ForecastKit(function=model, theta0=theta0, cov=cov)

# Fisher at theta0
fisher = fk.fisher(
    method="finite",
    stepsize=1e-2,
    num_points=5,
    extrapolation="ridders",
    levels=4,
)

# Analytic Fisher cross-check
fisher_ref = fisher_analytic(theta0[0], theta0[1], cov)

# Construct biased and unbiased data vectors
data_unbiased = model(theta0)
data_biased = data_unbiased + np.array([0.5, -0.8, 0.3], dtype=float)

# Systematic difference vector
delta_nu = fk.delta_nu(
    data_unbiased=data_unbiased,
    data_biased=data_biased,
)

# Fisher bias and parameter shift
bias_vec, delta_theta = fk.fisher_bias(
    fisher_matrix=fisher,
    delta_nu=delta_nu,
    method="finite",
    stepsize=1e-2,
    num_points=5,
    extrapolation="ridders",
    levels=4,
    rcond=1e-12,
)

# Analytic Δθ cross-check
delta_theta_ref = analytic_bias_delta_theta(theta0, cov, delta_nu)

# Print results
print("=== ForecastKit Fisher Bias Demo (GetDist) ===")
print("theta0 =", theta0)
print("covariance =\n", cov)
print("model(theta0) =", data_unbiased)
print("data_biased =", data_biased)
print("delta_nu =", delta_nu)

print_comparison("Fisher matrix", fisher, fisher_ref)
print("\nBias vector:\n", bias_vec)
print_comparison("Delta theta", delta_theta, delta_theta_ref)

# Parameter covariance and sigma-units
param_cov = np.linalg.pinv(fisher, rcond=1e-12)
sigma = np.sqrt(np.diag(param_cov))
print("\nParameter covariance F^{-1}:\n", param_cov)
print("Marginal 1σ =", sigma)
print("Delta theta / sigma =", delta_theta / sigma)

# Biased ForecastKit: same model and covariance, shifted center
fk_biased = ForecastKit(function=model, theta0=theta0 + delta_theta, cov=cov)

# Convert to GetDist Gaussian samples
gnd_unbiased = fk.getdist_fisher_gaussian(
    fisher=fisher,
    names=["theta1", "theta2"],
    labels=[r"\theta_1", r"\theta_2"],
    label="Unbiased",
)

gnd_biased = fk_biased.getdist_fisher_gaussian(
    fisher=fisher,
    names=["theta1", "theta2"],
    labels=[r"\theta_1", r"\theta_2"],
    label="Biased",
)

# Plot biased and unbiased contours
dk_red = "#f21901"
dk_yellow = "#e1af00"
line_width = 1.5

plotter = getdist_plots.get_subplot_plotter(width_inch=3.6)
plotter.settings.linewidth_contour = line_width
plotter.settings.linewidth = line_width
plotter.settings.figure_legend_frame = False

plotter.triangle_plot(
    [gnd_unbiased, gnd_biased],
    params=["theta1", "theta2"],
    legend_labels=["Unbiased", "Biased"],
    legend_ncol=1,
    filled=[False, False],
    contour_colors=[dk_yellow, dk_red],
    contour_lws=[line_width, line_width],
    contour_ls=["-", "-"],
)

# Save and/or show
out = Path("plots/forecast_kit_fisher_bias_demo.pdf")
out.parent.mkdir(parents=True, exist_ok=True)
plotter.export(str(out))
print(f"saved: {out}")
plt.show()

print("Done.")