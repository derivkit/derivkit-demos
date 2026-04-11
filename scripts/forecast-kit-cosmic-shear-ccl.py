#!/usr/bin/env python3
"""DerivKit demo: cosmic-shear Fisher and DALI forecast with IA and baryons.

This script builds a simple cosmic-shear demo model using:
- a Smail redshift distribution split into equipopulated source bins,
- weak lensing tracers with a simple intrinsic-alignment (IA) scaling,
- a van Daalen 2019 baryonic suppression model through pyccl,
- a diagonal Gaussian covariance built from a fractional signal floor.

It then computes:
- a Fisher matrix forecast,
- a DALI forecast (doublet by default),
- posterior samples with Gaussian priors,
- a GetDist triangle plot comparing Fisher and DALI.

This is a lightweight demonstration script meant for CLI use.

Examples
--------
Run with defaults:
    python -m scripts.forecast-kit-cosmic-shear-ccl

Show the plot interactively:
    python -m scripts.forecast-kit-cosmic-shear-ccl --show

Save the plot:
    python -m scripts.forecast-kit-cosmic-shear-ccl --save plots/cosmic_shear_demo.png

Use a different derivative backend:
    python -m scripts.forecast-kit-cosmic-shear-ccl --method adaptive

Use DALI triplet tensors:
    python -m scripts.forecast-kit-cosmic-shear-ccl --forecast-order 3

Change fiducial parameters:
    python -m scripts.forecast-kit-cosmic-shear-ccl \
        --om-m 0.315 --sig8 0.8 --ia-amp 0.5 --ia-eta 2.2 --fbar 0.7
"""

from __future__ import annotations

# must be set before importing numpy / pyccl
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from getdist import plots as getdist_plots

from derivkit import ForecastKit


def smail_source_bins(
    z: np.ndarray,
    n_source: int,
    *,
    z0: float = 0.13,
    alpha: float = 0.78,
    beta: float = 2.0,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Build a Smail n(z) and split it into equipopulated source bins.

    Parameters
    ----------
    z
        Redshift grid.
    n_source
        Number of source bins.
    z0, alpha, beta
        Smail distribution parameters.

    Returns
    -------
    nz_parent
        Parent normalized redshift distribution.
    source_bins
        List of normalized per-bin redshift distributions on the same z grid.
    bin_edges
        Equipopulated bin edges in redshift.
    """
    z = np.asarray(z, dtype=float)

    nz = (z / z0) ** beta * np.exp(-((z / z0) ** alpha))
    nz /= np.trapezoid(nz, z)

    cdf = np.concatenate(
        (
            [0.0],
            np.cumsum(0.5 * (nz[1:] + nz[:-1]) * (z[1:] - z[:-1])),
        )
    )

    edges = np.interp(np.linspace(0.0, cdf[-1], n_source + 1), cdf, z)
    edges[0] = z[0]
    edges[-1] = z[-1]

    source_bins: list[np.ndarray] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi < edges[-1]:
            mask = (z >= lo) & (z < hi)
        else:
            mask = (z >= lo) & (z <= hi)

        nz_i = np.where(mask, nz, 0.0)
        norm = np.trapezoid(nz_i, z)
        if norm <= 0.0:
            raise ValueError(f"Encountered empty source bin in range [{lo}, {hi}].")
        nz_i /= norm
        source_bins.append(nz_i)

    return nz, source_bins, edges


def shear_power_spectra(
    theta: np.ndarray,
    *,
    z: np.ndarray,
    ell: np.ndarray,
    source_bins: list[np.ndarray],
    omega_b: float = 0.045,
    h: float = 0.67,
    n_s: float = 0.96,
    z_pivot_ia: float = 0.62,
) -> np.ndarray:
    """Compute cosmic-shear C_ell data vector with IA and baryons.

    Parameters
    ----------
    theta
        Model parameters [Omega_m, sigma8, A_IA, eta_IA, f_bar].
    z
        Redshift grid.
    ell
        Multipole grid.
    source_bins
        Source-bin redshift distributions.
    omega_b, h, n_s
        Fixed cosmological parameters for the demo.
    z_pivot_ia
        Pivot redshift for IA scaling.

    Returns
    -------
    data_vector
        Flattened array containing all unique tomographic C_ell blocks.
    """
    om_m, sig8, ia_amp, ia_eta, fbar = map(float, theta)

    omega_c = om_m - omega_b
    if omega_c <= 0.0:
        raise ValueError(
            f"Need Omega_m > Omega_b. Got Omega_m={om_m:.6f}, Omega_b={omega_b:.6f}."
        )

    cosmo = ccl.Cosmology(
        Omega_c=omega_c,
        Omega_b=omega_b,
        h=h,
        sigma8=sig8,
        n_s=n_s,
        transfer_function="boltzmann_camb",
    )

    ia_signal = ia_amp * ((1.0 + z) / (1.0 + z_pivot_ia)) ** ia_eta

    vd = ccl.baryons.BaryonsvanDaalen19(fbar=fbar, mass_def="500c")
    pk_nl = cosmo.get_nonlin_power()
    pk_bar = vd.include_baryonic_effects(cosmo, pk_nl)

    tracers = [
        ccl.WeakLensingTracer(cosmo, dndz=(z, nz_i), ia_bias=(z, ia_signal))
        for nz_i in source_bins
    ]

    n_tr = len(tracers)
    n_cls = n_tr * (n_tr + 1) // 2
    out = np.empty(n_cls * ell.size, dtype=float)

    k = 0
    for i in range(n_tr):
        for j in range(i, n_tr):
            out[k : k + ell.size] = ccl.angular_cl(
                cosmo,
                tracers[i],
                tracers[j],
                ell,
                p_of_k_a=pk_bar,
            )
            k += ell.size

    return out


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run a DerivKit cosmic-shear Fisher/DALI demo."
    )

    parser.add_argument("--n-source", type=int, default=5, help="Number of source bins.")
    parser.add_argument("--n-z", type=int, default=300, help="Number of redshift samples.")
    parser.add_argument("--z-max", type=float, default=3.0, help="Maximum redshift.")
    parser.add_argument("--n-ell", type=int, default=20, help="Number of ell values.")
    parser.add_argument("--ell-min", type=float, default=20.0, help="Minimum ell.")
    parser.add_argument("--ell-max", type=float, default=2000.0, help="Maximum ell.")

    parser.add_argument("--om-m", type=float, default=0.315, help="Fiducial Omega_m.")
    parser.add_argument("--sig8", type=float, default=0.80, help="Fiducial sigma8.")
    parser.add_argument("--ia-amp", type=float, default=0.50, help="Fiducial IA amplitude.")
    parser.add_argument("--ia-eta", type=float, default=2.2, help="Fiducial IA redshift slope.")
    parser.add_argument("--fbar", type=float, default=0.70, help="Fiducial baryon parameter.")

    parser.add_argument(
        "--method",
        type=str,
        default="finite",
        choices=["finite", "adaptive", "polyfit"],
        help="Derivative method passed to ForecastKit.",
    )
    parser.add_argument(
        "--extrapolation",
        type=str,
        default="ridders",
        help="Extrapolation method for finite differences.",
    )
    parser.add_argument(
        "--forecast-order",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Forecast order for DALI: 1=Fisher, 2=doublet, 3=triplet.",
    )
    parser.add_argument(
        "--cov-frac",
        type=float,
        default=0.05,
        help="Fractional diagonal covariance amplitude.",
    )
    parser.add_argument(
        "--prior-frac",
        type=float,
        default=0.10,
        help="Fractional 1-sigma Gaussian prior width relative to |theta0|.",
    )

    parser.add_argument(
        "--label-fisher",
        type=str,
        default="Fisher + Gaussian prior",
        help="Legend label for Fisher samples.",
    )
    parser.add_argument(
        "--label-dali",
        type=str,
        default="DALI + Gaussian prior",
        help="Legend label for DALI samples.",
    )

    parser.add_argument("--show", action="store_true", help="Display the triangle plot.")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path for the triangle plot.",
    )
    parser.add_argument(
        "--no-tight-layout",
        action="store_true",
        help="Disable tight_layout before saving/showing.",
    )

    return parser


def main() -> None:
    """Run the CLI workflow."""
    parser = build_parser()
    args = parser.parse_args()

    print("Thread limits set. Now importing/running numpy, pyccl, and DerivKit workflow.")

    ell = np.geomspace(args.ell_min, args.ell_max, args.n_ell)
    z = np.linspace(0.0, args.z_max, args.n_z)

    nz_parent, source_bins, source_edges = smail_source_bins(z, n_source=args.n_source)

    theta0 = np.array(
        [args.om_m, args.sig8, args.ia_amp, args.ia_eta, args.fbar],
        dtype=float,
    )

    def model(theta: np.ndarray) -> np.ndarray:
        return shear_power_spectra(theta, z=z, ell=ell, source_bins=source_bins)

    y0 = model(theta0)

    floor = 1e-12 * np.max(np.abs(y0))
    sigma_i = args.cov_frac * np.maximum(np.abs(y0), floor)
    cov = np.diag(sigma_i**2)

    fk = ForecastKit(function=model, theta0=theta0, cov=cov)

    fisher_kwargs: dict[str, object] = {"method": args.method}
    dali_kwargs: dict[str, object] = {
        "forecast_order": args.forecast_order,
        "method": args.method,
    }

    if args.method == "finite":
        fisher_kwargs["extrapolation"] = args.extrapolation
        dali_kwargs["extrapolation"] = args.extrapolation

    print("Computing Fisher matrix...")
    fisher = fk.fisher(**fisher_kwargs)

    print(f"Computing DALI forecast (order={args.forecast_order})...")
    dali = fk.dali(**dali_kwargs)

    names = ["om_m", "sig8", "ia_amp", "ia_eta", "f_bar"]
    labels = [
        r"\Omega_m",
        r"\sigma_8",
        r"A_{\rm IA}",
        r"\eta_{\rm IA}",
        r"f_{\rm bar}",
    ]

    sigma_prior = args.prior_frac * np.maximum(np.abs(theta0), 1e-12)
    fisher_prior = np.diag(1.0 / sigma_prior**2)
    fisher_post = fisher + fisher_prior

    fisher_samples_post = fk.getdist_fisher_gaussian(
        fisher=fisher_post,
        names=names,
        labels=labels,
        label=args.label_fisher,
    )

    prior_terms = [
        ("gaussian", {"mean": theta0, "cov": np.diag(sigma_prior**2)}),
    ]

    dali_samples_post = fk.getdist_dali_emcee(
        dali=dali,
        names=names,
        labels=labels,
        label=args.label_dali,
        prior_terms=prior_terms,
    )

    samples = [fisher_samples_post, dali_samples_post]
    colors = cmr.take_cmap_colors(
        "cmr.prinsenvlag",
        len(samples),
        cmap_range=(0.2, 0.8),
        return_fmt="hex",
    )

    print("Building GetDist triangle plot...")
    plotter = getdist_plots.get_subplot_plotter(width_inch=7)
    plotter.triangle_plot(
        samples,
        params=names,
        filled=False,
        contour_colors=colors,
        contour_lws=[2.0] * len(samples),
        contour_ls=["-"] * len(samples),
    )

    title = (
        f"Cosmic shear demo: n_source={args.n_source}, "
        f"n_ell={args.n_ell}, method={args.method}"
    )
    if hasattr(plotter, "fig") and plotter.fig is not None:
        plotter.fig.suptitle(title, y=1.02, fontsize=12)
        if not args.no_tight_layout:
            plotter.fig.tight_layout()

    print("\nSummary")
    print("-------")
    print(f"theta0           = {theta0}")
    print(f"source bin edges = {source_edges}")
    print(f"data length      = {y0.size}")
    print(f"cov frac         = {args.cov_frac}")
    print(f"prior frac       = {args.prior_frac}")
    print(f"method           = {args.method}")
    print(f"forecast order   = {args.forecast_order}")
    print(f"Fisher shape     = {fisher.shape}")

    if args.save is not None:
        outpath = Path(args.save)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {outpath}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
