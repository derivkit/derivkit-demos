"""Toy cluster-count forecast using CROW and DerivKit.

crow: https://github.com/LSSTDESC/crow/tree/main
derivkit: https://docs.derivkit.org/main/

This script builds a simple cluster number-count model and computes parameter
constraints using either Fisher, DALI, or both. The observable is a vector of
cluster counts in redshift–proxy bins, and the covariance is taken to be
Poisson, Cov_ii = N_i at the fiducial.

The model uses a CCL cosmology, a halo mass function, and a simple
mass–observable relation through MurataUnbinned. It is meant as a minimal
working example rather than a fully realistic cluster analysis.

Basic usage (from the project root):
    python -m scripts.derivkit-cluster-counts-forecast
    python -m scripts.derivkit-cluster-counts-forecast --forecast_mode fisher
    python -m scripts.derivkit-cluster-counts-forecast --forecast_mode dali
    python -m scripts.derivkit-cluster-counts-forecast --forecast_mode both

Alternatively, from inside scripts:
    python derivkit-cluster-counts-forecast.py --forecast_mode both

Choose which parameters to vary:
    python -m scripts.derivkit-cluster-counts-forecast --params Omega_m sigma8
    python -m scripts.derivkit-cluster-counts-forecast --params Omega_m sigma8 mu0 mu1 mu2 sigma0 sigma1 sigma2

Override fiducial parameter values:
    python -m scripts.derivkit-cluster-counts-forecast --Omega_m 0.31 --sigma8 0.83
    python -m scripts.derivkit-cluster-counts-forecast --mu0 14.6 --mu1 0.0 --mu2 0.0 --sigma0 0.25 --sigma1 0.0 --sigma2 0.0

Change binning:
    python -m scripts.derivkit-cluster-counts-forecast --n-z-bins 3 --n-proxy-bins 3
    python -m scripts.derivkit-cluster-counts-forecast --n-z-bins 4 --n-proxy-bins 4

Run Fisher only without plotting:
    python -m scripts.derivkit-cluster-counts-forecast --forecast_mode fisher --no-plot

Run DALI with a different stepsize:
    python -m scripts.derivkit-cluster-counts-forecast --forecast_mode dali --stepsize 1e-2

Use 10 percent Gaussian priors on all active parameters:
    python -m scripts.derivkit-cluster-counts-forecast --forecast_mode fisher --prior-frac 0.1
    python -m scripts.derivkit-cluster-counts-forecast --forecast_mode both --prior-frac 0.1

Use custom Gaussian priors on selected parameters:
    python -m scripts.derivkit-cluster-counts-forecast --forecast_mode both \\
        --prior-params Omega_m sigma8 mu0 sigma0 \\
        --prior-sigmas 0.03 0.05 0.2 0.05

Add finite DALI prior bounds at +/- 5 sigma around the fiducial:
    python -m scripts.derivkit-cluster-counts-forecast --forecast_mode dali --prior-frac 0.1 --prior-nsigma-bounds 5

Output plot:
    Saved as toy_crow_counts_triangle.pdf (or use --output to change name)

Notes:
This is a toy setup with Poisson-only covariance. For more stable results,
use more bins or fewer parameters. DALI uses emcee sampling, but the main cost
comes from computing higher-order derivatives. Computation also depends on the
derivative-method backend from DerivKit. See the DerivKit documentation for
more details.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from getdist import plots as getdist_plots
from derivkit import ForecastKit

from crow.recipes.binned_grid import GridBinnedClusterRecipe
from crow.cluster_modules.mass_proxy import MurataUnbinned
from crow.cluster_modules.kernel import SpectroscopicRedshift
from crow.cluster_modules.completeness_models import CompletenessAguena16
from crow.cluster_modules.shear_profile import ClusterShearProfile

# GetDist compatibility for older calls
if not hasattr(np, "infty"):
    np.infty = np.inf


DEFAULT_PARAMS = OrderedDict(
    [
        ("Omega_m", 0.31),
        ("sigma8", 0.8102),
        ("mu0", 3.0),
        ("mu1", 0.8),
        ("mu2", -0.3),
        ("sigma0", 0.3),
        ("sigma1", 0.0),
        ("sigma2", 0.0),
    ]
)

PARAM_LABELS = {
    "Omega_m": r"\Omega_m",
    "sigma8": r"\sigma_8",
    "mu0": r"\mu_0",
    "mu1": r"\mu_1",
    "mu2": r"\mu_2",
    "sigma0": r"\sigma_0",
    "sigma1": r"\sigma_1",
    "sigma2": r"\sigma_2",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Toy cluster-count forecast with CROW + DerivKit."
    )

    parser.add_argument(
        "--forecast_mode",
        choices=["fisher", "dali", "both"],
        default="both",
        help="Which forecast to run.",
    )

    parser.add_argument(
        "--params",
        nargs="+",
        choices=list(DEFAULT_PARAMS.keys()),
        default=list(DEFAULT_PARAMS.keys()),
        help="Active parameter subset, in order.",
    )

    parser.add_argument("--Omega_m", type=float,
                        default=DEFAULT_PARAMS["Omega_m"])
    parser.add_argument("--sigma8", type=float,
                        default=DEFAULT_PARAMS["sigma8"])

    parser.add_argument("--mu0", type=float, default=DEFAULT_PARAMS["mu0"])
    parser.add_argument("--mu1", type=float, default=DEFAULT_PARAMS["mu1"])
    parser.add_argument("--mu2", type=float, default=DEFAULT_PARAMS["mu2"])
    parser.add_argument("--sigma0", type=float,
                        default=DEFAULT_PARAMS["sigma0"])
    parser.add_argument("--sigma1", type=float,
                        default=DEFAULT_PARAMS["sigma1"])
    parser.add_argument("--sigma2", type=float,
                        default=DEFAULT_PARAMS["sigma2"])

    parser.add_argument("--Omega_b", type=float, default=0.04897)
    parser.add_argument("--h", type=float, default=0.6766)
    parser.add_argument("--n_s", type=float, default=0.9665)

    parser.add_argument("--sky-area", type=float, default=440.0)
    parser.add_argument("--mass-min", type=float, default=12.5)
    parser.add_argument("--mass-max", type=float, default=15.0)

    parser.add_argument("--z-min", type=float, default=0.2)
    parser.add_argument("--z-max", type=float, default=0.5)
    parser.add_argument("--n-z-bins", type=int, default=3)

    parser.add_argument("--proxy-min", type=float, default=1.0)
    parser.add_argument("--proxy-max", type=float, default=1.6)
    parser.add_argument("--n-proxy-bins", type=int, default=3)

    parser.add_argument("--mass-grid-size", type=int, default=30)
    parser.add_argument("--redshift-grid-size", type=int, default=10)
    parser.add_argument("--proxy-grid-size", type=int, default=12)

    parser.add_argument("--cov-floor", type=float, default=1.0)

    parser.add_argument("--deriv-method", default="finite")
    parser.add_argument("--stepsize", type=float, default=1e-3)
    parser.add_argument("--num-points", type=int, default=5)
    parser.add_argument("--extrapolation", default="ridders")
    parser.add_argument("--levels", type=int, default=4)

    parser.add_argument("--forecast-order", type=int, default=2)

    parser.add_argument(
        "--output",
        type=str,
        default="toy_crow_counts_triangle.pdf",
        help="Output plot filename.",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting and only print/save numerical results.",
    )

    # ------------------------------------------------------------------
    # Prior controls
    # ------------------------------------------------------------------

    parser.add_argument(
        "--prior-frac",
        type=float,
        default=None,
        help=(
            "Relative 1-sigma Gaussian prior width for active parameters. "
            "Example: --prior-frac 0.1 means sigma_prior = 0.1 * abs(theta0)."
        ),
    )

    parser.add_argument(
        "--prior-params",
        nargs="+",
        choices=list(DEFAULT_PARAMS.keys()),
        default=None,
        help=(
            "Parameter names for custom Gaussian priors. "
            "Must be accompanied by --prior-sigmas."
        ),
    )

    parser.add_argument(
        "--prior-sigmas",
        nargs="+",
        type=float,
        default=None,
        help=(
            "1-sigma Gaussian prior widths for --prior-params, in the same order."
        ),
    )

    parser.add_argument(
        "--prior-nsigma-bounds",
        type=float,
        default=None,
        help=(
            "If set, build DALI prior_bounds as theta0 +/- Nsigma * sigma_prior "
            "for all active parameters that have Gaussian priors."
        ),
    )

    return parser.parse_args()


def make_bin_edges(vmin, vmax, n_bins):
    edges = np.linspace(vmin, vmax, n_bins + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n_bins)]


def make_theta0(args, active_params):
    values = {
        "Omega_m": args.Omega_m,
        "sigma8": args.sigma8,
        "mu0": args.mu0,
        "mu1": args.mu1,
        "mu2": args.mu2,
        "sigma0": args.sigma0,
        "sigma1": args.sigma1,
        "sigma2": args.sigma2,
    }
    return np.array([values[p] for p in active_params], dtype=float)


def make_full_param_dict(theta, active_params, args):
    params = {
        "Omega_m": args.Omega_m,
        "sigma8": args.sigma8,
        "mu0": args.mu0,
        "mu1": args.mu1,
        "mu2": args.mu2,
        "sigma0": args.sigma0,
        "sigma1": args.sigma1,
        "sigma2": args.sigma2,
    }
    for name, value in zip(active_params, theta):
        params[name] = float(value)
    return params


def build_murata_mass_distribution(p):
    """
    Build MurataUnbinned from the mass-richness parameters.
    """
    mass_distribution = MurataUnbinned(
        pivot_log_mass=14.625862906,
        pivot_redshift=0.6,
    )

    mass_distribution.parameters["mu0"] = float(p["mu0"])
    mass_distribution.parameters["mu1"] = float(p["mu1"])
    mass_distribution.parameters["mu2"] = float(p["mu2"])
    mass_distribution.parameters["sigma0"] = float(p["sigma0"])
    mass_distribution.parameters["sigma1"] = float(p["sigma1"])
    mass_distribution.parameters["sigma2"] = float(p["sigma2"])

    return mass_distribution


def make_counts_model(args, active_params):
    """
    Return a CROW-based toy cluster-count model for a chosen active parameter subset.
    """
    redshift_distribution = SpectroscopicRedshift()
    completeness = CompletenessAguena16()

    mass_interval = (args.mass_min, args.mass_max)
    sky_area = args.sky_area

    z_bins = make_bin_edges(args.z_min, args.z_max, args.n_z_bins)
    proxy_bins = make_bin_edges(args.proxy_min, args.proxy_max, args.n_proxy_bins)

    mass_grid_size = args.mass_grid_size
    redshift_grid_size = args.redshift_grid_size
    proxy_grid_size = args.proxy_grid_size

    def model(theta):
        p = make_full_param_dict(theta, active_params, args)

        Omega_m = p["Omega_m"]
        Omega_c = Omega_m - args.Omega_b

        if Omega_c <= 0:
            raise ValueError(
                "Need Omega_m > Omega_b so that Omega_c stays positive."
            )

        cosmo = ccl.Cosmology(
            Omega_c=Omega_c,
            Omega_b=args.Omega_b,
            h=args.h,
            sigma8=p["sigma8"],
            n_s=args.n_s,
        )

        mass_distribution = build_murata_mass_distribution(p)

        hmf = ccl.halos.MassFuncTinker08(mass_def="200c")
        cluster_theory = ClusterShearProfile(cosmo, hmf, 4.0, True)

        recipe = GridBinnedClusterRecipe(
            mass_interval=mass_interval,
            cluster_theory=cluster_theory,
            redshift_distribution=redshift_distribution,
            mass_distribution=mass_distribution,
            completeness=completeness,
            proxy_grid_size=proxy_grid_size,
            redshift_grid_size=redshift_grid_size,
            mass_grid_size=mass_grid_size,
        )

        recipe.setup()

        counts_list = []
        for z_bin in z_bins:
            for proxy_bin in proxy_bins:
                counts = recipe.evaluate_theory_prediction_counts(
                    z_bin,
                    proxy_bin,
                    sky_area,
                )
                counts_list.append(counts)

        return np.array(counts_list, dtype=float)

    return model, z_bins, proxy_bins


def make_poisson_covariance(model, theta0, floor=1.0):
    counts_fid = np.asarray(model(theta0), dtype=float)
    counts_safe = np.maximum(counts_fid, floor)
    cov = np.diag(counts_safe)
    return cov, counts_fid


def get_fiducial_param_dict(args):
    return {
        "Omega_m": args.Omega_m,
        "sigma8": args.sigma8,
        "mu0": args.mu0,
        "mu1": args.mu1,
        "mu2": args.mu2,
        "sigma0": args.sigma0,
        "sigma1": args.sigma1,
        "sigma2": args.sigma2,
    }


def build_prior_sigma_map(args, active_params):
    """
    Build a dictionary {param_name: sigma_prior} for active parameters.

    Priority:
        1. User-passed custom priors via --prior-params / --prior-sigmas
        2. Relative priors via --prior-frac
    """
    sigma_map = {}

    fid = get_fiducial_param_dict(args)

    if args.prior_frac is not None:
        if args.prior_frac <= 0.0:
            raise ValueError("--prior-frac must be positive.")
        for p in active_params:
            ref = abs(float(fid[p]))
            if ref == 0.0:
                continue
            sigma_map[p] = args.prior_frac * ref

    if (args.prior_params is None) ^ (args.prior_sigmas is None):
        raise ValueError(
            "Use --prior-params and --prior-sigmas together."
        )

    if args.prior_params is not None and args.prior_sigmas is not None:
        if len(args.prior_params) != len(args.prior_sigmas):
            raise ValueError(
                "--prior-params and --prior-sigmas must have the same length."
            )
        for p, s in zip(args.prior_params, args.prior_sigmas):
            if p not in active_params:
                raise ValueError(
                    f"Custom prior requested for '{p}', but it is not in active params: "
                    f"{active_params}"
                )
            if s <= 0.0:
                raise ValueError(f"Prior sigma for '{p}' must be positive.")
            sigma_map[p] = float(s)

    return sigma_map


def build_fisher_prior_matrix(active_params, sigma_map):
    """
    Build a diagonal Gaussian prior Fisher matrix in active-parameter order.
    """
    n = len(active_params)
    fisher_prior = np.zeros((n, n), dtype=float)

    for i, p in enumerate(active_params):
        if p in sigma_map:
            fisher_prior[i, i] = 1.0 / (sigma_map[p] ** 2)

    return fisher_prior


def build_dali_prior_terms_and_bounds(theta0, active_params, sigma_map, nsigma_bounds=None):
    """
    Build DerivKit DALI prior_terms and optional prior_bounds.

    We use a multivariate Gaussian prior centered on theta0 with diagonal covariance
    for those parameters that have priors. Unconstrained parameters are omitted from
    the Gaussian prior by giving them effectively infinite variance in the covariance
    matrix used here.
    """
    if not sigma_map:
        return None, None

    n = len(active_params)
    mean = np.array(theta0, dtype=float)

    cov_prior = np.full((n, n), 0.0, dtype=float)
    have_any = False

    for i, p in enumerate(active_params):
        if p in sigma_map:
            cov_prior[i, i] = sigma_map[p] ** 2
            have_any = True
        else:
            cov_prior[i, i] = 1.0e30

    prior_terms = None
    if have_any:
        prior_terms = [("gaussian", {"mean": mean, "cov": cov_prior})]

    prior_bounds = None
    if nsigma_bounds is not None:
        if nsigma_bounds <= 0.0:
            raise ValueError("--prior-nsigma-bounds must be positive.")
        prior_bounds = []
        for i, p in enumerate(active_params):
            if p in sigma_map:
                s = sigma_map[p]
                prior_bounds.append(
                    (float(theta0[i] - nsigma_bounds * s), float(theta0[i] + nsigma_bounds * s))
                )
            else:
                prior_bounds.append((-np.inf, np.inf))

    return prior_terms, prior_bounds


def print_prior_summary(active_params, sigma_map):
    if not sigma_map:
        print("No Gaussian priors applied.")
        return

    print("Gaussian priors applied:")
    for p in active_params:
        if p in sigma_map:
            print(f"  {p:>15s} : sigma_prior = {sigma_map[p]}")


def main():
    args = parse_args()
    active_params = list(args.params)

    model, z_bins, proxy_bins = make_counts_model(args, active_params)
    theta0 = make_theta0(args, active_params)

    y0 = model(theta0)
    cov, counts_fid = make_poisson_covariance(model, theta0, floor=args.cov_floor)

    sigma_map = build_prior_sigma_map(args, active_params)
    fisher_prior = build_fisher_prior_matrix(active_params, sigma_map)
    dali_prior_terms, dali_prior_bounds = build_dali_prior_terms_and_bounds(
        theta0=theta0,
        active_params=active_params,
        sigma_map=sigma_map,
        nsigma_bounds=args.prior_nsigma_bounds,
    )

    print("active_params =", active_params)
    print("theta0 =", theta0)
    print("z_bins =", z_bins)
    print("proxy_bins =", proxy_bins)
    print("data vector size =", len(y0))
    print("predicted counts =", y0)
    print("fiducial counts =", counts_fid)
    print("covariance =")
    print(cov)
    print_prior_summary(active_params, sigma_map)

    if np.any(np.diag(fisher_prior) > 0.0):
        print("fisher_prior =")
        print(fisher_prior)

    if dali_prior_bounds is not None:
        print("dali prior_bounds =", dali_prior_bounds)

    fk = ForecastKit(function=model, theta0=theta0, cov=cov)

    fisher = None
    dali = None
    plot_objects = []
    contour_colors = []
    contour_lws = []
    contour_ls = []
    legend_labels = []

    names = active_params
    labels = [PARAM_LABELS[p] for p in active_params]

    if args.forecast_mode in ["fisher", "both"]:
        print("I am now calculating fisher")
        fisher = fk.fisher(
            method=args.deriv_method,
            stepsize=args.stepsize,
            num_points=args.num_points,
            extrapolation=args.extrapolation,
            levels=args.levels,
        )
        print("fisher_like =")
        print(fisher)

        fisher_post = fisher + fisher_prior
        if np.any(np.diag(fisher_prior) > 0.0):
            print("fisher_post = fisher_like + fisher_prior")
            print(fisher_post)
        else:
            fisher_post = fisher

        evals = np.linalg.eigvalsh(fisher_post)
        print("fisher_post eigenvalues =", evals)

        cov_params = np.linalg.inv(fisher_post)
        sigma_params = np.sqrt(np.diag(cov_params))
        corr_params = cov_params / np.outer(sigma_params, sigma_params)

        print("parameter covariance =")
        print(cov_params)

        print("marginalized sigmas =")
        for p, s in zip(active_params, sigma_params):
            print(f"  {p:>15s} : sigma = {s}")

        print("parameter correlation matrix =")
        print(corr_params)

        gnd_fisher = fk.getdist_fisher_gaussian(
            fisher=fisher_post,
            names=names,
            labels=labels,
            label="Fisher" if not sigma_map else "Fisher + prior",
        )
        plot_objects.append(gnd_fisher)
        contour_colors.append("#f21901")
        contour_lws.append(1.2)
        contour_ls.append("-")
        legend_labels.append("Fisher" if not sigma_map else "Fisher + prior")

    if args.forecast_mode in ["dali", "both"]:
        print("I am now calculating DALI")
        dali = fk.dali(
            forecast_order=args.forecast_order,
            method=args.deriv_method,
            stepsize=args.stepsize,
            num_points=args.num_points,
            extrapolation=args.extrapolation,
            levels=args.levels,
        )

        dali_kwargs = {
            "dali": dali,
            "names": names,
            "labels": labels,
            "label": "DALI" if not sigma_map else "DALI + prior",
        }

        if dali_prior_terms is not None:
            dali_kwargs["prior_terms"] = dali_prior_terms

        if dali_prior_bounds is not None:
            dali_kwargs["prior_bounds"] = dali_prior_bounds

        samples_dali = fk.getdist_dali_emcee(**dali_kwargs)

        plot_objects.append(samples_dali)
        contour_colors.append("#e1af00")
        contour_lws.append(1.2)
        contour_ls.append("-")
        legend_labels.append("DALI" if not sigma_map else "DALI + prior")

    if not args.no_plot:
        width = 3.8 if len(active_params) <= 2 else 7.0
        plotter = getdist_plots.get_subplot_plotter(width_inch=width)
        plotter.settings.linewidth_contour = 1.2
        plotter.settings.linewidth = 1.2
        plotter.settings.figure_legend_frame = False
        plotter.settings.legend_rect_border = False

        plotter.triangle_plot(
            plot_objects,
            params=names,
            filled=False,
            contour_colors=contour_colors,
            contour_lws=contour_lws,
            contour_ls=contour_ls,
            legend_labels=legend_labels,
        )

        plotter.export(args.output)
        print(f"saved plot: {args.output}")

        plt.show()


if __name__ == "__main__":
    main()
