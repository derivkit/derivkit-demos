"""
This script builds a simple cluster-count model and computes parameter
constraints using either Fisher, DALI, or both.

This script calls 10_cluster_counts.py and computes posterior contours
for a user-specified sweep of parameters. The resulting contours are
overlaid in a single plot to illustrate how the constraints change as
the parameters are varied.

You can run a sweep over ANY parameter or combination of parameters
that are cirrently accessible via command-line arguments.

Examples:

Sweep over sky area (fisher)
python demo-scripts/11_run_cluster_sweep_overlay.py \
  --forecast_mode fisher \
  --params Omega_m sigma8 \
  --sweep sky_area=440,1000,2000,5000,10000 \
  --output skyarea_overlay.pdf

Sweep over nuisance parameters (dali):
python demo-scripts/run_cluster_sweep_overlay.py \
  --forecast_mode dali \
  --params Omega_m sigma8 mu0 sigma0 \
  --sweep mu0=0.8,1.0,1.2 sigma0=0.1,0.2,0.3 \
  --output mu0_sigma0_grid.pdf

Sweep over nuisance parameter sigma0:
python demo-scripts/11_run_cluster_sweep_overlay.py \
  --forecast_mode dali \
  --params Omega_m sigma8 mu0 \
  --sweep mu0=1.1,1.2 \
  --output mu0_overlay_dali.pdf

Keep in mind DALI computes higher order derivatives so it is slower
than Fisher.
"""

import sys
import ast
import itertools
import importlib.util
from pathlib import Path
import argparse

import cmasher as cmr
from getdist import plots as getdist_plots


def load_cluster_demo_module():
    here = Path(__file__).resolve().parent
    script_path = here / "10_cluster_counts.py"

    spec = importlib.util.spec_from_file_location("cluster_counts_demo", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_wrapper_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the cluster-count demo for a sweep of exposed parameters "
            "and overlay all resulting contours."
        )
    )
    parser.add_argument(
        "--forecast_mode",
        choices=["fisher", "dali", "both"],
        default="both",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=["Omega_m", "sigma8"],
        help="Active forecast parameters passed to 10_cluster_counts.py",
    )
    parser.add_argument(
        "--sweep",
        nargs="+",
        required=True,
        help=(
            "Sweep specification(s), e.g. "
            "'sky_area=440,1000,2000' or "
            "'Omega_m=0.28,0.30,0.32 sigma8=0.78,0.82,0.86'"
        ),
    )
    parser.add_argument(
        "--output",
        default="cluster_sweep_overlay.pdf",
    )
    parser.add_argument(
        "--filled",
        action="store_true",
    )
    parser.add_argument(
        "--colormap",
        default="cmr.pride",
    )
    parser.add_argument(
        "--cmap_min",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--cmap_max",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--legend_loc",
        default="upper right",
    )

    # Optional pass-through arguments to the main demo
    parser.add_argument("--prior-frac", type=float, default=None)
    parser.add_argument("--prior-params", nargs="+", default=None)
    parser.add_argument("--prior-sigmas", nargs="+", type=float, default=None)
    parser.add_argument("--prior-nsigma-bounds", type=float, default=None)
    parser.add_argument("--stepsize", type=float, default=None)
    parser.add_argument("--num_points", type=int, default=None)
    parser.add_argument("--levels", type=int, default=None)
    parser.add_argument("--deriv_method", type=str, default=None)
    parser.add_argument("--extrapolation", type=str, default=None)
    parser.add_argument("--forecast_order", type=int, default=None)

    return parser.parse_args()


def maybe_parse_scalar(value):
    """
    Convert strings like '440', '0.3', 'True', 'None' to Python scalars.
    Otherwise keep as string.
    """
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def parse_sweep_specs(specs):
    """
    Convert:
        ["sky_area=440,1000", "Omega_m=0.28,0.30"]
    into:
        {
            "sky_area": [440, 1000],
            "Omega_m": [0.28, 0.30],
        }
    """
    sweep_dict = {}

    for item in specs:
        if "=" not in item:
            raise ValueError(
                f"Bad sweep spec '{item}'. Expected form like name=v1,v2,v3"
            )

        name, values_str = item.split("=", 1)
        name = name.strip()

        if not name:
            raise ValueError(f"Bad sweep spec '{item}': empty parameter name")

        values = [maybe_parse_scalar(v.strip()) for v in values_str.split(",") if v.strip()]
        if not values:
            raise ValueError(f"Bad sweep spec '{item}': no values supplied")

        sweep_dict[name] = values

    return sweep_dict


def generate_cases(sweep_dict):
    """
    Build Cartesian product of all sweep values.
    """
    keys = list(sweep_dict.keys())
    value_lists = [sweep_dict[k] for k in keys]

    cases = []
    for combo in itertools.product(*value_lists):
        case = dict(zip(keys, combo))
        cases.append(case)

    return cases


def format_case_label(case):
    parts = []
    for k, v in case.items():
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def build_demo_args(demo_module, wrapper_args, case):
    """
    Reuse the original demo parser so we inherit all defaults from
    10_cluster_counts.py instead of copying them here.
    """
    argv = [
        "10_cluster_counts.py",
        "--forecast_mode", wrapper_args.forecast_mode,
        "--params", *wrapper_args.params,
        "--no-plot",
    ]

    # sweep-controlled parameters become CLI flags
    for key, value in case.items():
        flag = "--" + key.replace("_", "-")
        argv += [flag, str(value)]

    # pass-through optional knobs
    if wrapper_args.prior_frac is not None:
        argv += ["--prior-frac", str(wrapper_args.prior_frac)]

    if wrapper_args.prior_params is not None:
        argv += ["--prior-params", *wrapper_args.prior_params]

    if wrapper_args.prior_sigmas is not None:
        argv += ["--prior-sigmas", *[str(x) for x in wrapper_args.prior_sigmas]]

    if wrapper_args.prior_nsigma_bounds is not None:
        argv += ["--prior-nsigma-bounds", str(wrapper_args.prior_nsigma_bounds)]

    if wrapper_args.stepsize is not None:
        argv += ["--stepsize", str(wrapper_args.stepsize)]

    if wrapper_args.num_points is not None:
        argv += ["--num_points", str(wrapper_args.num_points)]

    if wrapper_args.levels is not None:
        argv += ["--levels", str(wrapper_args.levels)]

    if wrapper_args.deriv_method is not None:
        argv += ["--deriv_method", wrapper_args.deriv_method]

    if wrapper_args.extrapolation is not None:
        argv += ["--extrapolation", wrapper_args.extrapolation]

    if wrapper_args.forecast_order is not None:
        argv += ["--forecast_order", str(wrapper_args.forecast_order)]

    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        args = demo_module.parse_args()
    finally:
        sys.argv = old_argv

    return args


def main():
    wrapper_args = parse_wrapper_args()
    demo = load_cluster_demo_module()

    sweep_dict = parse_sweep_specs(wrapper_args.sweep)
    cases = generate_cases(sweep_dict)

    print("=" * 80)
    print("Sweep dictionary:")
    print(sweep_dict)
    print(f"Total cases: {len(cases)}")
    print("=" * 80)

    colors = cmr.take_cmap_colors(
        wrapper_args.colormap,
        len(cases),
        cmap_range=(0.5, 0.85),
        return_fmt="hex",
    )

    plot_objects = []
    contour_colors = []
    legend_labels = []

    names = wrapper_args.params
    labels = [demo.PARAM_LABELS[p] for p in names]

    for case, color in zip(cases, colors):
        label = format_case_label(case)

        print("\n" + "=" * 80)
        print(f"Running case: {label}")
        print("=" * 80)

        args = build_demo_args(demo, wrapper_args, case)
        active_params = list(args.params)

        model, z_bins, proxy_bins = demo.make_counts_model(args, active_params)
        theta0 = demo.make_theta0(args, active_params)
        cov, counts_fid = demo.make_poisson_covariance(model, theta0, floor=args.cov_floor)

        sigma_map = demo.build_prior_sigma_map(args, active_params)
        fisher_prior = demo.build_fisher_prior_matrix(active_params, sigma_map)
        dali_prior_terms, dali_prior_bounds = demo.build_dali_prior_terms_and_bounds(
            theta0=theta0,
            active_params=active_params,
            sigma_map=sigma_map,
            nsigma_bounds=args.prior_nsigma_bounds,
        )

        print("active_params =", active_params)
        print("theta0 =", theta0)
        print("fiducial counts =", counts_fid)

        fk = demo.ForecastKit(function=model, theta0=theta0, cov=cov)

        if args.forecast_mode in ["fisher", "both"]:
            fisher = fk.fisher(
                method=args.deriv_method,
                stepsize=args.stepsize,
                num_points=args.num_points,
                extrapolation=args.extrapolation,
                levels=args.levels,
            )
            fisher_post = fisher + fisher_prior

            gnd_fisher = fk.getdist_fisher_gaussian(
                fisher=fisher_post,
                names=names,
                labels=labels,
                label=f"Fisher: {label}",
            )

            plot_objects.append(gnd_fisher)
            contour_colors.append(color)
            legend_labels.append(f"Fisher: {label}")

        if args.forecast_mode in ["dali", "both"]:
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
                "label": f"DALI: {label}",
            }

            if dali_prior_terms is not None:
                dali_kwargs["prior_terms"] = dali_prior_terms

            if dali_prior_bounds is not None:
                dali_kwargs["prior_bounds"] = dali_prior_bounds

            samples_dali = fk.getdist_dali_emcee(**dali_kwargs)

            plot_objects.append(samples_dali)
            contour_colors.append(color)
            legend_labels.append(f"DALI: {label}")

    width = 3.8 if len(names) <= 2 else 7.0
    plotter = getdist_plots.get_subplot_plotter(width_inch=width)
    plotter.settings.linewidth_contour = 2.5
    plotter.settings.linewidth = 2.5
    plotter.settings.figure_legend_frame = False
    plotter.settings.legend_rect_border = False
    plotter.settings.legend_fontsize = 16

    plotter.triangle_plot(
        plot_objects,
        params=names,
        filled=wrapper_args.filled,
        contour_colors=contour_colors,
        legend_labels=legend_labels,
    )

    plotter.export(wrapper_args.output)
    print(f"\nSaved combined plot to: {wrapper_args.output}")


if __name__ == "__main__":
    main()
