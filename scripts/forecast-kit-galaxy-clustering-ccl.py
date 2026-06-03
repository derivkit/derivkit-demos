"""DerivKit demo: galaxy-clustering CCL forecast with Binny, CCL, and TJPCov.

Run from the derivkit-demos repository root.

The YAML config lives under scripts/configs, so use:

    --config scripts/configs/cosmo_srd_gc.yaml

Scale cuts:

    By default no kmax scale cut is applied. To apply a simple, fiducial,
    bin-pair-dependent clustering scale cut, pass --kmax. The cut uses

        ell + 0.5 <= kmax * min(chi_i, chi_j)

    where chi_i and chi_j are evaluated at the fiducial effective redshifts
    of the two lens bins. The mask is fixed at the fiducial point before
    derivatives are computed, so the data-vector length stays constant.

Single Fisher run using YAML defaults:

    python -m scripts.forecast-kit-galaxy-clustering-ccl \
        --config scripts/configs/cosmo_srd_gc.yaml \
        --correlations diagonal \
        --show

Compare autos-only and unique auto/cross correlations:

    python -m scripts.forecast-kit-galaxy-clustering-ccl \
        --config scripts/configs/cosmo_srd_gc.yaml \
        --scenario-correlations diagonal,unique \
        --show

Sweep cosmology plus lens photo-z shift/stretch:

    python -m scripts.forecast-kit-galaxy-clustering-ccl \
        --config scripts/configs/cosmo_srd_gc.yaml \
        --theta-names Omega_c,sigma8,mean_offset,scatter_scale \
        --scenario-correlations diagonal,unique \
        --mean-offsets 0.0 \
        --scatter-scales 0.027,0.030,0.033 \
        --run-name gc_cosmo_photoz_sweep \
        --show

sweep all cosmo params and shift and stretch

        python -m scripts.forecast-kit-galaxy-clustering-ccl \
          --config scripts/configs/cosmo_srd_gc.yaml \
          --theta-names Omega_m,sigma8,Omega_b,h,n_s,w0,wa,mean_offset,scatter_scale \
          --scenario-correlations diagonal,unique \
          --mean-offsets 0.0 \
          --scatter-scales 0.03 \
          --run-name gc_all_cosmo_photoz_default \
          --show

sweep over gbias params and cosmo

        python -m scripts.forecast-kit-galaxy-clustering-ccl \
          --config scripts/configs/cosmo_srd_gc.yaml \
          --theta-names Omega_m,sigma8,bias_0,bias_1,bias_2,bias_3,bias_4 \
          --scenario-correlations diagonal,unique \
          --show

Internal derivative consistency check:

    python -m scripts.forecast-kit-galaxy-clustering-ccl \
        --config scripts/configs/cosmo_srd_gc.yaml \
        --theta-names Omega_m,sigma8,mean_offset,scatter_scale \
        --scenario-correlations diagonal \
        --mean-offsets 0.0 \
        --scatter-scales 0.03 \
        --check-derivative-methods finite \
        --check-derivative-stepsizes 1e-4,1e-3,1e-2,1e-1 \
        --run-name gc_derivative_consistency \
        --show

Another internal derivative conistency check (differnt derivative backends):

        python -m scripts.forecast-kit-galaxy-clustering-ccl \
          --config scripts/configs/cosmo_srd_gc.yaml \
          --theta-names Omega_m,sigma8,mean_offset,scatter_scale \
          --scenario-correlations diagonal \
          --mean-offsets 0.0 \
          --scatter-scales 0.03 \
          --check-derivative-presets ridders_default,local_polynomial,adaptive \
          --run-name gc_derivative_backend_consistency \
          --show

compare runs w diffrent scale cuts

        python -m scripts.forecast-kit-galaxy-clustering-ccl \
          --config scripts/configs/cosmo_srd_gc.yaml \
          --theta-names Omega_m,sigma8,mean_offset,scatter_scale \
          --scenario-correlations diagonal \
          --mean-offsets 0.0 \
          --scatter-scales 0.03 \
          --kmax-values 0.2,0.3 \
          --kmax-units h/Mpc \
          --run-name gc_scale_cut_comparison_kmax0p2_0p3 \
          --show

Optional DALI:

    DALI small sweep aka quicker

        python -m scripts.forecast-kit-galaxy-clustering-ccl \
          --config scripts/configs/cosmo_srd_gc.yaml \
          --theta-names Omega_m,sigma8,mean_offset,scatter_scale \
          --scenario-correlations diagonal,unique \
          --mean-offsets 0.0 \
          --scatter-scales 0.03 \
          --include-dali \
          --forecast-order 3 \
          --run-name gc_all_cosmo_photoz_default_fisher_dali \
          --show


    DALI larger sweep (this will be slower)

        python -m scripts.forecast-kit-galaxy-clustering-ccl \
          --config scripts/configs/cosmo_srd_gc.yaml \
          --theta-names Omega_m,sigma8, Omega_b,h,n_s,w0,wa,mean_offset,scatter_scale \
          --scenario-correlations diagonal,unique \
          --mean-offsets 0.0 \
          --scatter-scales 0.03 \
          --include-dali \
          --forecast-order 3 \
          --run-name gc_all_cosmo_photoz_default_fisher_dali_full \
          --show
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import os
import warnings
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", message="No bandpower windows associated with these data")
warnings.filterwarnings("ignore", message="No bandpower windows found.*")
warnings.filterwarnings("ignore", message="Missing n_ell_coupled info.*")

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import yaml
from getdist import plots as getdist_plots

from binny import NZTomography
from derivkit import ForecastKit
from tjpcov.covariance_calculator import CovarianceCalculator


PHOTOZ_PARAM_NAMES = {"mean_offset", "scatter_scale"}

DERIVATIVE_CHECK_PRESETS = {
    "ridders_default": {
        "label": "finite/ridders default",
        "method": "finite",
        "kwargs": {
            "extrapolation": "ridders",
            "stepsize": 0.01,
        },
    },
    "local_polynomial": {
        "label": "local polynomial",
        "method": "local_polynomial",
        "kwargs": {},
    },
    "adaptive": {
        "label": "adaptive polynomial",
        "method": None,
        "kwargs": {},
    },
}

DEFAULT_ELL_MIN = 20.0
DEFAULT_ELL_MAX = 1000.0
DEFAULT_N_ELL = 20
DEFAULT_KMAX_UNITS = "h/Mpc"


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    """Loads a yaml file."""
    if path is None:
        return {}

    with Path(path).open("r") as stream:
        data = yaml.safe_load(stream)

    return data or {}


def get_tomography_entry(cfg, *, role: str, year: str) -> dict:
    """Loads binny LSST survey present for a given year."""
    for entry in cfg["tomography"]:
        if entry.get("role") == role and str(entry.get("year")) == str(year):
            return entry

    raise ValueError(f"Could not find tomography entry role={role!r}, year={year!r}.")


def get_lens_number_density(cfg, *, year: str) -> float:
    """This just extract the num density from binny survey config."""
    entry = get_tomography_entry(cfg, role="lens", year=year)
    return float(entry["sample_properties"]["number_density"]["n_gal_arcmin2"])


def get_fsky(cfg, *, year: str) -> float:
    """This just extracts the fsky from binny survey config."""
    area = float(cfg["survey_meta"]["footprint"]["years"][str(year)]["survey_area"])
    return area / 41252.96124941927


def get_binny_lens_uncertainty_defaults(*, year: str) -> tuple[float, float]:
    """This just extracts the uncertainties from binny survey config.

    Note that since binny has more uncertainties you can extend this and play with outliers etc.
    """
    cfg = NZTomography.load_survey_config("lsst")
    entry = get_tomography_entry(cfg, role="lens", year=year)
    uncertainties = entry.get("uncertainties", {})

    mean_offset = float(uncertainties.get("mean_offset", 0.0))
    scatter_scale = float(uncertainties.get("scatter_scale", 0.03))

    return mean_offset, scatter_scale


def update_lens_uncertainties(
    cfg,
    *,
    year: str,
    mean_offset: float,
    scatter_scale: float,
):
    """This just updates the uncertainties from binny survey config so oyu can vary them."""
    cfg_new = copy.deepcopy(cfg)
    entry = get_tomography_entry(cfg_new, role="lens", year=year)

    if "uncertainties" not in entry:
        entry["uncertainties"] = {}

    entry["uncertainties"]["mean_offset"] = float(mean_offset)
    entry["uncertainties"]["scatter_scale"] = float(scatter_scale)

    return cfg_new


def get_result_z(result, cfg) -> np.ndarray:
    """This just returns the z values from binny survey config."""
    if hasattr(result, "z"):
        return np.asarray(result.z, dtype=float)

    zcfg = cfg["z_grid"]
    return np.linspace(
        float(zcfg["start"]),
        float(zcfg["stop"]),
        int(zcfg["n"]),
    )


def result_bins_to_list(result) -> list[np.ndarray]:
    """This just returns the bins values from binny survey config as a list of np.ndarray."""
    keys = sorted(result.bins.keys())
    return [np.asarray(result.bins[key], dtype=float) for key in keys]


def bin_centers_from_bins(z: np.ndarray, bins: list[np.ndarray]) -> np.ndarray:
    """This just returns the center values from binny survey config."""
    centers = []

    for nz_i in bins:
        norm = np.trapezoid(nz_i, z)
        if norm <= 0.0:
            raise ValueError("Encountered empty lens bin.")

        centers.append(np.trapezoid(z * nz_i, z) / norm)

    return np.asarray(centers, dtype=float)


def is_bias_bin_param(name: str) -> bool:
    """Return True for per-bin galaxy-bias parameters like bias_0, bias_1, etc."""
    return name.startswith("bias_") and name.removeprefix("bias_").isdigit()


def is_non_ccl_param(name: str) -> bool:
    """Return True for nuisance/model parameters that are not CCL cosmology inputs."""
    return (
        name in PHOTOZ_PARAM_NAMES
        or name == "bias_prefactor"
        or is_bias_bin_param(name)
    )


def build_lsst_lens_bins(
    *,
    year: str,
    mean_offset: float,
    scatter_scale: float,
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    cfg = NZTomography.load_survey_config("lsst")
    cfg = update_lens_uncertainties(
        cfg,
        year=year,
        mean_offset=mean_offset,
        scatter_scale=scatter_scale,
    )

    tomo = NZTomography()
    result = tomo.build_bins(
        cfg=cfg,
        role="lens",
        year=year,
    )

    z = get_result_z(result, cfg)
    bins = result_bins_to_list(result)

    return z, bins, cfg


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_bin_pairs(bin_pairs: str | None) -> list[tuple[int, int]]:
    if bin_pairs is None or not bin_pairs.strip():
        raise ValueError("Need --bin-pairs when custom correlations are used.")

    pairs = []

    for item in bin_pairs.split(","):
        item = item.strip()
        if not item:
            continue

        if "-" not in item:
            raise ValueError(
                f"Could not parse bin pair {item!r}. "
                f"Use format like 0-0,0-1,1-1."
                f"Upper triangle convention."
            )

        left, right = item.split("-", maxsplit=1)
        i = int(left)
        j = int(right)

        if i > j:
            i, j = j, i

        pairs.append((i, j))

    if not pairs:
        raise ValueError("No valid bin pairs were parsed from --bin-pairs.")

    return pairs


def resolve_bin_pairs(
    *,
    n_bins: int,
    correlations: str,
    custom_bin_pairs: str | None,
) -> list[tuple[int, int]]:
    if correlations == "diagonal":
        return [(i, i) for i in range(n_bins)]

    if correlations == "unique":
        return [(i, j) for i in range(n_bins) for j in range(i, n_bins)]

    if correlations == "custom":
        pairs = parse_bin_pairs(custom_bin_pairs)

        for i, j in pairs:
            if i < 0 or j < 0 or i >= n_bins or j >= n_bins:
                raise ValueError(
                    f"Custom bin pair {(i, j)} is outside valid range 0 to {n_bins - 1}."
                )

        return pairs

    raise ValueError(f"Unknown correlations mode {correlations!r}.")


def params_from_theta(
    theta: np.ndarray,
    theta_names: list[str],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    params = dict(defaults)

    if len(theta) != len(theta_names):
        raise ValueError(
            f"theta has length {len(theta)}, but theta_names has length "
            f"{len(theta_names)}."
        )

    for name, value in zip(theta_names, theta, strict=True):
        params[name] = float(value)

    return params


def validate_density_parameter_choice(theta_names: list[str]) -> None:
    if "Omega_m" in theta_names and "Omega_c" in theta_names:
        raise ValueError("Use either Omega_m or Omega_c in theta_names, not both.")


def add_derived_density_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(defaults)

    has_omega_m = "Omega_m" in defaults
    has_omega_c = "Omega_c" in defaults
    has_omega_b = "Omega_b" in defaults

    if not has_omega_b:
        raise ValueError("Need Omega_b in cosmology.")

    if has_omega_m and not has_omega_c:
        defaults["Omega_c"] = float(defaults["Omega_m"]) - float(defaults["Omega_b"])

    if has_omega_c and not has_omega_m:
        defaults["Omega_m"] = float(defaults["Omega_c"]) + float(defaults["Omega_b"])

    if float(defaults["Omega_c"]) <= 0.0:
        raise ValueError("Need Omega_m > Omega_b.")

    return defaults


def ccl_kwargs_from_params(params: dict[str, Any]) -> dict[str, Any]:
    ccl_kwargs = {}

    for name, value in params.items():
        if is_non_ccl_param(name):
            continue
        if value is None:
            continue
        ccl_kwargs[name] = value

    if "Omega_m" in ccl_kwargs:
        omega_m = float(ccl_kwargs.pop("Omega_m"))
        omega_b = float(ccl_kwargs["Omega_b"])
        omega_c = omega_m - omega_b

        if omega_c <= 0.0:
            raise ValueError("Need Omega_m > Omega_b.")

        ccl_kwargs["Omega_c"] = omega_c

    if "Omega_c" not in ccl_kwargs:
        raise ValueError("Need either Omega_m or Omega_c.")

    ccl_kwargs.setdefault("m_nu", 0.0)
    ccl_kwargs.setdefault("mass_split", "equal")

    return ccl_kwargs


def build_cosmology_from_params(params: dict[str, Any]):
    return ccl.Cosmology(**ccl_kwargs_from_params(params))


def linear_galaxy_bias(
    cosmo,
    *,
    z_centers: np.ndarray,
    bias_prefactor: float,
) -> np.ndarray:
    scale_factor = 1.0 / (1.0 + np.asarray(z_centers, dtype=float))
    growth = ccl.growth_factor(cosmo, scale_factor)
    return np.round(float(bias_prefactor) / growth, 6)


def add_bias_bin_defaults(
    defaults: dict[str, Any],
    *,
    year: str,
) -> dict[str, Any]:
    defaults = dict(defaults)

    mean_offset = float(defaults["mean_offset"])
    scatter_scale = float(defaults["scatter_scale"])
    bias_prefactor = float(defaults["bias_prefactor"])

    z, lens_bins, _ = build_lsst_lens_bins(
        year=year,
        mean_offset=mean_offset,
        scatter_scale=scatter_scale,
    )

    cosmo = build_cosmology_from_params(defaults)
    z_centers = bin_centers_from_bins(z, lens_bins)

    bias_values = linear_galaxy_bias(
        cosmo,
        z_centers=z_centers,
        bias_prefactor=bias_prefactor,
    )

    for i, value in enumerate(bias_values):
        defaults.setdefault(f"bias_{i}", float(value))

    return defaults


def scenario_label(
    *,
    correlations: str,
    mean_offset: float,
    scatter_scale: float,
    kmax: float | None = None,
    include_dali: bool = False,
) -> str:
    label = (
        f"{correlations}, "
        rf"$\Delta z={mean_offset:+.3f}$, "
        rf"$s_z={scatter_scale:.3f}$"
    )

    if kmax is not None:
        label += rf", $k_{{\max}}={kmax:g}$"

    if include_dali:
        label += ", DALI"

    return label


def scenario_slug(
    *,
    correlations: str,
    mean_offset: float,
    scatter_scale: float,
    kmax: float | None = None,
) -> str:
    dz = f"{mean_offset:+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
    sz = f"{scatter_scale:.4f}".replace(".", "p")

    if kmax is None:
        kmax_text = "none"
    else:
        kmax_text = f"{kmax:g}".replace(".", "p")

    return f"{correlations}_dz{dz}_sz{sz}_kmax{kmax_text}"


def safe_slug(text: str) -> str:
    return (
        text.replace(",", "_")
        .replace(" ", "_")
        .replace(".", "p")
        .replace("+", "p")
        .replace("-", "m")
        .replace("/", "_")
    )


def build_run_name(
    *,
    year: str,
    theta_names: list[str],
    correlations: list[str],
    mean_offsets: list[float],
    scatter_scales: list[float],
    method: str,
    include_dali: bool,
) -> str:
    parts = [
        f"lsst_y{year}",
        "gc",
        "_".join(theta_names),
        "corr_" + "_".join(correlations),
        "dz_" + "_".join(f"{value:+.3f}" for value in mean_offsets),
        "sz_" + "_".join(f"{value:.3f}" for value in scatter_scales),
        method,
    ]

    if include_dali:
        parts.append("dali")

    return safe_slug("__".join(parts))


def get_labels(theta_names: list[str]) -> list[str]:
    label_map = {
        "Omega_m": r"\Omega_m",
        "Omega_b": r"\Omega_b",
        "Omega_c": r"\Omega_c",
        "Omega_k": r"\Omega_k",
        "sigma8": r"\sigma_8",
        "h": r"h",
        "n_s": r"n_s",
        "w0": r"w_0",
        "wa": r"w_a",
        "mean_offset": r"\Delta z_{\rm lens}",
        "scatter_scale": r"\sigma_{z,\rm lens}",
        "bias_prefactor": r"b_{\rm pref}",
    }

    return [label_map.get(name, name) for name in theta_names]


def get_yaml_float_map(config: dict[str, Any], *keys: str) -> dict[str, float]:
    current = config

    for key in keys:
        current = current.get(key, {})
        if current is None:
            return {}

    return {
        str(name): float(value)
        for name, value in current.items()
        if value is not None
    }


def get_prior_sigma(
    *,
    theta_names: list[str],
    config: dict[str, Any],
    defaults: dict[str, Any],
) -> np.ndarray:
    prior_map = get_yaml_float_map(config, "fisher", "gaussian_priors")

    prior_sigma = []

    for name in theta_names:
        if name in prior_map:
            prior_sigma.append(prior_map[name])
        elif is_bias_bin_param(name):
            prior_sigma.append(0.1 * abs(float(defaults[name])))
        else:
            raise ValueError(
                f"Missing Gaussian prior for {name!r}. "
                "Add it under fisher.gaussian_priors."
            )

    return np.asarray(prior_sigma, dtype=float)


def get_step_map(config: dict[str, Any]) -> dict[str, float]:
    derivative_cfg = config.get("fisher", {}).get("derivative", {})
    kwargs = derivative_cfg.get("kwargs", {})
    stepsize = kwargs.get("stepsize", {})

    if isinstance(stepsize, dict):
        return {
            str(name): float(value)
            for name, value in stepsize.items()
            if value is not None
        }

    return {}


def ordered_stepsize(
    *,
    theta_names: list[str],
    config: dict[str, Any],
    override_stepsize: float | None,
) -> float | np.ndarray | None:
    if override_stepsize is not None:
        return float(override_stepsize)

    derivative_cfg = config.get("fisher", {}).get("derivative", {})
    kwargs = derivative_cfg.get("kwargs", {})
    stepsize = kwargs.get("stepsize", None)

    if stepsize is None:
        return 0.01

    if isinstance(stepsize, dict):
        missing = [name for name in theta_names if name not in stepsize]
        if missing:
            raise ValueError(
                "Missing derivative steps in YAML for varied parameters: "
                f"{missing}. Add them under fisher.derivative.kwargs.stepsize."
            )

        return np.asarray([float(stepsize[name]) for name in theta_names], dtype=float)

    return float(stepsize)


def build_derivative_settings(
    *,
    theta_names: list[str],
    config: dict[str, Any],
    method_override: str | None,
    stepsize_override: float | None,
) -> tuple[str, dict[str, Any]]:
    derivative_cfg = config.get("fisher", {}).get("derivative", {})

    method = method_override or derivative_cfg.get("method", "finite")
    kwargs = dict(derivative_cfg.get("kwargs", {}))

    if "stepsize" in kwargs:
        kwargs.pop("stepsize")

    stepsize = ordered_stepsize(
        theta_names=theta_names,
        config=config,
        override_stepsize=stepsize_override,
    )

    if stepsize is not None:
        kwargs["stepsize"] = stepsize

    return method, kwargs


def build_derivative_check_presets(names: list[str]) -> list[dict[str, Any]]:
    checks = []

    for name in names:
        if name not in DERIVATIVE_CHECK_PRESETS:
            raise ValueError(
                f"Unknown derivative check preset {name!r}. "
                f"Allowed presets are {sorted(DERIVATIVE_CHECK_PRESETS)}."
            )

        item = DERIVATIVE_CHECK_PRESETS[name]
        checks.append(
            {
                "name": name,
                "label": item["label"],
                "method": item["method"],
                "kwargs": dict(item["kwargs"]),
            }
        )

    return checks


def build_derivative_checks(
    *,
    config: dict[str, Any],
    theta_names: list[str],
    check_derivative_presets: str | None,
    check_finite_stepsizes: str | None,
    check_derivative_methods: str | None,
    check_derivative_stepsizes: str | None,
) -> list[dict[str, Any]]:
    checks = []

    if check_derivative_presets is not None:
        checks.extend(
            build_derivative_check_presets(parse_str_list(check_derivative_presets))
        )

    if check_finite_stepsizes is not None:
        for stepsize in parse_float_list(check_finite_stepsizes):
            checks.append(
                {
                    "name": f"ridders_h_{stepsize:g}",
                    "label": f"finite/ridders h={stepsize:g}",
                    "method": "finite",
                    "kwargs": {
                        "extrapolation": "ridders",
                        "stepsize": float(stepsize),
                    },
                }
            )

    if check_derivative_methods is not None or check_derivative_stepsizes is not None:
        if check_derivative_methods is None or check_derivative_stepsizes is None:
            raise ValueError(
                "Need both --check-derivative-methods and "
                "--check-derivative-stepsizes when using the legacy check options."
            )

        for method in parse_str_list(check_derivative_methods):
            for stepsize in parse_float_list(check_derivative_stepsizes):
                _, kwargs = build_derivative_settings(
                    theta_names=theta_names,
                    config=config,
                    method_override=method,
                    stepsize_override=stepsize,
                )
                checks.append(
                    {
                        "name": f"{method}_h_{stepsize:g}",
                        "label": f"{method} h={stepsize:g}",
                        "method": method,
                        "kwargs": kwargs,
                    }
                )

    return checks


def build_check_derivative_settings(
    *,
    theta_names: list[str],
    config: dict[str, Any],
    check_method: str,
    check_stepsize: float,
) -> tuple[str, dict[str, Any]]:
    _, kwargs = build_derivative_settings(
        theta_names=theta_names,
        config=config,
        method_override=check_method,
        stepsize_override=check_stepsize,
    )

    return check_method, kwargs


def get_dali_positive_params(config):
    return set(
        config.get("fisher", {})
        .get("dali", {})
        .get("positive_params", [])
    )


def dali_theta_names(theta_names, positive_params):
    return [
        f"log_{name}" if name in positive_params else name
        for name in theta_names
    ]


def to_dali_theta(theta0, theta_names, positive_params):
    out = []
    for value, name in zip(theta0, theta_names, strict=True):
        if name in positive_params:
            if value <= 0:
                raise ValueError(f"Cannot log-transform non-positive parameter {name}={value}.")
            out.append(np.log(float(value)))
        else:
            out.append(float(value))
    return np.asarray(out, dtype=float)


def from_dali_theta(theta_dali, theta_names, positive_params):
    out = []
    for value, name in zip(theta_dali, theta_names, strict=True):
        if name in positive_params:
            out.append(np.exp(float(value)))
        else:
            out.append(float(value))
    return np.asarray(out, dtype=float)


def build_base_defaults(
    *,
    config: dict[str, Any],
    year: str,
    mean_offset_override: float | None,
    scatter_scale_override: float | None,
    bias_prefactor_override: float | None,
) -> dict[str, Any]:
    defaults = {}

    defaults.update(config.get("cosmology", {}))
    defaults.update(config.get("nuisance", {}))

    binny_mean_offset, binny_scatter_scale = get_binny_lens_uncertainty_defaults(
        year=year
    )

    defaults.setdefault("mean_offset", binny_mean_offset)
    defaults.setdefault("scatter_scale", binny_scatter_scale)
    defaults.setdefault("bias_prefactor", 1.0)

    if mean_offset_override is not None:
        defaults["mean_offset"] = float(mean_offset_override)

    if scatter_scale_override is not None:
        defaults["scatter_scale"] = float(scatter_scale_override)

    if bias_prefactor_override is not None:
        defaults["bias_prefactor"] = float(bias_prefactor_override)

    return add_derived_density_defaults(defaults)



def kmax_to_inverse_mpc(
    *,
    kmax: float | None,
    kmax_units: str,
    h: float,
) -> float | None:
    if kmax is None:
        return None

    if kmax <= 0.0:
        raise ValueError("Need --kmax > 0 when scale cuts are used.")

    if kmax_units == "1/Mpc":
        return float(kmax)

    if kmax_units == "h/Mpc":
        return float(kmax) * float(h)

    raise ValueError(f"Unknown kmax units {kmax_units!r}.")


def build_pair_ell_cuts(
    *,
    cosmo,
    ell: np.ndarray,
    z_centers: np.ndarray,
    bin_pairs: list[tuple[int, int]],
    kmax: float | None,
    kmax_units: str,
    h: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    ell = np.asarray(ell, dtype=float)

    if kmax is None:
        return [ell.copy() for _ in bin_pairs], np.full(len(bin_pairs), np.inf)

    kmax_mpc = kmax_to_inverse_mpc(
        kmax=kmax,
        kmax_units=kmax_units,
        h=h,
    )

    scale_factors = 1.0 / (1.0 + np.asarray(z_centers, dtype=float))
    chi = np.asarray(ccl.comoving_radial_distance(cosmo, scale_factors), dtype=float)

    selected_ells = []
    ell_max_values = []

    for i, j in bin_pairs:
        chi_pair = min(float(chi[i]), float(chi[j]))
        ell_max = float(kmax_mpc) * chi_pair - 0.5
        ell_max_values.append(ell_max)

    ell_max_values = np.asarray(ell_max_values, dtype=float)
    common_ell_max = float(np.min(ell_max_values))
    ell_common = ell[ell <= common_ell_max]

    if ell_common.size == 0:
        raise ValueError(
            "Scale cut removed all ell values for at least one bin pair. "
            "Increase --kmax, decrease --ell-min, or use a wider ell grid."
        )

    selected_ells = [ell_common.copy() for _ in bin_pairs]

    return selected_ells, ell_max_values


def galaxy_clustering_power_spectra(
    theta: np.ndarray,
    *,
    theta_names: list[str],
    defaults: dict[str, Any],
    year: str,
    selected_ells: list[np.ndarray],
    correlations: str,
    custom_bin_pairs: str | None,
) -> np.ndarray:
    params = params_from_theta(theta, theta_names, defaults)

    mean_offset = float(params["mean_offset"])
    scatter_scale = float(params["scatter_scale"])
    bias_prefactor = float(params["bias_prefactor"])

    if scatter_scale <= 0.0:
        raise ValueError("Need lens scatter_scale > 0.")

    z, lens_bins, _ = build_lsst_lens_bins(
        year=year,
        mean_offset=mean_offset,
        scatter_scale=scatter_scale,
    )

    cosmo = build_cosmology_from_params(params)

    z_centers = bin_centers_from_bins(z, lens_bins)
    base_bias_values = linear_galaxy_bias(
        cosmo,
        z_centers=z_centers,
        bias_prefactor=bias_prefactor,
    )

    bias_values = np.asarray(
        [
            float(params.get(f"bias_{i}", base_bias_values[i]))
            for i in range(len(lens_bins))
        ],
        dtype=float,
    )

    tracers = []
    for nz_i, b_i in zip(lens_bins, bias_values, strict=True):
        bias = np.full_like(z, b_i, dtype=float)
        tracers.append(
            ccl.NumberCountsTracer(
                cosmo,
                has_rsd=False,
                dndz=(z, nz_i),
                bias=(z, bias),
                mag_bias=None,
            )
        )

    bin_pairs = resolve_bin_pairs(
        n_bins=len(tracers),
        correlations=correlations,
        custom_bin_pairs=custom_bin_pairs,
    )

    if len(selected_ells) != len(bin_pairs):
        raise ValueError(
            "selected_ells length does not match the number of bin pairs. "
            "Build scale cuts at the fiducial point for the same correlation mode."
        )

    data_parts = []
    for (i, j), ell_ij in zip(bin_pairs, selected_ells, strict=True):
        data_parts.append(
            ccl.angular_cl(
                cosmo,
                tracers[i],
                tracers[j],
                ell_ij,
            )
        )

    return np.concatenate(data_parts)


def build_clustering_sacc(
    *,
    z: np.ndarray,
    lens_bins: list[np.ndarray],
    selected_ells: list[np.ndarray],
    cls: np.ndarray,
    bin_pairs: list[tuple[int, int]],
) -> object:
    import sacc

    s = sacc.Sacc()

    tracer_names = []
    for i, nz_i in enumerate(lens_bins):
        name = f"DESgc__{i}"
        tracer_names.append(name)

        s.add_tracer(
            "NZ",
            name,
            quantity="galaxy_density",
            spin=0,
            z=z,
            nz=nz_i,
        )

    expected_size = sum(ell_ij.size for ell_ij in selected_ells)
    if cls.size != expected_size:
        raise ValueError(
            f"cls has size {cls.size}, but expected {expected_size} from "
            "the selected ell arrays."
        )

    k = 0
    for (i, j), ell_ij in zip(bin_pairs, selected_ells, strict=True):
        n_ell = ell_ij.size
        cl_ij = cls[k : k + n_ell]
        s.add_ell_cl(
            "cl_00",
            tracer_names[i],
            tracer_names[j],
            ell_ij,
            cl_ij,
        )
        k += n_ell

    return s, tracer_names


def tjpcov_clustering_covariance(
    *,
    theta0: np.ndarray,
    theta_names: list[str],
    defaults: dict[str, Any],
    year: str,
    selected_ells: list[np.ndarray],
    correlations: str,
    custom_bin_pairs: str | None,
    cov_jitter: float = 1.0e-12,
) -> np.ndarray:
    params = params_from_theta(theta0, theta_names, defaults)

    mean_offset = float(params["mean_offset"])
    scatter_scale = float(params["scatter_scale"])
    bias_prefactor = float(params["bias_prefactor"])

    z, lens_bins, cfg = build_lsst_lens_bins(
        year=year,
        mean_offset=mean_offset,
        scatter_scale=scatter_scale,
    )

    cosmo = build_cosmology_from_params(params)

    z_centers = bin_centers_from_bins(z, lens_bins)
    bias_values = linear_galaxy_bias(
        cosmo,
        z_centers=z_centers,
        bias_prefactor=bias_prefactor,
    )

    bin_pairs = resolve_bin_pairs(
        n_bins=len(lens_bins),
        correlations=correlations,
        custom_bin_pairs=custom_bin_pairs,
    )

    cls = galaxy_clustering_power_spectra(
        theta0,
        theta_names=theta_names,
        defaults=defaults,
        year=year,
        selected_ells=selected_ells,
        correlations=correlations,
        custom_bin_pairs=custom_bin_pairs,
    )

    sacc_file, tracer_names = build_clustering_sacc(
        z=z,
        lens_bins=lens_bins,
        selected_ells=selected_ells,
        cls=cls,
        bin_pairs=bin_pairs,
    )

    n_gal_arcmin2 = get_lens_number_density(cfg, year=year)
    fsky = get_fsky(cfg, year=year)

    tjpcov_cfg = {
        "tjpcov": {
            "sacc_file": sacc_file,
            "cosmo": cosmo,
            "cov_type": ["FourierGaussianFsky"],
            "fsky": fsky,
            "use_mpi": False,
        }
    }

    for name, b_i in zip(tracer_names, bias_values, strict=True):
        tjpcov_cfg["tjpcov"][f"Ngal_{name}"] = n_gal_arcmin2
        tjpcov_cfg["tjpcov"][f"bias_{name}"] = float(b_i)

    with contextlib.redirect_stdout(io.StringIO()):
        cov = np.asarray(CovarianceCalculator(tjpcov_cfg).get_covariance(), dtype=float)

    if cov_jitter > 0.0:
        scale = np.max(np.abs(np.diag(cov)))
        cov = cov + cov_jitter * scale * np.eye(cov.shape[0])

    return cov


def run_single_scenario(
    *,
    theta_names: list[str],
    base_defaults: dict[str, Any],
    config: dict[str, Any],
    year: str,
    ell: np.ndarray,
    correlations: str,
    custom_bin_pairs: str | None,
    mean_offset: float,
    scatter_scale: float,
    method: str,
    derivative_kwargs: dict[str, Any],
    derivative_checks: list[dict[str, Any]],
    include_dali: bool,
    forecast_order: int,
    cov_jitter: float,
    kmax: float | None,
    kmax_units: str,
):
    defaults = dict(base_defaults)
    defaults["mean_offset"] = float(mean_offset)
    defaults["scatter_scale"] = float(scatter_scale)

    theta0 = np.array(
        [defaults[name] for name in theta_names],
        dtype=float,
    )

    fid_params = params_from_theta(theta0, theta_names, defaults)
    print("  building nz / tomography")
    fid_z, fid_lens_bins, cfg_check = build_lsst_lens_bins(
        year=year,
        mean_offset=float(fid_params["mean_offset"]),
        scatter_scale=float(fid_params["scatter_scale"]),
    )

    print("  done: nz / tomography")

    fid_cosmo = build_cosmology_from_params(fid_params)
    z_centers = bin_centers_from_bins(fid_z, fid_lens_bins)
    bin_pairs = resolve_bin_pairs(
        n_bins=len(fid_lens_bins),
        correlations=correlations,
        custom_bin_pairs=custom_bin_pairs,
    )

    selected_ells, ell_max_per_pair = build_pair_ell_cuts(
        cosmo=fid_cosmo,
        ell=ell,
        z_centers=z_centers,
        bin_pairs=bin_pairs,
        kmax=kmax,
        kmax_units=kmax_units,
        h=float(fid_params["h"]),
    )

    def model(theta: np.ndarray) -> np.ndarray:
        return galaxy_clustering_power_spectra(
            theta,
            theta_names=theta_names,
            defaults=defaults,
            year=year,
            selected_ells=selected_ells,
            correlations=correlations,
            custom_bin_pairs=custom_bin_pairs,
        )

    print("  computing fiducial Cl data vector")
    y0 = model(theta0)

    print(f"  done: data vector length = {y0.size}")

    print("  computing covariance")
    cov = tjpcov_clustering_covariance(
        theta0=theta0,
        theta_names=theta_names,
        defaults=defaults,
        year=year,
        selected_ells=selected_ells,
        correlations=correlations,
        custom_bin_pairs=custom_bin_pairs,
        cov_jitter=cov_jitter,
    )

    print(f"  done: covariance shape = {cov.shape}")

    fk = ForecastKit(
        function=model,
        theta0=theta0,
        cov=cov,
    )

    print("  computing Fisher matrix")
    fisher = fk.fisher(method=method, **derivative_kwargs)

    consistency_checks = []

    for check in derivative_checks:
        check_method = check["method"]
        check_kwargs = dict(check["kwargs"])

        try:
            if check_method is None:
                fisher_check = fk.fisher(**check_kwargs)
            else:
                fisher_check = fk.fisher(method=check_method, **check_kwargs)
        except Exception as exc:
            consistency_checks.append(
                {
                    "name": check["name"],
                    "label": check["label"],
                    "method": check_method or "adaptive",
                    "kwargs": check_kwargs,
                    "fisher": None,
                    "relative_difference": np.nan,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue

        denom = max(np.linalg.norm(fisher), 1.0e-30)
        rel_diff = np.linalg.norm(fisher_check - fisher) / denom

        consistency_checks.append(
            {
                "name": check["name"],
                "label": check["label"],
                "method": check_method or "adaptive",
                "kwargs": check_kwargs,
                "fisher": fisher_check,
                "relative_difference": rel_diff,
                "status": "ok",
                "error": "",
            }
        )

    prior_sigma = get_prior_sigma(
        theta_names=theta_names,
        config=config,
        defaults=defaults,
    )
    fisher_post = fisher + np.diag(1.0 / prior_sigma**2)

    fisher_samples = fk.getdist_fisher_gaussian(
        fisher=fisher_post,
        names=theta_names,
        labels=get_labels(theta_names),
        label=scenario_label(
            correlations=correlations,
            mean_offset=mean_offset,
            scatter_scale=scatter_scale,
            kmax=kmax,
        ),
    )

    print("  done: Fisher matrix")

    dali = None
    dali_samples = None

    if include_dali:
        positive_params = get_dali_positive_params(config)

        theta0_dali = to_dali_theta(
            theta0,
            theta_names,
            positive_params,
        )

        dali_names = dali_theta_names(
            theta_names,
            positive_params,
        )

        prior_sigma_dali = []
        for value, name, sigma in zip(theta0, theta_names, prior_sigma, strict=True):
            if name in positive_params:
                prior_sigma_dali.append(float(sigma) / float(value))
            else:
                prior_sigma_dali.append(float(sigma))

        prior_sigma_dali = np.asarray(prior_sigma_dali, dtype=float)

        def dali_model(theta_dali):
            theta_physical = from_dali_theta(
                theta_dali,
                theta_names,
                positive_params,
            )
            return model(theta_physical)

        fk_dali = ForecastKit(
            function=dali_model,
            theta0=theta0_dali,
            cov=cov,
        )

        print("  computing DALI tensors")
        dali = fk_dali.dali(
            method=method,
            forecast_order=forecast_order,
            **derivative_kwargs,
        )

        prior_terms = [
            (
                "gaussian",
                {
                    "mean": theta0_dali,
                    "cov": np.diag(prior_sigma_dali ** 2),
                },
            ),
        ]
        print("  done: DALI tensors")

        print("  building DALI samples")

        dali_samples = fk_dali.getdist_dali_emcee(
            dali=dali,
            names=dali_names,
            labels=get_labels(dali_names),
            label=scenario_label(
                correlations=correlations,
                mean_offset=mean_offset,
                scatter_scale=scatter_scale,
                kmax=kmax,
                include_dali=True,
            ),
            prior_terms=prior_terms,
        )

        print("  done: DALI samples")

    bias_values = linear_galaxy_bias(
        fid_cosmo,
        z_centers=z_centers,
        bias_prefactor=float(defaults["bias_prefactor"]),
    )

    return {
        "label": scenario_label(
            correlations=correlations,
            mean_offset=mean_offset,
            scatter_scale=scatter_scale,
            kmax=kmax,
        ),
        "slug": scenario_slug(
            correlations=correlations,
            mean_offset=mean_offset,
            scatter_scale=scatter_scale,
            kmax=kmax,
        ),
        "theta0": theta0,
        "y0": y0,
        "cov": cov,
        "fisher": fisher,
        "fisher_post": fisher_post,
        "fisher_samples": fisher_samples,
        "dali": dali,
        "dali_samples": dali_samples,
        "correlations": correlations,
        "mean_offset": mean_offset,
        "scatter_scale": scatter_scale,
        "bin_pairs": bin_pairs,
        "base_ell": ell,
        "selected_ells": selected_ells,
        "ell_max_per_pair": ell_max_per_pair,
        "kmax": np.nan if kmax is None else float(kmax),
        "kmax_units": kmax_units,
        "z_centers": z_centers,
        "bias_values": bias_values,
        "n_gal_arcmin2": get_lens_number_density(cfg_check, year=year),
        "fsky": get_fsky(cfg_check, year=year),
        "prior_sigma": prior_sigma,
        "consistency_checks": consistency_checks,
    }


def save_scenario_arrays(
    *,
    output_dir: Path,
    theta_names: list[str],
    ell: np.ndarray,
    result: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{result['slug']}.npz"

    payload = {
        "theta_names": np.asarray(theta_names, dtype=object),
        "theta0": result["theta0"],
        "ell": ell,
        "base_ell": result["base_ell"],
        "selected_ells": np.asarray(result["selected_ells"], dtype=object),
        "ell_max_per_pair": result["ell_max_per_pair"],
        "kmax": np.asarray(result["kmax"]),
        "kmax_units": np.asarray(result["kmax_units"], dtype=object),
        "y0": result["y0"],
        "cov": result["cov"],
        "fisher": result["fisher"],
        "fisher_post": result["fisher_post"],
        "prior_sigma": result["prior_sigma"],
        "correlations": np.asarray(result["correlations"], dtype=object),
        "mean_offset": np.asarray(result["mean_offset"]),
        "scatter_scale": np.asarray(result["scatter_scale"]),
        "bin_pairs": np.asarray(result["bin_pairs"], dtype=int),
        "z_centers": result["z_centers"],
        "bias_values": result["bias_values"],
        "n_gal_arcmin2": np.asarray(result["n_gal_arcmin2"]),
        "fsky": np.asarray(result["fsky"]),
    }

    if result["dali"] is not None:
        payload["dali"] = np.asarray(result["dali"], dtype=object)

    if result["consistency_checks"]:
        payload["consistency_check_methods"] = np.asarray(
            [item["method"] for item in result["consistency_checks"]],
            dtype=object,
        )
        payload["consistency_check_labels"] = np.asarray(
            [item["label"] for item in result["consistency_checks"]],
            dtype=object,
        )
        payload["consistency_check_kwargs"] = np.asarray(
            [item["kwargs"] for item in result["consistency_checks"]],
            dtype=object,
        )
        payload["consistency_check_relative_differences"] = np.asarray(
            [item["relative_difference"] for item in result["consistency_checks"]],
            dtype=float,
        )
        payload["consistency_check_fishers"] = np.asarray(
            [item["fisher"] for item in result["consistency_checks"]],
            dtype=object,
        )

    np.savez(path, **payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DerivKit galaxy-clustering CCL scenario comparisons."
    )

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--year", type=str, default="1", choices=["1", "10"])
    parser.add_argument("--n-ell", type=int, default=DEFAULT_N_ELL)
    parser.add_argument("--ell-min", type=float, default=DEFAULT_ELL_MIN)
    parser.add_argument("--ell-max", type=float, default=DEFAULT_ELL_MAX)
    parser.add_argument(
        "--kmax",
        type=float,
        default=None,
        help=(
            "Optional fiducial kmax scale cut. If set, each bin pair keeps only "
            "ell + 0.5 <= kmax * min(chi_i, chi_j)."
        ),
    )
    parser.add_argument(
        "--kmax-units",
        type=str,
        default=DEFAULT_KMAX_UNITS,
        choices=["h/Mpc", "1/Mpc"],
        help="Units for --kmax. Use h/Mpc for conventional h-scaled cuts.",
    )

    parser.add_argument("--bias-prefactor", type=float, default=None)
    parser.add_argument("--mean-offset", type=float, default=None)
    parser.add_argument("--scatter-scale", type=float, default=None)

    parser.add_argument(
        "--mean-offsets",
        type=str,
        default=None,
        help="Comma-separated photo-z mean shifts for scenario sweeps.",
    )

    parser.add_argument(
        "--scatter-scales",
        type=str,
        default=None,
        help="Comma-separated photo-z scatter/stretch values for scenario sweeps.",
    )

    parser.add_argument(
        "--kmax-values",
        type=str,
        default=None,
        help="Comma-separated kmax values for scale-cut scenario sweeps, e.g. 0.2,0.3.",
    )

    parser.add_argument(
        "--theta-names",
        type=str,
        default=None,
        help=(
            "Comma-separated varied parameters. If omitted, uses "
            "fisher.var_pars from YAML."
        ),
    )

    parser.add_argument(
        "--correlations",
        type=str,
        default="diagonal",
        choices=["diagonal", "unique", "custom"],
        help="Single correlation mode used when --scenario-correlations is not given.",
    )

    parser.add_argument(
        "--scenario-correlations",
        type=str,
        default=None,
        help="Comma-separated correlation modes to compare, e.g. diagonal,unique.",
    )

    parser.add_argument(
        "--bin-pairs",
        type=str,
        default=None,
        help="Custom bin pairs for custom correlations, e.g. 0-0,0-1,1-1.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Override derivative method from YAML, e.g. finite.",
    )

    parser.add_argument(
        "--stepsize",
        type=float,
        default=None,
        help="Override YAML derivative steps with one scalar stepsize.",
    )

    parser.add_argument(
        "--check-derivative-methods",
        type=str,
        default=None,
        help="Comma-separated methods for internal Fisher consistency checks.",
    )

    parser.add_argument(
        "--check-derivative-stepsizes",
        type=str,
        default=None,
        help="Comma-separated scalar stepsizes for internal consistency checks.",
    )

    parser.add_argument(
        "--check-finite-stepsizes",
        type=str,
        default=None,
        help=(
            "Comma-separated finite/Ridders stepsizes to compare, e.g. "
            "1e-4,1e-3,1e-2. Only finite derivatives use these."
        ),
    )

    parser.add_argument(
        "--check-derivative-presets",
        type=str,
        default=None,
        help=(
            "Comma-separated derivative backend presets to compare, e.g. "
            "ridders_default,local_polynomial,adaptive."
        ),
    )

    parser.add_argument("--forecast-order", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--include-dali", action="store_true")
    parser.add_argument("--cov-jitter", type=float, default=1.0e-12)

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name used to auto-build plot and array output paths.",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path for saved contour plot. If omitted, built from --run-name.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for saved scenario arrays. If omitted, built from --run-name.",
    )

    parser.add_argument("--show", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_yaml(args.config)
    ell = np.geomspace(args.ell_min, args.ell_max, args.n_ell)

    base_defaults = build_base_defaults(
        config=config,
        year=args.year,
        mean_offset_override=args.mean_offset,
        scatter_scale_override=args.scatter_scale,
        bias_prefactor_override=args.bias_prefactor,
    )

    base_defaults = add_bias_bin_defaults(
        base_defaults,
        year=args.year,
    )

    if args.theta_names is not None:
        theta_names = parse_str_list(args.theta_names)
    else:
        theta_names = list(config.get("fisher", {}).get("var_pars", []))

    if not theta_names:
        raise ValueError("Need --theta-names or fisher.var_pars in the YAML config.")

    validate_density_parameter_choice(theta_names)

    method, derivative_kwargs = build_derivative_settings(
        theta_names=theta_names,
        config=config,
        method_override=args.method,
        stepsize_override=args.stepsize,
    )

    mean_offsets = (
        parse_float_list(args.mean_offsets)
        if args.mean_offsets is not None
        else [float(base_defaults["mean_offset"])]
    )

    scatter_scales = (
        parse_float_list(args.scatter_scales)
        if args.scatter_scales is not None
        else [float(base_defaults["scatter_scale"])]
    )

    scenario_correlations = (
        parse_str_list(args.scenario_correlations)
        if args.scenario_correlations is not None
        else [args.correlations]
    )

    kmax_values = (
        parse_float_list(args.kmax_values)
        if getattr(args, "kmax_values", None) is not None
        else [args.kmax]
    )

    valid_correlations = {"diagonal", "unique", "custom"}
    for corr in scenario_correlations:
        if corr not in valid_correlations:
            raise ValueError(
                f"Unknown correlation mode {corr!r}. "
                f"Allowed values are {sorted(valid_correlations)}."
            )

    derivative_checks = build_derivative_checks(
        config=config,
        theta_names=theta_names,
        check_derivative_presets=args.check_derivative_presets,
        check_finite_stepsizes=getattr(args, "check_finite_stepsizes", None),
        check_derivative_methods=args.check_derivative_methods,
        check_derivative_stepsizes=args.check_derivative_stepsizes,
    )

    run_name = args.run_name
    if run_name is None:
        run_name = build_run_name(
            year=args.year,
            theta_names=theta_names,
            correlations=scenario_correlations,
            mean_offsets=mean_offsets,
            scatter_scales=scatter_scales,
            method=method,
            include_dali=args.include_dali,
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("outputs/data/gc_scenario_forecasts") / run_name
    )

    plot_path = (
        Path(args.save)
        if args.save is not None
        else Path("outputs/plots") / f"{run_name}.pdf"
    )

    results = []
    samples = []

    print("Running galaxy-clustering scenario forecasts")
    print(f"run name             = {run_name}")
    print(f"theta names          = {theta_names}")
    print(f"scenario correlations= {scenario_correlations}")
    print(f"mean offsets         = {mean_offsets}")
    print(f"scatter scales       = {scatter_scales}")
    print(f"method               = {method}")
    print(f"derivative kwargs    = {derivative_kwargs}")
    print(f"kmax values          = {kmax_values}")
    print(f"kmax units           = {args.kmax_units}")
    print(f"include DALI         = {args.include_dali}")
    print(f"output dir           = {output_dir}")
    print(f"plot path            = {plot_path}")

    if derivative_checks:
        print("derivative checks    =")
        for check in derivative_checks:
            print(f"  - {check['label']}: method={check['method']}, kwargs={check['kwargs']}")

    for correlations in scenario_correlations:
        for mean_offset in mean_offsets:
            for scatter_scale in scatter_scales:
                for kmax in kmax_values:
                    label = scenario_label(
                        correlations=correlations,
                        mean_offset=mean_offset,
                        scatter_scale=scatter_scale,
                        kmax=kmax,
                    )
                    print()
                    print(f"Running scenario: {label}")

                    result = run_single_scenario(
                        theta_names=theta_names,
                        base_defaults=base_defaults,
                        config=config,
                        year=args.year,
                        ell=ell,
                        correlations=correlations,
                        custom_bin_pairs=args.bin_pairs,
                        mean_offset=mean_offset,
                        scatter_scale=scatter_scale,
                        method=method,
                        derivative_kwargs=derivative_kwargs,
                        derivative_checks=derivative_checks,
                        include_dali=args.include_dali,
                        forecast_order=args.forecast_order,
                        cov_jitter=args.cov_jitter,
                        kmax=kmax,
                        kmax_units=args.kmax_units,
                    )

                    save_scenario_arrays(
                        output_dir=output_dir,
                        theta_names=theta_names,
                        ell=ell,
                        result=result,
                    )

                    results.append(result)
                    samples.append(result["fisher_samples"])

                    for check in result["consistency_checks"]:
                        if check.get("status") != "ok":
                            continue

                        prior_sigma = result["prior_sigma"]
                        fisher_check_post = check["fisher"] + np.diag(
                            1.0 / prior_sigma**2
                        )

                        check_samples = ForecastKit(
                            function=lambda theta: theta,
                            theta0=result["theta0"],
                            cov=np.eye(len(theta_names)),
                        ).getdist_fisher_gaussian(
                            fisher=fisher_check_post,
                            names=theta_names,
                            labels=get_labels(theta_names),
                            label=f"{result['label']}, {check['label']}",
                        )

                        samples.append(check_samples)

                    if args.include_dali and result["dali_samples"] is not None:
                        samples.append(result["dali_samples"])

                    print(f"data length = {result['y0'].size}")
                    print(f"cov shape   = {result['cov'].shape}")
                    print(f"bin pairs   = {result['bin_pairs']}")
                    print(
                        "ell counts  = "
                        f"{[ell_ij.size for ell_ij in result['selected_ells']]}"
                    )
                    if kmax is not None:
                        print(
                            "ell max/pair= "
                            f"{np.round(result['ell_max_per_pair'], 1).tolist()}"
                        )

                    for check in result["consistency_checks"]:
                        if check.get("status") == "ok":
                            print(
                                "consistency check: "
                                f"{check['label']}, "
                                f"rel_diff={check['relative_difference']:.3e}"
                            )
                        else:
                            print(
                                "consistency check failed/skipped: "
                                f"{check['label']}, "
                                f"error={check['error']}"
                            )

    colors = cmr.take_cmap_colors(
        "cmr.pride_r",
        len(samples),
        cmap_range=(0.2, 0.8),
        return_fmt="hex",
    )

    print()
    print("Building combined triangle plot")

    width = 7 if len(theta_names) <= 3 else 9
    plotter = getdist_plots.get_subplot_plotter(width_inch=width)

    plotter.triangle_plot(
        samples,
        params=theta_names,
        filled=False,
        contour_colors=colors,
        contour_lws=[2.0 for _ in samples],
        contour_ls=["-" for _ in samples],
    )

    if plotter.fig is not None:
        title = f"Galaxy clustering scenario comparison: LSST Y{args.year}"
        title += ", Fisher and DALI" if args.include_dali else ", Fisher only"

        plotter.fig.suptitle(
            title,
            y=1.02,
            fontsize=12,
        )
        plotter.fig.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved arrays to: {output_dir}")

    print()
    print("Summary")
    for result in results:
        print(
            f"{result['label']}: "
            f"n_pairs={len(result['bin_pairs'])}, "
            f"data_len={result['y0'].size}, "
            f"cov_shape={result['cov'].shape}"
        )

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
