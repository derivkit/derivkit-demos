"""DerivKit — ForecastKit 2D DALI Contour Demo.

Summary
-------
This demo shows how to use DerivKit's ForecastKit to build local DALI
(Derivative Approximation of the Likelihood; see arXiv:1401.6892)
expansions in a nonlinear 2D toy model, draw DALI samples, and visualize
the resulting contour structure in parameter space.

Usage
-----
Run the script version from the command line, for example:

    python -m scripts.forecast-kit-dali-2d --method adaptive
    python -m scripts.forecast-kit-dali-2d --method adaptive --plot

Requirements
------------
- derivkit installed and importable in your Python environment.
- getdist installed and importable in your Python environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import plots as getdist_plots

from derivkit import ForecastKit
from utils.style import apply_plot_style


def model_2d(theta) -> np.ndarray:
    """Nonlinear 2D toy model with curved degeneracy."""
    x, eps = float(theta[0]), float(theta[1])

    k = 3.0
    a = 4.0
    c = 6.0

    o1 = 1e2 * np.exp((x - k * eps) ** 2) * np.exp(a * eps)
    o2 = 4e1 * np.exp(0.5 * x) * (1.0 + 0.3 * eps + c * eps**3)

    return np.array([o1, o2], dtype=float)


def parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="forecastkit dali demo (2d)")
    ap.add_argument("--plot", action="store_true", help="show the figure instead of saving")
    ap.add_argument(
        "--method",
        type=str,
        default="adaptive",
        help='derivative backend (e.g., "adaptive", "finite")',
    )
    ap.add_argument("--dpi", type=int, default=150, help="figure dpi (default: 150)")
    return ap.parse_args()


def main() -> None:
    """Run the 2D ForecastKit DALI demo."""
    args = parse_args()

    apply_plot_style()

    dk_red = "#f21901"
    line_width = 1.5

    theta0_2d = np.array([0.18, 0.02], dtype=float)
    cov_2d = np.array(
        [
            [1.0, 0.95],
            [0.95, 1.0],
        ],
        dtype=float,
    )

    fk_2d = ForecastKit(function=model_2d, theta0=theta0_2d, cov=cov_2d)

    dali_2d = fk_2d.dali(method=args.method, forecast_order=2)
    F_2d = dali_2d[1][0]
    D1_2d, D2_2d = dali_2d[2]

    print("\n2D Fisher matrix F:")
    print(F_2d)
    print("\n2D DALI D1 tensor shape:", D1_2d.shape)
    print("2D DALI D2 tensor shape:", D2_2d.shape)

    samples_2d = fk_2d.getdist_dali_emcee(
        dali=dali_2d,
        names=["x", "eps"],
        labels=[r"x", r"\epsilon"],
        label="DALI",
    )

    plotter = getdist_plots.get_subplot_plotter(width_inch=5)
    plotter.settings.linewidth_contour = line_width
    plotter.settings.linewidth = line_width
    plotter.settings.figure_legend_frame = False
    plotter.settings.legend_rect_border = False
    plotter.settings.axes_labelsize = 25

    plotter.triangle_plot(
        [samples_2d],
        params=["x", "eps"],
        filled=False,
        contour_colors=[dk_red],
        contour_lws=[line_width],
        contour_ls=["-"],
    )

    out = Path("plots/dali_plot_2d.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.export(str(out))
    print(f"saved: {out}")
    print("\n2D sample rows:", samples_2d.numrows)

    if args.plot:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
