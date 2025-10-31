"""Defines a general plot style."""

import matplotlib as mpl
from cycler import cycler

__all__ = [
    "apply_plot_style",
    "DEFAULT_COLORS",
    "DEFAULT_FONTSIZE",
    "DEFAULT_LINEWIDTH",
    "DEFAULT_MARKERSIZE",
    "DEFAULT_MARKEREDGEWIDTH",
    "use_blue",
    "use_yellow",
    "use_red",
]

DEFAULT_LINEWIDTH = 1.5
DEFAULT_FONTSIZE = 15
DEFAULT_MARKERSIZE = 10
DEFAULT_MARKEREDGEWIDTH = DEFAULT_LINEWIDTH
DEFAULT_COLORS = {
    "blue": "#3b9ab2",
    "yellow": "#e1af00",
    "red": "#f21901",
}

def _rgba(hex_color: str, alpha: float) -> tuple:
    c = mpl.colors.to_rgba(hex_color)
    return c[0], c[1], c[2], alpha

def apply_plot_style(*, base: str | None = None,
                     linewidth: float | None = None,
                     fontsize: int | None = None,
                     markersize: int | None = None,
                     markeredgewidth: float | None = None):
    """Apply a general plot style to matplotlib."""
    base = base or DEFAULT_COLORS["blue"]
    linewidth  = linewidth  if linewidth  is not None else DEFAULT_LINEWIDTH
    fontsize   = fontsize   if fontsize   is not None else DEFAULT_FONTSIZE
    markersize = markersize if markersize is not None else DEFAULT_MARKERSIZE
    markeredgewidth = markeredgewidth if markeredgewidth is not None else DEFAULT_MARKEREDGEWIDTH

    blue_strong = _rgba(base, 1.00)  # lines, labels, ticks, edges
    blue_medium = _rgba(base, 0.85)  # axes spine, legend edge
    blue_soft   = _rgba(base, 0.55)  # grid lines & minor ticks

    mpl.rcParams.update(
        {
            # ---- fonts / math ----
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": fontsize,          # global base size
            "axes.titlesize": fontsize,     # axes title
            "axes.labelsize": fontsize,     # x/y labels
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "legend.fontsize": fontsize - 1,
            "legend.title_fontsize": fontsize,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "it",
            "mathtext.rm": "DejaVu Sans",
            "mathtext.it": "DejaVu Sans:italic",
            "mathtext.bf": "DejaVu Sans:bold",
            "axes.unicode_minus": False,

            # ---- global color cycle (single blue) ----
            "axes.prop_cycle": cycler(color=[base]),

            # ---- line/marker defaults ----
            "lines.linewidth": linewidth,
            "lines.markersize": markersize,
            "lines.marker": "none",
            "lines.markeredgewidth": markeredgewidth,
            "patch.linewidth": markeredgewidth,  # legend box / rectangles
            "axes.linewidth": markeredgewidth,  # axes spines (add this line)

            # (optional) scatter defaults to match lines
            "scatter.marker": "o",
            "scatter.edgecolors": "face",

            # ---- monochrome color mapping ----
            "text.color": blue_strong,
            "axes.labelcolor": blue_strong,
            "axes.titlecolor": blue_strong,
            "axes.edgecolor": blue_medium,
            "xtick.color": blue_strong,
            "ytick.color": blue_strong,

            # ---- ticks ----
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,

            # ---- grid ----
            "axes.grid": False,
            "grid.color": blue_soft,
            "grid.alpha": 1.0,
            "grid.linestyle": ":",
            "grid.linewidth": 0.9 * markeredgewidth, # optional consistency

            # ---- legend ----
            "legend.edgecolor": blue_medium,
            "legend.facecolor": "white",
            "legend.framealpha": 0.92,
            "legend.labelcolor": blue_strong,

            # ---- figure / axes faces ----
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",

            # ---- patches (bars/hists) ----
            "patch.edgecolor": blue_strong,
            "patch.facecolor": _rgba(base, 0.15),
        }
    )

def use_blue(**kw):
    """Apply the plot style with blue as the base color."""
    return apply_plot_style(base=DEFAULT_COLORS["blue"], **kw)


def use_yellow(**kw):
    """Apply the plot style with yellow as the base color."""
    return apply_plot_style(base=DEFAULT_COLORS["yellow"], **kw)


def use_red(**kw):
    """Apply the plot style with red as the base color."""
    return apply_plot_style(base=DEFAULT_COLORS["red"], **kw)
