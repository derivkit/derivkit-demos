"""Adaptive Fit Derivative Demo (10 nonlinear targets, defaults only)."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from derivkit.derivative_kit import DerivativeKit
from common.style import (
    apply_plot_style,
    _DEFAULT_COLORS,
    _DEFAULT_FONTSIZE,
    _DEFAULT_LINEWIDTH,
    _DEFAULT_MARKERSIZE,
)
from common.formatters import format_value_with_uncertainty
from common.utils import resolve_outdir, save_fig

blue_color = _DEFAULT_COLORS["blue"]
red_color  = _DEFAULT_COLORS["red"]

x0 = 0.30
order = 1

def slope_estimator(f_clean_fn) -> float:
    dk = DerivativeKit(f_clean_fn, x0)
    val = dk.adaptive.differentiate(order)  # defaults only
    return float(np.asarray(val).ravel()[0])

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

FUNS = [
    dict(
        key="exp-sin-poly-cos",
        label="exp(sin)+poly−cos",
        latex=r"$f(x)=e^{\,0.6\sin(3x)+0.3x}+0.15\,x^{3}-0.2\,\cos(5x)$",
        f=lambda x: np.exp(0.6*np.sin(3.0*x)+0.3*x) + 0.15*x**3 - 0.2*np.cos(5.0*x),
        fp=lambda x: (
            np.exp(0.6*np.sin(3.0*x)+0.3*x) * (0.6*3.0*np.cos(3.0*x) + 0.3)
            + 0.45*x**2 + np.sin(5.0*x)
        ),
    ),
    dict(
        key="sin-cos-poly",
        label="sin+cos+poly",
        latex=r"$f(x)=\sin(6x)+0.4\,\cos(2x)+0.2\,x^{2}-0.1\,x$",
        f=lambda x: np.sin(6.0*x)+0.4*np.cos(2.0*x)+0.2*x**2-0.1*x,
        fp=lambda x: 6.0*np.cos(6.0*x)-0.8*np.sin(2.0*x)+0.4*x-0.1,
    ),
    dict(
        key="x-sin-poly",
        label="x·sin+poly",
        latex=r"$f(x)=x\sin(4x)+0.2\,x^{3}$",
        f=lambda x: x*np.sin(4.0*x) + 0.2*x**3,
        fp=lambda x: np.sin(4.0*x) + 4.0*x*np.cos(4.0*x) + 0.6*x**2,
    ),
    dict(
        key="log1px2+sin",
        label="log(1+x²)+sin",
        latex=r"$f(x)=\ln(1+x^{2})+0.3\,\sin(5x)$",
        f=lambda x: np.log1p(x**2) + 0.3*np.sin(5.0*x),
        fp=lambda x: (2.0*x)/(1.0+x**2) + 1.5*np.cos(5.0*x),
    ),
    dict(
        key="tanh+poly",
        label="tanh+poly",
        latex=r"$f(x)=\tanh(2x)+0.2\,x^{3}-0.1\,x$",
        f=lambda x: np.tanh(2.0*x) + 0.2*x**3 - 0.1*x,
        fp=lambda x: 2.0/(np.cosh(2.0*x)**2) + 0.6*x**2 - 0.1,
    ),
    dict(
        key="exp-gauss-cos",
        label="exp(−x²)·cos",
        latex=r"$f(x)=e^{-x^{2}}\cos(3x)$",
        f=lambda x: np.exp(-x**2)*np.cos(3.0*x),
        fp=lambda x: np.exp(-x**2)*(-2.0*x*np.cos(3.0*x) - 3.0*np.sin(3.0*x)),
    ),
    dict(
        key="sqrt1px2+sin",
        label="sqrt(1+x²)+sin",
        latex=r"$f(x)=\sqrt{1+x^{2}}+0.2\,\sin(3x)$",
        f=lambda x: np.sqrt(1.0+x**2) + 0.2*np.sin(3.0*x),
        fp=lambda x: x/np.sqrt(1.0+x**2) + 0.6*np.cos(3.0*x),
    ),
    dict(
        key="sigmoid+poly",
        label="sigmoid+poly",
        latex=r"$f(x)=\sigma(3x)+0.1\,x^{2}$, $\sigma(z)=1/(1+e^{-z})$",
        f=lambda x: _sigmoid(3.0*x) + 0.1*x**2,
        fp=lambda x: 3.0*_sigmoid(3.0*x)*(1.0-_sigmoid(3.0*x)) + 0.2*x,
    ),
    dict(
        key="softplus-sin",
        label="softplus−cos",
        latex=r"$f(x)=\log(1+e^{2x})-0.3\,\cos(4x)$",
        f=lambda x: np.log1p(np.exp(2.0*x)) - 0.3*np.cos(4.0*x),
        fp=lambda x: 2.0*_sigmoid(2.0*x) + 1.2*np.sin(4.0*x),
    ),
    dict(
        key="atan+polycos",
        label="atan+poly·cos",
        latex=r"$f(x)=\tan^{-1}(3x)+0.2\,x^{2}\cos(2x)$",
        f=lambda x: np.arctan(3.0*x) + 0.2*x**2*np.cos(2.0*x),
        fp=lambda x: 3.0/(1.0+9.0*x**2) + 0.4*x*np.cos(2.0*x) - 0.4*x**2*np.sin(2.0*x),
    ),
]

def render_one(fun_def, outdir: Path):
    name   = fun_def["key"]
    label  = fun_def["label"]
    latex  = fun_def["latex"]
    f      = fun_def["f"]
    fprime = fun_def["fp"]

    # truth & estimate
    true_slope = float(fprime(x0))
    est_slope  = slope_estimator(f)
    err_abs    = abs(est_slope - true_slope)

    # viz grid around x0
    w = 0.30
    xx = np.linspace(x0 - w, x0 + w, 800)
    fx = f(xx)
    f0 = float(f(x0))

    # figure
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)

    # function curve
    ax.plot(xx, fx, "--", lw=_DEFAULT_LINEWIDTH, color=blue_color, label=r"$f(x)$")

    # tangent at x0 from adaptive estimate
    ax.plot(xx, f0 + est_slope*(xx - x0),
            lw=_DEFAULT_LINEWIDTH, color=blue_color, label=r"tangent @ $x_0$ (adaptive)")

    # mark x0 (red)
    ax.scatter([x0], [f0],
               s=_DEFAULT_MARKERSIZE*1.6,
               edgecolors=red_color, facecolor="none",
               linewidths=_DEFAULT_LINEWIDTH, zorder=3,
               label=r"$x_0$")

    # labels
    ax.set_title(f"adaptive derivative (defaults): {label}", fontsize=_DEFAULT_FONTSIZE+2)
    ax.set_xlabel(r"local coordinate $x$", fontsize=_DEFAULT_FONTSIZE)
    ax.set_ylabel(r"function $f(x)$", fontsize=_DEFAULT_FONTSIZE)
    ax.legend(frameon=True, fontsize=_DEFAULT_FONTSIZE-2, loc="best")

    # summary box
    true_line = rf"true: $f'(x_0)={true_slope:.4f}$ at $x_0={x0:.2f}$"
    est_line  = rf"est:  $a_\mathrm{{est}} = {format_value_with_uncertainty(est_slope, err_abs)}$"
    ax.text(0.02, 0.98, latex + "\n" + true_line + "\n" + est_line,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=_DEFAULT_FONTSIZE-1,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"))

    # save using your utils
    outfile = outdir / f"adaptive_fit_demo_{name}.png"
    save_fig(fig, outfile.parent, stem=outfile.stem, ext=outfile.suffix.lstrip("."))
    plt.close(fig)

    print(f"[saved] {outfile}")
    print(f"  true slope @ x0:      {true_slope:.6g}")
    print(f"  adaptive slope @ x0:  {est_slope:.6g} (|err|={err_abs:.6g})")

def main():
    apply_plot_style(base=blue_color)
    # Put outputs next to this file under ./plots (robust to CWD)
    outdir = resolve_outdir(None, file=__file__, default_rel="plots")

    for fun_def in FUNS:
        render_one(fun_def, outdir)

if __name__ == "__main__":
    main()
