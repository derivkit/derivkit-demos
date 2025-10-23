# Demos

This directory hosts runnable examples illustrating key DerivKit features, from basic derivative estimation to full Fisher and DALI forecasting workflows.

## Folder structure
- `adaptive/`      — adaptive polynomial fits, internal grids, diagnostics  
- `calculus/`      — gradient, Jacobian, Hessian demonstrations and calculus utilities  
- `dali/`          — DALI expansions, higher-order curvature, and non-Gaussian effects  
- `finite/`        — finite-difference derivatives and step-size comparisons  
- `forecasts/`     — Fisher and DALI forecasts, bias propagation, model-level examples  
- `likelihoods/`   — likelihood surfaces, 1D/2D contours, and analytic comparisons  
- `plots/`         — output figures (git-ignored)

## Naming convention
Inside each subfolder, use a short prefix + topic:

| Prefix | Purpose |
| :------ | :-------- |
| `adap_*.py` | Adaptive derivative demos |
| `calc_*.py` | Calculus (grad/jac/hess) demos |
| `dali_*.py` | DALI or higher-order likelihood demos |
| `fin_*.py` | Finite-difference examples |
| `fish_*.py` | Fisher forecast demos |
| `bias_*.py` | Bias estimation or propagation demos |
| `like_*.py` | Likelihood visualization demos |

Examples:
- `adaptive/adap_linear_noiseless_order1.py`  # incldue the derivative order where appropriate
- `adaptive/adap_internal_nodes.py`  
- `calculus/calc_jacobian_vs_fd.py`  
- `finite/fd_stepsize_convergence.py`  
- `forecasts/fish_bias_vs_spacing.py`  
- `dali/dali_curvature_vs_fisher.py`  
- `likelihoods/like_gaussian_2d.py`

## Saving plots
All scripts should save next to themselves under `../plots/` using the shared utils:

```python
from common.utils import resolve_outdir, save_fig
outdir = resolve_outdir(None, file=__file__, default_rel="plots")
outfile = save_fig(fig, outdir, stem="my_demo", ext="png")
