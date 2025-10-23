# Benchmarks

This directory hosts accuracy checks, diagnostics, metrics, performance tests, and sensitivity sweeps.

## Folder structure
- `accuracy/`      — correctness checks vs references
- `diagnostics/`   — sanity plots, posterior shapes, conditioning
- `metrics/`       — quantitative measures (MSE, bias, etc.)
- `performance/`   — timing, memory, scaling
- `sensitivity/`   — parameter sweeps / stability studies
- `plots/`         — output figures (git-ignored)

## Naming convention
Inside each subfolder, use a short prefix + topic:
- `acc_*.py`   — accuracy
- `diag_*.py`  — diagnostics
- `met_*.py`   — metrics
- `perf_*.py`  — performance
- `sens_*.py`  — sensitivity

Examples:
- `sensitivity/sens_error_vs_stepsize.py`
- `sensitivity/sens_mse_vs_noise.py`
- `diagnostics/diag_mc_posterior.py`

## Saving plots
All scripts should save next to themselves under `plots/` using the shared utils:

```python
from common.utils import resolve_outdir, save_fig
outdir = resolve_outdir(None, file=__file__, default_rel="../plots")
outfile = save_fig(fig, outdir, stem="my_benchmark", ext="png")
