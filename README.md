<p align="center">
  <img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-red.png" width="70" alt="DerivKit red"/>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-blue.png" width="70" alt="DerivKit blue"/>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-yellow.png" width="70" alt="DerivKit yellow"/>
</p>

---

# DerivKit Demos

[![Open CCL demo in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/derivkit/derivkit-demos/main?labpath=demo-notebooks/10-derivkit-ccl-demo.ipynb)

**Runnable demo collection** for the [DerivKit](https://github.com/derivkit/derivkit) ecosystem.  
This repository is **non-installable by design** — just clone it, create an environment, and run the demos in place.  
It contains ready-to-run Python scripts and optional notebooks showcasing core DerivKit modules such as **DerivativeKit**, **CalculusKit**, and **ForecastKit**.

---

## Repository structure

```text
derivkit-demos/
├─ scripts/  # Runnable demo scripts
│  ├─ calculus-kit-advanced.py
│  ├─ calculus-kit-simple.py
│  ├─ derivative-kit-advanced.py
│  ├─ derivative-kit-simple.py
│  ├─ derivative-kit-tabulated-advanced.py
│  ├─ derivative-kit-tabulated-simple.py
│  ├─ derivkit-cluster-counts-forecast.py
│  ├─ derivkit-cluster-counts-sweep.py
│  ├─ forecast-kit-dali-1d.py
│  ├─ forecast-kit-dali-2d.py
│  ├─ forecast-kit-fisher-bias.py
│  └─ forecast-kit-fisher.py
│
├─ notebooks/  # Optional notebooks
├─ utils/
│  └─ style.py  # Shared Matplotlib style and palette
│
├─ plots/  # Auto-generated figures (git-ignored)
├─ run_demo.py  # Launcher utility for demos
├─ requirements.txt  # Minimal dependencies
├─ environment.yaml  # Conda environment file
├─ pyproject.toml  # Metadata only (repo is not installable)
├─ LICENSE
└─ README.md
```

---


##  Demos included

**DerivativeKit**
- Derivative methods backends with analytic comparisons (finite, poly, adaptive).
- Derivatives of tabulated functions with interpolation schemes.


**CalculusKit**
- Gradients, Hessians, and Jacobians with analytic comparisons.

**ForecastKit**
- **Fisher Information** — analytic vs. numerical ellipses.
- **Fisher Bias** — systematic offset → parameter shift (Δθ).
- **DALI** — nonlinear likelihood expansions beyond Fisher.
- **Cluster counts** — forecasting and parameter sweeps for cluster-count examples

Figures are automatically saved to `plots/` as both `.pdf` and `.png` unless `--plot` is
used to display them interactively.

---

##  Quick start

> You **do not need to install** this repo — just clone and run inside a Python or Conda environment.

### 1 Create an environment

**With pip:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**With conda:**
```bash
conda env create -f environment.yaml
conda activate derivkit-demos
```

---

### 2 Run demos

List all available demos:
```bash
python run_demo.py --list
```

Run a specific demo:
```bash
python run_demo.py forecast-kit-fisher
python run_demo.py forecast-kit-fisher-bias
python run_demo.py forecast-kit-dali-1d
python run_demo.py forecast-kit-dali-2d
```

Run by fuzzy name:
```bash
python run_demo.py fisher
python run_demo.py fisher-bias
python run_demo.py dali
python run_demo.py cluster
```

Pass extra arguments through to the demo with --:
```bash
python run_demo.py dali -- --plot
python run_demo.py forecast-kit-dali-2d -- --method adaptive
python run_demo.py forecast-kit-fisher-bias -- --plot
```

Run directly:
```bash
python -m scripts.forecast-kit-fisher.py --plot
```

> The `--` separator passes additional arguments directly to the demo.

---

##  Styling

All demos use a shared visual style from:
```python
from utils.style import apply_plot_style, DEFAULT_COLORS
apply_plot_style()
```

Available colors:  
`DEFAULT_COLORS = {"blue": "#3b9ab2", "yellow": "#e1af00", "red": "#f21901"}`

---

##  Tips

- **ImportError: `utils.style`**  
  Always run scripts from the repo root or via `run_demo.py`.

- **No install step**  
  This repo is meant to be run in place; avoid `pip install .`.

- **Clean working tree**  
  Figures and notebook outputs are ignored via `.gitignore`.

---

##  Development tips

If you use `ruff`:
```bash
ruff check --fix .
```

Add or modify demos under `demo-scripts/`, following the numbered naming scheme (`NN-topic-name.py`) and including a top-level docstring explaining purpose, usage, and key results.


---

## Citation

If you use **derivkit** in your research, please cite:

```bibtex
@software{sarcevic2025derivkit,
  author       = {Nikolina Šarčević and Matthijs van der Wild and Cynthia Trendafilova and Bastien Carreres},
  title        = {derivkit: A Python Toolkit for Numerical Derivatives},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/derivkit/derivkit}},
}

```

---

##  License

MIT — see [LICENSE](LICENSE).
---

##  Acknowledgements

Developed as part of the **DerivKit ecosystem** (CalculusKit, ForecastKit, and friends).  
Thanks to all contributors and collaborators supporting derivative-based cosmological inference tools.

## Useful links
- [DerivKit documentation](https://docs.derivkit.org)
- [DerivKit website](https://derivkit.org)
- [DerivKit GitHub repository](https://github.com/derivkit/derivkit)
- [DerivKit organization](https://github.com/derivkit)
