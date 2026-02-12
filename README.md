

<p align="center">
  <img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-red.png" width="70" alt="DerivKit red"/>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-blue.png" width="70" alt="DerivKit blue"/>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-yellow.png" width="70" alt="DerivKit yellow"/>
<img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-red.png" width="70" alt="DerivKit red"/>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-blue.png" width="70" alt="DerivKit blue"/>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-yellow.png" width="70" alt="DerivKit yellow"/>
<img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-red.png" width="70" alt="DerivKit red"/>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-blue.png" width="70" alt="DerivKit blue"/>
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-yellow.png" width="70" alt="DerivKit yellow"/>
<img src="https://raw.githubusercontent.com/derivkit/derivkit-logo/main/png/logo-red.png" width="70" alt="DerivKit red"/>
  &nbsp;&nbsp;
</p>


---

# DerivKit Demos

[![Open CCL demo in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/derivkit/derivkit-demos/main?labpath=demo-notebooks/10-derivkit-ccl-demo.ipynb)


**Runnable demo collection** for the [DerivKit](https://github.com/derivkit/derivkit) ecosystem.  
This repository is **non-installable by design** — just clone, create an environment, and run.  
It contains ready-to-execute Python scripts and optional Jupyter notebooks showcasing
key DerivKit modules such as **DerivativeKit**, **CalculusKit**, and **ForecastKit**.

---

##  Repository structure

```
derivkit-demos/
├─ demo-scripts/                 # Runnable demo scripts (no install required)
│  ├─ 01-derivative-kit-simple.py
│  ├─ 02-derivative-kit-advanced.py
│  ├─ 03-calculus-kit-simple.py
│  ├─ 04-calculus-kit-advanced.py
│  ├─ 05-forecast-kit-fisher.py
│  ├─ 06-forecast-kit-fisher-bias.py
│  ├─ 07-forecast-kit-dali.py
│  ├─ 08-derivative-kit-tabulated-simple.py
│  └─ 09-derivative-kit-tabulated-advanced.py

│
├─ demo-notebooks/               # Optional notebooks (not required for running)
├─ utils/
│  └─ style.py                   # Shared Matplotlib style and color palette
│
├─ plots/                        # Auto-generated figures (git-ignored)
├─ templates/                    # Plot or doc templates for demos
├─ forecast_demo.py              # Combined example launcher for ForecastKit demos
├─ run_demo.py                   # Launcher utility to run any demo by ID or name
│
├─ requirements.txt              # Minimal dependencies (DerivKit, NumPy, SciPy, Matplotlib)
├─ environment.yaml              # Conda environment file (optional)
├─ pyproject.toml                # Metadata only (not installable)
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
- **DALI** — Fisher vs. Doublet-DALI vs. exact log-likelihood.

Figures are automatically saved to `plots/` as both `.pdf` and `.png` unless `--plot` is used to display them interactively.

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
python run_demo.py 07 -- --plot
```

Run by fuzzy name:
```bash
python run_demo.py dali -- --plot
python run_demo.py fisher-bias -- --method adaptive
```

Run directly:
```bash
python demo-scripts/05-forecast-kit-fisher.py --plot
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

##  License

MIT — see [LICENSE](LICENSE).

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

##  Acknowledgements

Developed as part of the **DerivKit ecosystem** (CalculusKit, ForecastKit, and friends).  
Thanks to all contributors and collaborators supporting derivative-based cosmological inference tools.

## Useful links
- [DerivKit documentation](https://docs.derivkit.org)
- [DerivKit website](https://derivkit.org)
- [DerivKit GitHub repository](https://github.com/derivkit/derivkit)
- [DerivKit organization](https://github.com/derivkit)
