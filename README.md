

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
│  └─ 07-forecast-kit-dali.py
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

##  Demos included

**DerivativeKit / CalculusKit**
- Adaptive vs. finite-difference backends with analytic comparisons.


**DerivativeKit / CalculusKit**
- Gradients, Hessians, and Jacobians with analytic comparisons.

**ForecastKit**
- **Fisher Information (2D)** — analytic vs. numerical ellipses.
- **Fisher Bias** — systematic offset → parameter shift (Δθ).
- **DALI (1D)** — Fisher vs. Doublet-DALI vs. exact log-likelihood.

Figures are automatically saved to `plots/` as both `.pdf` and `.png` unless `--plot` is used to display them interactively.

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

##  Acknowledgements

Developed as part of the **DerivKit ecosystem** (CalculusKit, ForecastKit, and friends).  
Thanks to all contributors and collaborators supporting derivative-based cosmological inference tools.
