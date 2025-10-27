# Issues for derivkit/derivkit-examples

## #13: DEMO: Gaussian Process
- **State:** OPEN
- **Labels:** demo, plot, docs
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T02:29:36Z | **Updated:** 2025-10-23T02:29:36Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/13

**Goal:** Showcase derivative estimation using Gaussian Processes (GPs) — conceptually contrasting deterministic polynomial fits with probabilistic inference.

**Tasks:**
- Implement a minimal GP regression using your gaussian_process module.
- Visualize the mean function and ±1σ/2σ posterior envelopes.
- Estimate the slope and curvature at a chosen point (e.g. x0 = 0.3).
- Compare the GP-estimated derivative mean/std against the true slope.
- Save figure as plots/demo_gaussian_process.png and script as demos/demo_gaussian_process.py.

# Issues for derivkit/derivkit-examples

## #12: DEMO: Finite Difference
- **State:** OPEN
- **Labels:** demo, diagnostics, sensitivity, plot
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T02:28:33Z | **Updated:** 2025-10-23T02:28:33Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/12

**Goal:** Illustrate how numerical finite-difference derivatives are computed and how their accuracy depends on step size.

**Tasks:**
- Implement a minimal finite-difference estimator (forward, backward, central).
- Compare against DerivKit’s adaptive derivative on a smooth 1D function.
- Sweep step size $h$  and plot error vs $h$ to show optimal region.
- Add a visual showing truncation vs roundoff trade-off (“U-shape” error curve).
- Save figure as plots/demo_finite_difference.png and script as demos/demo_finite_difference.py.


# Issues for derivkit/derivkit-examples

## #11: BENCHMARK: Memory Scaling
- **State:** OPEN
- **Labels:** performance, benchmark, plot
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T02:26:17Z | **Updated:** 2025-10-23T02:26:17Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/11

**Goal:** Measure memory footprint with increasing polynomial order and grid density.

**Tasks:**
- Profile arrays and design matrix sizes.
- Report memory vs order and n_points.

# Issues for derivkit/derivkit-examples

## #10: BENCHMARK: Multiprocessing Scalability
- **State:** OPEN
- **Labels:** performance, benchmark, plot
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T02:25:06Z | **Updated:** 2025-10-23T02:25:06Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/10

**Goal:** Test strong/weak scaling across CPU cores.

**Tasks:**

- Use Python multiprocessing pool.
- Measure wall time, efficiency, and CPU usage.
- Plot speedup vs core count.

# Issues for derivkit/derivkit-examples

## #9: BENCHMARK: Timing
- **State:** OPEN
- **Labels:** performance, benchmark, plot
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T02:24:12Z | **Updated:** 2025-10-23T02:24:12Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/9

**Goal:** Benchmark DerivKit’s runtime scaling.

**Tasks:**

- Measure time vs n_points, order, and noise.
- Include single-thread and multi-process runs.
- Produce timing and speedup plots.

# Issues for derivkit/derivkit-examples

## #8: DIAGNOSTICS: Stability Region Visualization
- **State:** OPEN
- **Labels:** diagnostics, stability, plot
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T02:23:13Z | **Updated:** 2025-10-23T02:23:13Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/8

**Goal:** Show where adaptive fits remain numerically stable.

**Tasks:**
- Compute residuals or Jacobian determinant across parameter grid.
- Highlight regions of good/bad conditioning.

# Issues for derivkit/derivkit-examples

## #7: SENSITIVITY: Ridge Regularization Sweep
- **State:** OPEN
- **Labels:** sensitivity, plot
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T02:22:18Z | **Updated:** 2025-10-23T02:22:18Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/7

**Goal:** Study effect of ridge parameter on conditioning and bias.

**Tasks:**
- Sweep ridge strength, plot condition number, bias, and variance.
- Identify “sweet spot” region.

# Issues for derivkit/derivkit-examples

## #6: SENSITIVITY: Stepsize and Noise Sensitivity
- **State:** OPEN
- **Labels:** sensitivity, metrics, plot
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T02:21:16Z | **Updated:** 2025-10-23T02:21:16Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/6

**Goal:** Quantify estimator stability under different noise levels and step sizes.

**Tasks:**

- Sweep over σ_noise and spacing, compute RMSE/bias.
- Plot MSE vs noise level and error vs step size.
- Save as sensitivity/mse_vs_noise.py and sensitivity/error_vs_stepsize.py.

# Issues for derivkit/derivkit-examples

## #5: DIAGNOSTICS: Tangent Residual Analysis
- **State:** OPEN
- **Labels:** diagnostics, plot
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T02:20:06Z | **Updated:** 2025-10-23T02:20:06Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/5

Goal: Visualize residuals between model evaluations and local linear fits.

Tasks:

- Show residual vs position, highlight curvature regions.
- Add optional noise overlay for realism.
- Summarize residual RMS across spacing.

# Issues for derivkit/derivkit-examples

## #4: DEMO: Fisher Bias
- **State:** OPEN
- **Labels:** demo, plot, docs
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T02:18:52Z | **Updated:** 2025-10-23T02:18:52Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/4

**Goal:** Illustrate Fisher bias correction and model mismatch effects.

**Tasks:**

- Simulate biased data (shifted mean, calibration offset).
- Compute and plot recovered bias vector.
- Show contour shifts with/without correction.

# Issues for derivkit/derivkit-examples

## #3: DEMO: Calculus Kit (grad/jac/hess)
- **State:** OPEN
- **Labels:** diagnostics, plot, docs
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T02:17:10Z | **Updated:** 2025-10-23T20:55:51Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/3

**Goal:** Visualize residuals between model evaluations and local linear fits.

**Tasks:**

- Show residual vs position, highlight curvature regions.
- Add optional noise overlay for realism.
- Summarize residual RMS across spacing.


# Issues for derivkit/derivkit-examples

## #2: DEMO: DALI 
- **State:** OPEN
- **Labels:** demo, paper-ready, plot, docs
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T01:45:14Z | **Updated:** 2025-10-23T02:17:30Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/2

**Goal:** Show how DALI expansion (up to second/third order) approximates a full likelihood.

Tasks:
Build toy 1D and 2D likelihood examples.

- Compare Gaussian, Fisher, and DALI contours.
- Highlight non-Gaussian tails and curvature.

# Issues for derivkit/derivkit-examples

## #1: DEMO: Adaptive Fit 
- **State:** OPEN
- **Labels:** demo, paper-ready, plot, docs
- **Assignees:** nikosarcevic
- **Created:** 2025-10-23T01:44:22Z | **Updated:** 2025-10-23T02:19:12Z
- **URL:** https://github.com/derivkit/derivkit-examples/issues/1

**Goal:** Finalize the adaptive polynomial fit demo with MC inset and formatted slope/error display.

**Tasks:**

- Polish main figure layout (legend, inset, color consistency).
- Save as demos/adaptive_fit_demo.py and plots/adaptive_fit_demo.png.
- Add to documentation examples gallery.


