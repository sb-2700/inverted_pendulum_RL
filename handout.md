# SF3 Machine Learning — Project Handout

**Easter Term 2026** | Last updated: April 24, 2026

---

# Logistics

## Key Dates

- Introductory session: 9:15am, Friday 15 May — James Dyson Building Seminar Room
- **Interim report due: Friday 29 May at 4pm** (via Moodle)
- **Final report due: Friday 12 June at 4pm** (via Moodle)

## Demonstration Sessions

Fridays 9:15–10:45am and 2:00–3:30pm in James Dyson Building Seminar Room; Tuesdays 11:00–12:30 in LR3A.

- You **must** attend a session the week after submitting your interim report to receive feedback
- Extra sessions Monday 08 June over Zoom (help with final report)
- Friday 05 June: Zoom only (rooms in use for exams)
- Tuesday 09 June: James Dyson Building Seminar Room (NOT LR3A — exams there)

**Plan ahead**: after the final demo session, demonstrators cannot respond quickly enough to help before the deadline.

## Reports

Technical reports follow a roughly chronological structure (not Introduction/Methods/Results/Discussion). Each method section immediately precedes its own results and discussion.

### Interim Report (20 marks)

- Max 3 A4 pages, max 750 words (including captions and headings)
- Covers first two weeks (nonlinear modelling complete, preparing to move to control)
- Rougher figures acceptable; discussion need not be deep
- **Content split:**
  - 15% — brief project overview
  - 70% — progress to date (results, issues encountered and resolved, effect on future work)
  - 15% — plan for remainder of project

### Final Report (60 marks)

- Max 11 A4 pages, max 3000 words (including captions/headings; NOT including references)
- Covers full project (condense/rewrite interim content in light of later results)
- **Structure:**
  - Introduction + high-level summary (~1 page)
  - Technical description with subsections: modelling, control, extensions (~7–8 pages total)
  - Discussion: recommendations, critical evaluation (~2 pages)

## General Advice

- **Version control**: use git from day one. Your responsibility not to lose work.
- **Lab notebook**: keep a dated record of work, notes, plots, scrap working. Demonstrators will ask to see it.

---

# Week 1: Simulation and Simple Linear Models

## Setup

```python
from cartpole import CartPole
# visual=True turns on animation (don't use this in other sections!)
example_system = CartPole(visual=True)
cart_position = 0.0
cart_velocity = 0.1
pole_angle = 0.01
pole_velocity = 0.0
state = [cart_position, cart_velocity, pole_angle, pole_velocity]
example_system.setState(state)
for _ in range(100):
    example_system.performAction()
```

## Dynamical System

**State vector**: X = [x, ẋ, θ, θ̇]
- x: cart position (centre = 0)
- θ: pole angle, periodic on [−π, π]
  - θ = π → pole hanging vertically down (stable equilibrium)
  - θ = 0 → pole upright (unstable equilibrium — control target)

**Equation of motion — pole (eq. 1):**
```
3ẍ cosθ + 2L θ̈ = 3g sinθ − 6μ_θ θ̇ / mL
```

**Equation of motion — cart (eq. 2):**
```
(m + M)ẍ + (1/2)mL θ̈ cosθ − (1/2)mL θ̇² sinθ = F − μ_x ẋ
```

- F = external force on cart
- μ_x, μ_θ = friction coefficients for cart and pole
- m = pole mass, M = cart mass, L = pole half-length

`performAction(force)` updates state using Euler algorithm (multiple sub-steps per call). force=0 gives free dynamics.

---

### Task 1.1 — Rollouts

Simulate rollouts from various initial conditions with **no applied force**. Start from stable equilibrium with nonzero initial cart or angular velocity.

- Plot time evolution of all 4 state variables
- Plot phase portraits (pairs of variables against each other)
- Vary initial velocities to get: simple oscillation around stable equilibrium, and full pole rotation

**Hints:**
- Useful ranges: cart velocity [−10, 10], pole angle [−π, π], pole angular velocity [−15, 15]
- Use `remap_angle` to map θ back to [−π, π]
- Note: the simulator uses θ as a continuous variable internally (not remapped)

---

## Changes of State

Model the **change** ΔY = X(T) − X(0) after one `performAction()` call (with no force). Better target than Y itself because the change is small and nearly linear.

### Task 1.2 — Scanning the Functional Relationship

Fix all state variables at random values. Scan one variable at a time, plot Y then ΔY as a function of the scanned variable.

> **Important**: reset ALL state variables after each call to `performAction()` (unlike rollouts).

Then explore ΔY:
1. 1D scans of each variable
2. 2D contour plots: fix two variables, scan two — use `matplotlib.tricontourf`

**One of the variables has no effect on the next step — which one?**

---

## Linear Model

f(X) = C · X, where C is a 4×4 matrix (eq. 4).

### Task 1.3 — Fit Linear Model

- Gather 500 (X, ΔY) pairs: random initial states, one step each (pick suitable ranges)
- Fit C via linear regression
- Test with scatter plots:
  - Option A: input variable on x-axis, predicted and real ΔY on y-axis
  - Option B: real ΔY on x-axis, predicted ΔY on y-axis (perfect fit = straight line through origin)
- Repeat 1D scans from 1.2: overlay real ΔY and predicted ΔY
- **Which variables does the linear model predict well? Why?**

### Task 1.4 — Iterated Rollouts with Linear Model

Iterate the model (eq. 5):
```
X_{n+1} ← X_n + f(X_n)
```

Compare model rollouts vs true dynamics for various initial conditions (including full pole rotations).

- **Why does θ diverge without remapping?** Remap angle during iterations.
- **Do the true dynamics remap θ explicitly? Why not?**

---

# Week 2: Nonlinear Modelling

## Kernel / Basis Function Model

Nonlinear model for a **single** state variable (eq. 6):
```
f(X) = Σ_i α_i K(X, X_i)
```
- Sum over M basis function centres {X_i} — a subset of data points
- α_i are scalar coefficients
- Y is scalar (one state variable output); X is still the full state vector

Build **4 separate models** — one per output state variable (eq. 6 applied independently to each).

**Gaussian kernel (eq. 7):**
```
K(X, X') = exp( −Σ_j (X^(j) − X'^(j))² / (2 σ_j²) )
```
- X^(j) = j-th component of X (j=0: x, j=1: ẋ, j=2: θ, j=3: θ̇)
- σ_j = length scale hyperparameters

**Periodic correction for θ**: in the θ term only, replace (θ − θ')² with sin²((θ − θ')/2). This enforces 2π periodicity and keeps K in [0, 1].

## Finding the Kernel Model Parameters

**Full basis (N centres = N data points):**
```
[K_NN]_{i,i'} = K(X_i, X_i')          (eq. 9)
K_NN α_N = Y_N                          (eq. 10)
(K_NN + λI) α_N = Y_N                  (eq. 11, regularised)
α_N = [K_NN + λI]^{-1} Y_N             (eq. 12)
```

**Sparse basis (M centres, M << N):**
```
K_NM α_M = Y_N                          (eq. 13)
```
K_NM is N×M. Least squares solution (eq. 14):
```
α_M = [K_MN K_NM]^{-1} K_MN Y_N
```

Regularised sparse solution — applied **separately per state variable j** (eq. 15):
```
α_M^(j) = (K_MN K_NM + λ K_MM)^{-1} K_MN Y_N^(j)
```
where K_MM is M×M with elements K(X_i, X_i') for basis centre pairs.

**Hyperparameter choices:**
- λ: try 10⁻⁶ to 10⁻¹ on log scale; evaluate on validation data
- σ_j: good first guess = std of each variable in your dataset

> ⚠️ Use `np.linalg.lstsq` — **NEVER** `np.linalg.inv`.
> Note: the handout contains a typo writing `np.linalg.ltsq` — the correct function name is `np.linalg.lstsq`.

### Task 2.1 — Fit and Evaluate Nonlinear Model

- Use data from Week 1
- Target: ΔY (change in state) OR residual error of linear model
- Verify fit with scatter plots
- Study convergence: vary M (start M=10, double each time, select centres randomly) and vary N
- Plot 2D slices of target function and fit
- Use rollouts to compare model vs real dynamics
- **Quantify**: how long does model stay accurate, in time units and oscillation cycles?
- Compare rollout accuracy to point-wise accuracy on random test data

### Task 2.2 — JAX Hyperparameter Optimisation

Write a function: hyperparams → fit model on training data → return MSE on validation data.

Use `jax.grad` to differentiate it, then `scipy.optimize.minimize` (method='L-BFGS-B') with sensible initial values and bounds.

Report:
- Validation MSE before and after optimisation
- Optimal length scale values — why those values?
- Plots showing optimised model fits better
- Rollout comparison vs Task 2.1

> MSE is a suitable metric. Negative log-likelihood is better if modelling Gaussian noise variance.

### Task 2.3 — sin/cos Angle Features

Replace θ as a raw input with **[sin(θ), cos(θ)]**. Remove the sin²((θ−θ')/2) periodic kernel correction — no longer needed.

Feature vector becomes 5D: [x, ẋ, sin(θ), cos(θ), θ̇]

- Refit both linear and nonlinear models
- Remove all calls to angle remapping
- Comment on results with illustrative plots

---

# Week 3: Controlling the System

## Adding Force to the Model

`performAction(action)` passes action through tanh before applying as force (scaled by `max_force`).

Extend input to include action. Input is now **5D**: [x, ẋ, θ, θ̇, F] (or **6D** with sin/cos features: [x, ẋ, sin(θ), cos(θ), θ̇, F]).

### Task 3.1 — Refit Models with Action Input

- Collect new data: random initial conditions AND random actions, one step each
- Choose reasonable min/max force values
- Refit both linear and nonlinear models with extended input
- Verify with scatter plots, 1D/2D scans, rollouts
- **What are the max/min forces where the model still makes accurate predictions?**

## Policies and Loss Functions

**Policy**: p(X) maps state to action. Goal: keep pole upright (θ = 0).

**Simple loss (eq. 16):**
```
l(X) = −cos(θ)
```

**Better loss targeting full desired state X₀ = [0,0,0,0] (eq. 17):**
```
l(X) = 1 − exp(−|X − X₀|² / (2 σ_l²))
```
- σ_l = scaling factor (can be separate per state component)
- Saturates to 1 for large departures (loss is independent of exact state when far from target)

**Trajectory loss over T steps (eq. 18):**
```
L = Σ_{t=1}^{T} l(X_t)
```

`CartPole` class has a built-in pointwise loss function.

## Linear Policy

p(X) = p · X, where p is a coefficient vector (eq. 19).

### Task 3.2 — Optimise Linear Policy (True Dynamics)

- Evaluate trajectory loss for rollouts (short horizon, long enough for a couple of oscillation periods)
- 1D and 2D scans of loss vs p parameters (before optimisation)
- Optimise p using JAX gradients + scipy L-BFGS-B on **true dynamics**
  - Replace numpy calls in CartPole with jax.numpy equivalents
  - Use `jax.jit` and `jax.scan` (replaces for loops) for speed
- Start from slightly displaced upright position → find stabilising p
- Try from downward stable position → **what happens?**
- Tune σ_l (and possibly max_force)
- Plot time evolution showing pole kept upright

### Task 3.3 — Model Predictive Control

Same as 3.2 but optimise p using **model rollouts** (from Week 2) instead of true dynamics. Limit time horizon to where the model is still accurate.

---

# Week 4: Open-Ended Extensions

Reports are **not** penalised for failed extensions — document what you tried and why it didn't work.

### Suggestion 1: Sensitivity and Stability

1. Add noise to **observed** dynamics (not real dynamics) → refit models, characterise accuracy degradation
2. Optimise and test linear control policy — compare to noise-free case
3. Add noise to **actual** dynamics → repeat and compare

### Suggestion 2: Nonlinear Controllers

Use a nonlinear policy p(X):
- Basis function approach (like the nonlinear dynamics model)
- Or other nonlinear transformations of X

---

# Quick Reference

| Symbol | Meaning |
|--------|---------|
| X | State vector [x, ẋ, θ, θ̇] |
| Y | Next state after one performAction() |
| ΔY | Change in state = Y − X (model target) |
| C | Linear model coefficient matrix (4×4) |
| K | Kernel function |
| α | Kernel model coefficients |
| σ_j | Length scale hyperparameters (one per input feature) |
| λ | Regularisation parameter |
| p | Linear policy coefficient vector |
| σ_l | Loss function scaling factor |
| M | Number of basis function centres |
| N | Number of data points |
| K_NM | N×M kernel matrix (all N data pts vs M centres) |
| K_MN | M×N kernel matrix (M centres vs all N data pts) |
| K_MM | M×M kernel matrix (centres vs centres) |

| θ | Physical meaning |
|---|-----------------|
| θ = π | Pole hanging down (stable equilibrium) |
| θ = 0 | Pole upright (unstable equilibrium — control target) |

| Week | Tasks | Deadline |
|------|-------|----------|
| 1 | 1.1 Rollouts, 1.2 Scan ΔY, 1.3 Linear model, 1.4 Iterated rollouts | — |
| 2 | 2.1 Kernel model, 2.2 JAX hyperparams, 2.3 sin/cos features | Interim report 29 May |
| 3 | 3.1 Add action, 3.2 Linear policy, 3.3 Model predictive control | — |
| 4 | Extensions + writeup | Final report 12 June |
