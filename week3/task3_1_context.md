# SF3 Machine Learning — Task 3.1 context

Read this before writing or running any code. It gives you everything needed to
build `week3/task3_1.py` from scratch in a single script, consistent with the
Week 1–2 work already in this repo.

## Where this sits in the project

Week 3 is **control**. Task 3.1 is the bridge: before we can control the cart,
the dynamics model must take the **action** into account. So far the model has
mapped state → one-step change in state. Now the system has **five** driving
variables: the four state variables plus the applied force.

This task does **not** build a controller. It only:
1. extends both models to include force as an input,
2. verifies (with the usual plots) that they predict the force-augmented
   one-step dynamics, and
3. determines the range of forces over which the model stays accurate.

## The system (conventions — do not deviate)

- State `X = [x, x_dot, theta, theta_dot]`. `theta` periodic on (−π, π];
  `theta = π` is pole-down (stable), `theta = 0` is pole-up (control target).
- Simulator: `cartpole.py`, class `CartPole` with `setState`, `getState`,
  `performAction(action)`, `loss`. One `performAction` call = one model step.
- **Model target throughout the project is the CHANGE in state**
  `ΔX = X(after one performAction) − X(before)`, not the next state itself.
- **From Task 2.3 onward the angle is represented as `sin θ, cos θ`** (no raw
  `θ` as a feature, no `remap_angle` calls inside the model pipeline).
- Best Week-2 model carried forward = **delta-NM**: sparse Gaussian-kernel
  regression, target ΔX, M basis centres (a subset of the data), sin/cos
  features, hyperparameters tuned by gradient descent (JAX).
- Never use `np.linalg.inv`; use `lstsq` / `solve`. (Handout rule.)

## The action → force transform (the crux of this task)

`performAction(a)` does **not** apply `a` directly. It applies

```
force = max_force * tanh(a / max_force)      # max_force = 20
```

So the raw action `a` saturates: the reachable applied force is the open
interval (−20, 20), and approaching the ends costs huge actions (F = 15 needs
a ≈ ±19.5; F = 19.3 needs a ≈ ±40).

### Decision: model on the APPLIED FORCE F, not the raw action a

The 5th input feature is **F (post-tanh)**. Reasons:
- The equations of motion depend on F, not a. We verified ΔX is **near-linear
  in F** (straight-line fits to ΔX vs F have ~0.6–2.3% residuals over
  [−15, 15]). Modelling on F means the model never has to relearn the tanh.
- Sampling F uniformly covers the controllable region evenly; sampling a
  uniformly would pile samples near ±20 through the tanh.
- For control later (3.2/3.3) a policy emits an action `a`; we just apply the
  same tanh, `F = 20*tanh(a/20)`, before feeding the model — smooth and
  differentiable, so this choice costs nothing for the policy optimisation.

To apply a chosen force F in the *true* simulator, invert the tanh:
`a = max_force * arctanh(F / max_force)` (clip |F| < 20 first).

## Feature vector and models

```
phi(X, F) = [x, x_dot, sin θ, cos θ, θ_dot, F]      # 6-D
target     = ΔX = [Δx, Δẋ, Δθ, Δθ̇]                 # 4-D (4 separate models)
```

**Linear** (baseline, sin/cos + force): least-squares `ΔX ≈ Φ C`, `C` is 6×4,
no bias term (matches handout `f(X)=CX` form). Use `np.linalg.lstsq`.

**Kernel (delta-NM)**, one independent model per output dimension j:
```
Gaussian kernel:  K(a,b) = exp( −½ Σ_d ((a_d − b_d)/σ_d)² )   # per-dim σ, d=1..6
choose M centres Z ⊂ training features
α_M = (K_MN K_NM + λ K_MM)^{-1} K_MN Y_N        # eq. 15, sparse delta-NM
prediction:  f(x*) = K(x*, Z) · α_M
```
Solve the (M×M) system with `jnp.linalg.solve` (+ tiny jitter), **not** inv.
Write it **modularly**: a `basis` switch selects `"NM"` (M centres, default) or
`"NN"` (M = N, every point a centre) so the form can be swapped later.

## Hyperparameter tuning (from scratch, multi-restart)

Per output dimension there are **7 hyperparameters**: 6 length scales σ (one per
feature) + 1 regularisation λ. Tune them **from scratch** (do NOT seed from
Week 2):
- objective = validation MSE of that output's model;
- optimise in log-space with `scipy.optimize.minimize`, `method="L-BFGS-B"`,
  Jacobian from `jax.grad` of a `jax.jit`-compiled loss;
- **multiple random initialisations** per output (start from the data-std
  heuristic σ_d = std of feature d, then add a few perturbed restarts; keep the
  best validation MSE);
- bounds keep σ and λ positive and sane.
- Tip: the differentiable solve recomputes an M×N kernel each step, so use a
  modest tuning subset (~1000–1200 pts, ~200 centres) for tuning, then refit the
  final α on the full data with the chosen M centres. Expect σ_F to come out
  large (a nearly-flat Gaussian along F) — that is the model agreeing that the
  force-dependence is essentially linear. Worth a sentence in the report.

## Choosing the force range, and the headline question

Two different things, answered at two different times:
- **Up-front design choice:** sample F uniformly over `[−F_train, F_train]` for
  data collection. Default `F_train = 15` (wide but comfortably reachable).
- **Measured from results:** *"what max/min forces stay accurate?"* Answer it
  with an **accuracy-vs-force sweep** — bin test data by |F| (from 0 out past
  the train edge toward the ceiling, e.g. up to 19.5) and plot test MSE vs |F|
  for both models. Expect: flat/low inside the trained range, rising on
  extrapolation beyond `F_train` and as |F| → 20 (tanh saturation makes those
  forces sparse/hard to reach). The usable range is where MSE stays acceptable.

## Required outputs (save to `figures/`, named `task3_1_<name>.png`)

- `task3_1_scatter.png` — predicted vs true ΔX, 4 panels, linear + kernel.
- `task3_1_scans_1d.png` — 1-D scans sweeping each of x, ẋ, θ, θ̇, **F**.
- `task3_1_force_scan.png` — focused F-only sweep, true/linear/kernel; should
  look linear.
- `task3_1_contours_2d.png` — 2-D contour scans, e.g. (F, θ) and (F, θ̇).
- `task3_1_rollouts.png` — iterated rollouts where the **same force sequence**
  (zero / constant / sinusoidal) is fed to the true sim and to each model; the
  true rollout applies `a = arctanh`-inverse of each F. With sin/cos features no
  angle remap is needed during iteration.
- `task3_1_force_accuracy.png` — the accuracy-vs-force sweep (headline answer).
- `task3_1_results.json` — config, test MSEs, tuned hyperparameters, sweep data.

## Sanity checks (expected)

- Linear test MSE is large (order 1+); kernel test MSE 1–2 orders lower.
- Force scan is straight; the linear model captures the **force** column well
  even though it is poor on the state nonlinearities.
- Accuracy degrades outside `[−F_train, F_train]` and toward |F| = 20.

The script is self-contained: `python week3/task3_1.py`. A `TASK3_1_QUICK=1`
environment variable runs a small fast version for smoke-testing.
