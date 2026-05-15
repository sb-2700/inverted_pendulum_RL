# SF3 Machine Learning — Inverted Pendulum (Cartpole)

Cambridge Engineering SF3 project, Easter Term 2026.
Full task descriptions in `handout.md`.

## Key Dates
- **Interim report**: Friday 29 May at 4pm (Moodle) — covers Weeks 1 & 2
- **Final report**: Friday 12 June at 4pm (Moodle)

## Project Goal
Build data-driven models of cartpole dynamics, then optimise a controller using direct policy search / reinforcement learning.

## Repo Structure
```
cartpole.py          # Provided simulator — do not modify
handout.md           # Full project handout with all task descriptions
CLAUDE.md            # This file
week1/               # Tasks 1.1–1.4: simulation and linear models
week2/               # Tasks 2.1–2.3: nonlinear models + JAX
week3/               # Tasks 3.1–3.3: control / policy optimisation
week4/               # Extensions
figures/             # All saved plots (descriptive names, e.g. task1_1_rollout.png)
notes.md             # Dated lab notebook
```

## The Cartpole System

**State vector X = [x, ẋ, θ, θ̇]**
- x: cart position (0 = centre)
- θ: pole angle, periodic on [−π, π]
- **θ = π → pole hanging DOWN (stable equilibrium)**
- **θ = 0 → pole UPRIGHT (unstable equilibrium — the control target)**

`performAction(force)` steps the simulation (Euler, multiple sub-steps). force=0 = free dynamics.
`remap_angle()` maps θ back to [−π, π].
Action is passed through tanh before being applied as force (scaled by `max_force`).

## Modelling Approach — Progression by Week

| Week | Input | Target | Method |
|------|-------|--------|--------|
| 1 | X = [x, ẋ, θ, θ̇] | ΔY = Y − X | Linear: ΔY = C·X |
| 2 | X = [x, ẋ, θ, θ̇] | ΔY | Kernel: ΔY = Σ αᵢ K(X, Xᵢ) |
| 2.3+ | X = [x, ẋ, sin(θ), cos(θ), θ̇] | ΔY | Same kernel, new features |
| 3+ | X + action F | ΔY | Same models, 5D or 6D input |

**ΔY = X(T) − X(0)** = change in state after one `performAction()` call. This is the model target throughout (not Y itself).

## Key Equations

### Equations of Motion
```
Pole  (eq. 1):  3ẍ cosθ + 2L θ̈ = 3g sinθ − 6μ_θ θ̇ / mL
Cart  (eq. 2):  (m + M)ẍ + (1/2)mL θ̈ cosθ − (1/2)mL θ̇² sinθ = F − μ_x ẋ
```

### Linear Model (Week 1)
```
f(X) = C · X         C is 4×4
X_{n+1} = X_n + f(X_n)    (iterated rollout)
```

### Kernel Model (Week 2)
```
f(X) = Σ_i α_i K(X, X_i)       (4 separate models, one per output variable)

Gaussian kernel (eq. 7):
K(X, X') = exp( −Σ_j (X^(j) − X'^(j))² / (2 σ_j²) )

Periodic angle correction (used until Task 2.3):
Replace (θ − θ')² with sin²((θ − θ')/2) in the θ term only
```

### Sparse Kernel Regression (eq. 15, per output variable j)
```
α_M^(j) = (K_MN K_NM + λ K_MM)^{-1} K_MN Y_N^(j)
```
- N = number of data points; M = number of basis centres (M << N)
- K_MN is M×N; K_NM is N×M; K_MM is M×M
- Applied independently for each of the 4 output state variables

### Loss Function (Week 3)
```
l(X) = 1 − exp(−|X − X₀|² / (2 σ_l²))    (pointwise)
L = Σ_{t=1}^{T} l(X_t)                     (trajectory)
X₀ = [0, 0, 0, 0]  (upright, stationary, centred)
```

### Linear Policy (Week 3)
```
p(X) = p · X    (p is a vector, dot product gives scalar action)
```

## Critical Implementation Rules

1. **NEVER use `np.linalg.inv`** — use `np.linalg.lstsq` instead
   - Note: handout has a typo writing `np.linalg.ltsq` — correct name is `np.linalg.lstsq`
2. **Always remap θ** during iterated model rollouts (not needed in true dynamics)
3. **Hyperparameter starting points**: σ_j = std of each variable in dataset; λ ∈ [10⁻⁶, 10⁻¹]
4. **Basis centres**: select M randomly from data; start M=10, double each time
5. **Task 2.3 onwards**: use sin(θ) and cos(θ) as features, remove periodic kernel correction
6. **JAX**: use `jax.grad` for autodiff, `jax.jit` for speed, `jax.scan` to replace for loops
7. **Optimiser**: `scipy.optimize.minimize` with method='L-BFGS-B', provide bounds and initial values

## Workflow
- Save all figures to `figures/` with descriptive names
- Update `notes.md` with dates and what you worked on
- Commit after completing each task
- Update task checklist below as you go

## Task Checklist

| Task | Description | Status |
|------|-------------|--------|
| 1.1 | Rollouts — simulate and plot free dynamics | ⬜ |
| 1.2 | Scan functional relationship X → ΔY | ⬜ |
| 1.3 | Fit linear model, test predictions | ⬜ |
| 1.4 | Iterated rollouts with linear model | ⬜ |
| 2.1 | Fit nonlinear kernel model, evaluate convergence and rollouts | ⬜ |
| 2.2 | JAX hyperparameter optimisation | ⬜ |
| 2.3 | Replace θ with sin/cos features, refit models | ⬜ |
| 3.1 | Add action to model input, refit and verify | ⬜ |
| 3.2 | Optimise linear policy on true dynamics | ⬜ |
| 3.3 | Model predictive control (policy on model rollouts) | ⬜ |
| 4.x | Extensions | ⬜ |
