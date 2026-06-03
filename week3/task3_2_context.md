# SF3 Machine Learning — Task 3.2 context

Read this in full before writing or running any code. It defines the
conventions, the locked design decisions (with reasoning), and exactly what to
build for Task 3.2 of the SF3 cartpole project. Build a single self-contained
script `week3/task3_2.py`, consistent with the Week 1–3.1 work already in this
repo. Save all figures to `figures/` named `task3_2_<name>.png`.

## Where this sits in the project

Week 3 is **control**. Task 3.1 was still *modelling* (it added the applied
force as a 6th model input feature and verified the model predicts one-step
ΔX). Task 3.2 is the first *control* task: optimise a **linear feedback
policy** to balance the pole at the upright unstable equilibrium.

**CRITICAL — 3.2 does NOT use the learned model.** The handout says optimise
"under the evolution given by the **true model dynamics**." The plant in 3.2 is
the **real simulator**, re-implemented in JAX so it is differentiable. The
learned dynamics model from Weeks 2–3.1 does not appear until Task 3.3
(model-predictive control), which reuses this exact policy/loss machinery but
swaps the true dynamics for the model and compares. Keeping 3.2 on the true
dynamics isolates "can a linear policy do this at all?" from "is my model good
enough to plan with?".

## The system (conventions — do not deviate)

- State `X = [x, x_dot, theta, theta_dot]`. `theta = pi` is pole-DOWN (stable),
  `theta = 0` is pole-UP (the control target). `theta` periodic on (−pi, pi].
- Simulator: `cartpole.py`, class `CartPole` (`setState`, `getState`,
  `performAction(action)`, `loss`). **Do not modify `cartpole.py`.**
- One `performAction` call = one control step = **0.1 s** of sim time
  (`delta_time = 0.1`, internally `sim_steps = 50` semi-implicit Euler substeps).
- `performAction(a)` applies `force = max_force * tanh(a / max_force)`,
  `max_force = 20`. The argument `a` is the **action** (pre-tanh); the tanh
  bounds the force to the open interval (−20, 20).
- Physics constants (from `cartpole.py`): `gravity = 9.8`, `pole_length = 0.5`,
  `pole_mass = 0.5`, `cart_mass = 0.5`, `mu_c = 0.001`, `mu_p = 0.001`.
- Never use `np.linalg.inv`; use `lstsq` / `solve`. (Handout rule, carried over.)

### Measured timescales (already verified against the simulator — use these)
- Down-equilibrium small oscillation period ≈ **0.92 s ≈ 9 steps**.
- From near-upright (θ=0.1), the *uncontrolled* pole reaches |θ|>1 rad in
  **0.5 s (5 steps)** — the instability is fast, the controller must act quickly.
- "A couple of oscillation periods" ≈ 2 s ≈ 20 steps. Use **T = 40 steps (4 s)**
  as the default horizon: long enough that a stabilised vs a falling trajectory
  are clearly distinguished in the loss, short enough to differentiate cheaply.

## LOCKED DESIGN DECISIONS (with reasoning — keep these pinned)

1. **Plant = true dynamics, re-implemented in JAX.** Not the learned model.
   The single highest-risk component (see "The crux" below).

2. **Policy = raw 4-D linear state feedback** `p(X) = p · X`, `p` a length-4
   vector, output is a **scalar action** `a` (eq. 19). This action is fed to the
   JAX `performAction`, which applies the tanh internally — so the force is
   auto-bounded and large gains simply saturate it (the natural actuator limit,
   not a bug). NOTE this is the *action* (pre-tanh); contrast 3.1 where we
   *modelled* on the post-tanh force F. Both are correct, just different roles.
   - Raw θ (not sin/cos) in the policy is faithful to eq. 19, is the textbook
     LQR-style feedback for an unstable equilibrium, and is numerically
     identical to sin/cos near upright (θ stays small, sinθ≈θ, cosθ≈const).
   - sin/cos would only matter for global well-definedness over the full circle
     (a spinning pole), which a *linear* policy cannot achieve anyway. Held as
     an OPTIONAL labelled variant — see "Optional extension" at the end. Do the
     core raw-4-D version first and completely.

3. **Loss = handout eq. 17 with RAW θ, own JAX implementation, per-component
   σ_l.** `l(X) = 1 − exp(−Σ_j (X_j − X0_j)² / (2 σ_l_j²))`, `X0 = [0,0,0,0]`,
   trajectory loss `L = Σ_{t=1}^{T} l(X_t)`.
   - Raw θ is the faithful choice: eq. 17 uses raw `X`, and `cartpole.py`'s
     built-in `_loss` is literally `1 − exp(−dot(state,state)/(2·0.5²))` — raw
     state, single σ=0.5. Our JAX loss MUST reduce to the built-in when the
     per-component σ_l vector is the scalar 0.5 — assert this as a check.
   - Per-component σ_l (one per state variable) because the handout explicitly
     allows it and it is physically motivated (care strongly about θ, weakly
     about cart position x).
   - The exp form SATURATES to 1 for large departures → gradient ≈ 0 far from
     target. This flatness is the mechanistic reason the downward-start
     experiment stalls (see experiments). It is a feature, not a bug.
   - Periodicity caveat (raw θ fails to recognise "upright after a full
     rotation"): real but irrelevant here because the stabilisation task never
     wraps. One-line report comment only; do NOT build (1−cosθ) now.

4. **σ_l selected by a σ_l-INDEPENDENT quality-metric sweep** (see dedicated
   section). Do NOT optimise σ_l jointly with `p` — changing σ_l changes the
   objective, so comparing optimised *loss* across σ_l values is circular (wider
   σ_l always reports lower loss without better control). Score controllers on
   metrics that do not depend on σ_l.

5. **T = 40 steps (4 s)** default horizon. **max_force kept at 20.** Near
   upright the forces are tiny so 20 never saturates; run ONE check raising
   max_force in the downward case to demonstrate the failure is structural (a
   linear policy can't swing up), not actuator-limited.

6. **Fixed optimisation IC**, per handout "given a fixed initial state":
   `X0_up = [0, 0, 0.2, 0]` (pole 0.2 rad off upright). The downward experiment
   uses `X0_down = [0, 0, pi, 0]`.

## The crux: a differentiable JAX copy of the dynamics

Optimising `p` needs ∂L/∂p, which flows through the entire rollout
(p → action → state → … → loss). So re-implement `performAction` in
`jax.numpy`, matching `cartpole.py` EXACTLY:
  - `force = max_force * jnp.tanh(action / max_force)`
  - loop `sim_steps = 50` substeps with `dt = delta_time / sim_steps`
  - same `s=sin(θ)`, `c=cos(θ)`, `m = 4(m_c+m_p) − 3 m_p c²`,
    `cart_accel`, `pole_accel` expressions as in `cartpole.py`
  - **semi-implicit Euler in the SAME update order** as `cartpole.py`:
    update `cart_velocity`, then `pole_velocity`, then `pole_angle`, then
    `cart_location` (this order is symplectic; do not reorder).
  - Use `jax.lax.fori_loop` (or unrolled scan) over the 50 substeps so it is
    jittable and differentiable. Do NOT remap the angle inside the dynamics
    (the true simulator doesn't; sin/cos in the EOM handle periodicity).

**GATING CHECK (must run first, must pass before anything else):** for a batch
of ~200 random (state, action) pairs, compare one-step JAX `performAction`
against the NumPy `CartPole.performAction`. Max abs difference must be
< 1e-6 per state component. Print PASS/FAIL. If FAIL, stop — every downstream
result is invalid.

## Rollout, loss, optimisation machinery (the Task-2.2 pattern, applied to a rollout)

- **Rollout**: `jax.lax.scan` over T steps. Carry = state (length 4). At each
  step: `a = p · state`; `state = performAction_jax(state, a)`; emit the
  per-step pointwise loss `l(state)` (and optionally the action, for plotting).
- **Trajectory loss** `L(p; X0, σ_l) = Σ_t l(X_t)`, `jax.jit`-compiled.
- **Gradient** `jax.grad(L, argnums=p)`; wrap so it returns NumPy arrays.
- **Optimiser** `scipy.optimize.minimize(L, p0, jac=gradL, method="L-BFGS-B")`.
  Use **multiple random restarts** for `p0` (e.g. 8, small random + a couple of
  hand-seeded sign patterns) and keep the lowest-loss optimum. Report restart
  spread as a robustness note.
- Speed: jit the rollout/loss once; scan (not Python loops) over T.

## Pre-optimisation loss scans (the "before optimisation" figures)

From the fixed upright IC, with the other gains fixed at zero (or a sensible
stabilising guess — state which):
- **1-D scans**: sweep each of the 4 components of `p` individually over a
  sensible range, plot L vs that gain. → `task3_2_loss_scans_1d.png`.
- **2-D contour**: sweep the two dominant gains `p_theta` and `p_theta_dot`
  jointly, filled contour of L. → `task3_2_loss_contours_2d.png`.
These show the basin the optimiser descends into and motivate the restarts.

## σ_l selection — do this RIGOROUSLY (σ_l-independent metrics)

The objective changes with σ_l, so DO NOT rank σ_l by optimised loss. Protocol:
1. Fix the optimisation IC = `X0_up`.
2. Define a held-out spread of ~20 ICs: small random displacements,
   e.g. θ ∈ ±0.3 rad, θ̇ ∈ ±0.5, x ∈ ±0.5, ẋ ∈ ±0.5 (fixed seed).
3. Define σ_l-INDEPENDENT quality metrics, evaluated on a long-enough
   verification rollout (e.g. 80 steps) of the controller on each held-out IC:
   - settling time to |θ| < 0.05 rad (NaN/cap if never),
   - mean |θ| over the final 1 s (steady-state error),
   - peak |θ| excursion,
   - total control effort Σ a²,
   - fraction of held-out ICs "stabilised" (|θ|<0.05 held to the end).
4. **Sweep scalar σ_l** on a log grid ≈ [0.1, 3], ~6 values. For each: full
   multi-restart optimisation of `p` from the fixed IC, then score the resulting
   controller on the held-out spread. Plot the metrics vs σ_l.
   → `task3_2_sigma_sweep.png`. Pick the σ_l whose curve best trades settling
   time / steady-state error / effort, and state the choice + reason.
5. **Per-component refinement** from the best scalar: test one structured
   variant that tightens σ on θ (care about angle) and loosens σ on x (don't
   care where the cart ends up). Keep whichever scores better on the held-out
   metrics. Report the final per-component σ_l vector and the reasoning.

This yields a defensible "I chose σ_l = … because the sweep shows …" with a
figure behind it, which is exactly what the handout asks.

## The two experiments (the heart of the deliverable)

**A. Slightly displaced upright** (`X0_up = [0,0,0.2,0]`): optimise `p`. Expect
SUCCESS — linear feedback is exactly right near an unstable equilibrium.
Deliverables:
  - `task3_2_timeevo_upright.png`: time evolution of all 4 state variables AND
    the action a(t) under the optimised policy, showing θ→0 and held. THIS IS
    THE MONEY PLOT the handout asks for ("demonstrate the pole is kept upright").
  - `task3_2_phase_upright.png`: θ vs θ̇ phase portrait spiralling into the
    origin.
  - (optional) `task3_2_convergence.png`: L vs optimiser iteration.

**B. Downward stable** (`X0_down = [0,0,pi,0]`): re-run the SAME optimisation.
Expect FAILURE. Deliverable `task3_2_timeevo_downward.png` showing it does not
stabilise. The report point — make the script print evidence for it:
  - at θ=π the system sits in the SATURATED region of the loss (l≈1) where the
    gradient ≈ 0, so L-BFGS-B sees almost no signal and stalls (print initial
    ‖∇L‖ at the downward IC vs the upright IC to evidence this);
  - more fundamentally a SINGLE linear gain cannot both pump energy in to swing
    up and then gently balance — swing-up is intrinsically nonlinear.
  - The ONE max_force check: re-run B with a larger max_force (e.g. 40, 80) and
    show it STILL fails → the failure is structural, not actuator-limited.
    Print the outcome. (Answers the handout's "you may also need to change
    max_force" and "what happened from the downward position".)

## Robustness (nice for the report, cheap to add)

After picking the final `p` and σ_l from experiment A, evaluate that controller
on the held-out IC spread and report the stabilised fraction + metric summary
in the JSON. Optionally overlay a few held-out rollouts on the time-evo figure.

## Required outputs

Figures (to `figures/`, `task3_2_<name>.png`):
- `task3_2_loss_scans_1d.png`   — 1-D loss scans over each gain (pre-opt)
- `task3_2_loss_contours_2d.png`— 2-D loss contour over (p_theta, p_theta_dot)
- `task3_2_sigma_sweep.png`     — quality metrics vs σ_l (selection evidence)
- `task3_2_timeevo_upright.png` — 4 states + action under optimised policy (KEY)
- `task3_2_phase_upright.png`   — θ vs θ̇ spiral into origin
- `task3_2_timeevo_downward.png`— downward-start failure
- (optional) `task3_2_convergence.png`
- `task3_2_results.json` — dynamics-match max error; chosen σ_l vector (+ how
  chosen); optimised `p` for upright (and downward attempt); max_force; T; ICs;
  per-IC final trajectory loss; held-out stabilised fraction + metric summary;
  σ_l sweep data; downward-case ‖∇L‖ and max_force-check outcomes; restart
  spread.

## Sanity checks (expected)

- JAX-vs-NumPy dynamics match < 1e-6 (GATE).
- Loss reduces to the built-in `cartpole.py` `_loss` at σ_l = 0.5 scalar (assert).
- Upright: optimised policy drives |θ| < 0.05 and holds it; forces stay well
  inside (−20, 20) (no saturation needed near upright).
- Downward: does NOT stabilise; ‖∇L‖ at the downward IC ≪ ‖∇L‖ at the upright
  IC; raising max_force does not rescue it.

## Engineering conventions

- Single self-contained script `week3/task3_2.py`. Import the JAX dynamics from
  within it (or a small local module — but a single file is preferred, matching
  3.1). Local `cartpole.py` copy in the repo root for the NumPy reference.
- `TASK3_2_QUICK=1` env toggle → small fast smoke test (few restarts, short T,
  coarse σ_l grid, few held-out ICs) so the full run can be validated cheaply
  before committing to the expensive version.
- Reproducible: seed all RNGs. Print a concise progress log per stage.
- Do not use a git worktree; work in the repo root.

## OPTIONAL extension (only if core is complete and time allows)

sin/cos policy variant: `p(X) = p · [x, ẋ, sinθ, cosθ, θ̇]` (length-5 `p`).
Build it behind a config switch, re-run experiment A, and compare to the raw
4-D policy. Expected: numerically near-identical near upright (the point worth
reporting), and STILL unable to swing up from downward (linear ⇒ no energy
pumping, regardless of representation). Label its figures `..._sincos.png`.
Do NOT let this delay or complicate the core raw-4-D deliverable.
