"""
SF3 Machine Learning -- Week 3, Task 3.2 (rebuild)
==================================================

Optimise a linear feedback policy  a = p . X  to balance the cartpole at the
upright unstable equilibrium under the TRUE dynamics, per the handout:

  1. trajectory loss (eq. 17 with per-component sigma_l, eq. 18 sum over T);
  2. pre-optimisation 1-D / 2-D loss scans over components of p;
  3. optimise p with JAX gradients (jit + lax.scan rollout, L-BFGS-B with
     multi-restart) from a fixed IC -- first slightly displaced from upright,
     then from the downward position;
  4. time-evolution plots demonstrating the pole is kept upright.

THE FIX vs the first attempt (task3_2.py): theta is wrapped to [-pi, pi] at
every control step before the policy and the loss see it. The dynamics are
periodic in theta so the physics does not care, but the linear policy p . X
and the eq. 17 loss are NOT periodic -- with raw theta an over-rotation never
flips the sign of the theta feedback, so the policy cannot pump energy and
swing-up is impossible (the cart just runs away). With the wrap, the sign flip
at +/-pi gives bang-bang energy pumping and a single linear policy can both
swing up AND balance (verified against a known-good peer policy).

Run:
    python week3/task3_2_v2.py                  # full run
    TASK3_2_QUICK=1 python week3/task3_2_v2.py  # smoke test
"""

from __future__ import annotations
from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad

from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cartpole import CartPole, _loss as cp_loss   # noqa: E402

# =========================================================================
# CONFIG
# =========================================================================
QUICK = os.environ.get("TASK3_2_QUICK", "0") == "1"

CFG = {
    # horizons (one performAction = 0.1 s)
    "T":          40,     # optimisation horizon: 4 s ~ a couple of oscillation periods
    "T_verify":   200,    # verification rollout (20 s) -- must hold well past T
    "dt":         0.1,
    # per-experiment actuator ceiling (handout: "you may also need to change max_force")
    "max_force_up":   30.0,   # cartpole default; upright needs only ~5
    "max_force_down": 30.0,   # swing-up needs more authority than the default 20
    # eq. 17 loss scales [x, xdot, theta, thetadot] -- chosen, per handout, by hand:
    # tight on theta (the control target), loose on x (cart may drift), moderate
    # on the velocities (they must swing to catch/pump the pole).
    "sigma_l_up":   [3.0, 1.0, 0.5, 1.0],
    "sigma_l_down": [3.0, 1.0, 0.5, 1.0],
    #"sigma_l_down": [8.0, 8.0, 8.0, 8.0],  # looser on theta to allow swing-up overshoot;
    # ICs
    "X0_up":   [0.0, 0.0, 0.2, 0.0],
    "X0_down": [0.0, 0.0, np.pi, 0.0],
    # optimiser
    "p_bound":    150.0,  # |p_i| <= p_bound for L-BFGS-B
    "n_restarts": 8,      # random restarts (hand seeds are added on top)
    "maxiter":    200,
    "ftol":       1e-10,
    "gtol":       1e-7,
    # positive log-uniform restart range for the downward experiment (the wrap
    # makes positive theta-gains act as a swing-up pump)
    "p_pos_lo":   float(np.e),
    "p_pos_hi":   float(np.e ** 5),
    # metrics / checks
    "theta_settled": 0.05,
    "n_gate":     200,
    "gate_tol":   1e-6,
    "xcheck_tol": 1e-4,   # numpy-vs-JAX full-rollout agreement
    "seed":       0,
    # peer's known-good downward policy -- used ONLY as a post-hoc cross-check,
    # never as an optimiser seed
    "p_friend":   [2.718, 3.828, 30.19, 5.1699],
}

if QUICK:
    CFG.update({
        "T_verify":   80,
        "n_restarts": 2,
        "maxiter":    60,
        "n_gate":     40,
    })

# =========================================================================
# PHYSICS CONSTANTS (copied from cartpole.py; do not deviate)
# =========================================================================
_REF = CartPole(visual=False)
MAX_FORCE   = float(_REF.max_force)
SIM_STEPS   = int(_REF.sim_steps)
DELTA_TIME  = float(_REF.delta_time)
GRAVITY     = float(_REF.gravity)
CART_MASS   = float(_REF.cart_mass)
POLE_MASS   = float(_REF.pole_mass)
POLE_LENGTH = float(_REF.pole_length)
MU_C        = float(_REF.mu_c)
MU_P        = float(_REF.mu_p)
del _REF

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
FIG_PREFIX = "task3_2_v2"

STATE_LABELS = [r"$x$", r"$\dot x$", r"$\theta$", r"$\dot\theta$"]
GAIN_LABELS  = [r"$p_x$", r"$p_{\dot x}$", r"$p_\theta$", r"$p_{\dot\theta}$"]


def fig_path(name: str) -> Path:
    return FIG_DIR / f"{FIG_PREFIX}_{name}.png"


# =========================================================================
# 1. JAX DYNAMICS (verbatim physics from task3_2.py, already gated vs numpy)
# =========================================================================
def _performAction_jax_with_force(state, force):
    """50 semi-implicit Euler substeps for a given force. Update order
    (cart_vel -> pole_vel -> pole_angle -> cart_loc) matches cartpole.py."""
    dt = DELTA_TIME / SIM_STEPS

    def body(carry, _):
        cl, cv, pa, pv = carry
        s = jnp.sin(pa)
        c = jnp.cos(pa)
        m = 4.0 * (CART_MASS + POLE_MASS) - 3.0 * POLE_MASS * c ** 2
        cart_accel = (
            2.0 * (POLE_LENGTH * POLE_MASS * pv ** 2 * s
                   + 2.0 * (force - MU_C * cv))
            - 3.0 * POLE_MASS * GRAVITY * c * s
            + 6.0 * MU_P * pv * c / POLE_LENGTH
        ) / m
        pole_accel = (
            -3.0 * c * (2.0 / POLE_LENGTH) * (
                POLE_LENGTH / 2.0 * POLE_MASS * pv ** 2 * s
                + force - MU_C * cv
            )
            + 6.0 * (CART_MASS + POLE_MASS) / (POLE_MASS * POLE_LENGTH)
              * (POLE_MASS * GRAVITY * s - 2.0 / POLE_LENGTH * MU_P * pv)
        ) / m
        cv = cv + dt * cart_accel
        pv = pv + dt * pole_accel
        pa = pa + dt * pv
        cl = cl + dt * cv
        return (cl, cv, pa, pv), None

    init = (state[0], state[1], state[2], state[3])
    final, _ = jax.lax.scan(body, init, None, length=SIM_STEPS)
    cl, cv, pa, pv = final
    return jnp.stack([cl, cv, pa, pv])


def performAction_jax(state, action, max_force):
    """One control step: tanh-bound the action, run the substeps."""
    force = max_force * jnp.tanh(action / max_force)
    return _performAction_jax_with_force(state, force)


def wrap_state(state):
    """Wrap theta (component 2) to [-pi, pi). Piecewise-linear in theta with
    gradient 1 almost everywhere, so JAX autodiff flows through unchanged.
    THE fix: the policy and the loss must see the physical angle."""
    th = jnp.mod(state[2] + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    return state.at[2].set(th)


def wrap_theta_np(theta: float) -> float:
    """Numpy twin of wrap_state's angle map, INCLUDING the boundary convention
    (pi -> -pi). cartpole._remap_angle keeps +pi, which would start the numpy
    cross-check rollout on the other side of the policy discontinuity and
    produce a mirror-image trajectory."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def gating_check_dynamics(rng, n, tol) -> tuple[float, bool]:
    """Single-step JAX vs numpy comparison on random (state, action) pairs.
    Raw states on both sides (no wrap) -- this checks ONLY the physics."""
    sim = CartPole(visual=False)
    step_jit = jit(lambda s, a: performAction_jax(s, a, MAX_FORCE))
    max_diff = 0.0
    for _ in range(n):
        st = np.array([rng.uniform(-2, 2), rng.uniform(-3, 3),
                       rng.uniform(-np.pi, np.pi), rng.uniform(-3, 3)])
        a = rng.uniform(-30.0, 30.0)
        sim.setState(st)
        sim.performAction(float(a))
        ref = sim.getState()
        out = np.asarray(step_jit(jnp.asarray(st), jnp.asarray(a)))
        max_diff = max(max_diff, float(np.max(np.abs(ref - out))))
    return max_diff, (max_diff < tol)


# =========================================================================
# 2. LOSS (eq. 17 per-component sigma_l, target X0 = 0) + builtin check
# =========================================================================
def pointwise_loss(state, sigma_l_vec):
    """l(X) = 1 - exp(-sum_j X_j^2 / (2 sigma_l_j^2))."""
    inv2s2 = 1.0 / (2.0 * sigma_l_vec ** 2)
    return 1.0 - jnp.exp(-jnp.sum(state ** 2 * inv2s2))


def loss_check_against_builtin() -> bool:
    """With sigma_l = 0.5 scalar our loss must equal cartpole._loss exactly."""
    sigma = jnp.full(4, 0.5)
    rng = np.random.default_rng(123)
    for _ in range(50):
        st = rng.uniform(-2.0, 2.0, size=4)
        ours = float(pointwise_loss(jnp.asarray(st), sigma))
        ref = float(cp_loss(st))
        if abs(ours - ref) > 1e-12:
            return False
    return True


# =========================================================================
# 3. ROLLOUT AND TRAJECTORY LOSS (jit + lax.scan, theta wrapped every step)
# =========================================================================
def make_rollout_fn(T: int, max_force: float):
    """jit'd rollout(p, X0) -> (states[T,4] wrapped, actions[T])."""
    def rollout(p, X0):
        def step(state, _):
            a = jnp.dot(p, state)
            nxt = wrap_state(performAction_jax(state, a, max_force))
            return nxt, (nxt, a)
        _, (states, actions) = jax.lax.scan(step, wrap_state(X0), None, length=T)
        return states, actions
    return jit(rollout)


def make_loss_fn(T: int, max_force: float):
    """Returns (loss_jit, grad_jit, loss_and_grad). Trajectory loss is the
    eq. 18 sum of eq. 17 over the T post-step (wrapped) states."""
    def traj_loss(p, X0, sigma_l_vec):
        def step(state, _):
            a = jnp.dot(p, state)
            nxt = wrap_state(performAction_jax(state, a, max_force))
            return nxt, pointwise_loss(nxt, sigma_l_vec)
        _, losses = jax.lax.scan(step, wrap_state(X0), None, length=T)
        return jnp.sum(losses)

    loss_jit = jit(traj_loss)
    grad_jit = jit(grad(traj_loss, argnums=0))

    def loss_and_grad(p_np, X0_np, sigma_np):
        args = (jnp.asarray(p_np, dtype=jnp.float64),
                jnp.asarray(X0_np, dtype=jnp.float64),
                jnp.asarray(sigma_np, dtype=jnp.float64))
        return float(loss_jit(*args)), np.asarray(grad_jit(*args), dtype=np.float64)

    return loss_jit, grad_jit, loss_and_grad


def rollout_numpy(p, X0, T, max_force):
    """Independent rollout on the true numpy CartPole with per-step remap --
    cross-check that the JAX path implements the same closed loop."""
    sim = CartPole(visual=False)
    sim.max_force = max_force
    X0 = np.asarray(X0, dtype=np.float64).copy()
    X0[2] = wrap_theta_np(X0[2])
    sim.setState(X0)
    states, actions = [X0], np.zeros(T)
    for t in range(T):
        a = float(np.dot(p, states[-1]))
        sim.performAction(a)
        actions[t] = a
        s = np.array(sim.getState(), dtype=np.float64)
        s[2] = wrap_theta_np(s[2])
        sim.setState(s)
        states.append(s)
    return np.stack(states), actions


def crosscheck_numpy_vs_jax(p, X0, rollout_jit, T, max_force) -> float:
    """Max abs state difference over the full closed-loop rollout (theta
    compared on the circle so a -pi/+pi boundary disagreement scores 0)."""
    s_np, _ = rollout_numpy(p, X0, T, max_force)
    s_j, _ = rollout_jit(jnp.asarray(p, dtype=jnp.float64),
                         jnp.asarray(X0, dtype=jnp.float64))
    s_jax = np.vstack([s_np[0][None, :], np.asarray(s_j)])
    diff = np.abs(s_np - s_jax)
    dth = np.abs(s_np[:, 2] - s_jax[:, 2])
    diff[:, 2] = np.minimum(dth, 2.0 * np.pi - dth)
    return float(np.max(diff))


# =========================================================================
# 4. OPTIMISATION (multi-restart L-BFGS-B)
# =========================================================================
def optimise_policy(loss_and_grad, X0, sigma_l_vec, n_restarts, seed,
                    hand_seeds=None, positive_restarts=False,
                    verbose=True) -> dict:
    """Multi-restart L-BFGS-B over p in [-p_bound, p_bound]^4. With
    positive_restarts, half the random starts are drawn log-uniform in
    [e, e^5]^4 (swing-up pump candidates), the rest N(0, 5)."""
    rng = np.random.default_rng(seed)
    b = CFG["p_bound"]
    bounds = [(-b, b)] * 4

    inits = [np.asarray(h, dtype=np.float64) for h in (hand_seeds or [])]
    for k in range(n_restarts):
        if positive_restarts and k % 2 == 0:
            inits.append(np.exp(rng.uniform(np.log(CFG["p_pos_lo"]),
                                            np.log(CFG["p_pos_hi"]), size=4)))
        else:
            inits.append(rng.normal(0.0, 5.0, size=4))
    inits = [np.clip(p0, -b, b) for p0 in inits]

    def f_and_g(p_np):
        L, gL = loss_and_grad(p_np, X0, sigma_l_vec)
        if not np.isfinite(L) or not np.isfinite(gL).all():
            return 1e20, np.zeros_like(p_np)
        return L, gL

    runs = []
    for k, p0 in enumerate(inits):
        res = minimize(f_and_g, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": CFG["maxiter"], "ftol": CFG["ftol"],
                                "gtol": CFG["gtol"]})
        runs.append({"p0": p0.tolist(), "p": res.x.tolist(),
                     "loss": float(res.fun), "nit": int(res.nit)})
        if verbose:
            print(f"  restart {k}: p0=[{', '.join(f'{v:+.2f}' for v in p0)}]"
                  f" -> p=[{', '.join(f'{v:+.2f}' for v in res.x)}]"
                  f"  L={res.fun:.4e}  iters={res.nit}")

    best = min(runs, key=lambda r: r["loss"])
    return {"p_best": np.array(best["p"]), "loss_best": best["loss"],
            "runs": runs}


# =========================================================================
# 5. METRICS
# =========================================================================
def settling_step(theta, thresh) -> int | None:
    """Smallest t with |theta(t')| < thresh for all t' >= t; None if never."""
    below = np.abs(theta) < thresh
    if not below.any():
        return None
    above = np.where(~below)[0]
    if len(above) == 0:
        return 0
    t = int(above[-1] + 1)
    return t if t < len(theta) else None


def rollout_metrics(states, actions, max_force) -> dict:
    """States are wrapped, so theta IS the physical angle."""
    theta = states[:, 2]
    dt = CFG["dt"]
    nfin = max(1, int(round(1.0 / dt)))
    st = settling_step(theta[1:], CFG["theta_settled"])
    forces = max_force * np.tanh(actions / max_force)
    return {
        "final_state":     states[-1].tolist(),
        "final_abs_theta": float(abs(theta[-1])),
        "peak_abs_theta":  float(np.max(np.abs(theta))),
        "settled":         st is not None,
        "settle_time_s":   (st * dt) if st is not None else None,
        "held_to_end":     bool(np.all(np.abs(theta[-nfin:]) < CFG["theta_settled"])),
        "max_abs_action":  float(np.max(np.abs(actions))),
        "max_abs_force":   float(np.max(np.abs(forces))),
        "force_sat_frac":  float(np.mean(np.abs(forces) > 0.95 * max_force)),
    }


# =========================================================================
# 6. PLOTS
# =========================================================================
def _mask_wraps(theta):
    """NaN-out points where theta jumps across +/-pi so plots don't draw
    spurious vertical/horizontal lines through the wrap."""
    th = theta.astype(float).copy()
    th[np.where(np.abs(np.diff(th)) > np.pi)[0]] = np.nan
    return th


def plot_loss_scans_1d(loss_jit, X0, sigma_l_vec, p_base, savepath, n=60):
    """Sweep each gain individually, others held at p_base."""
    lo = [-30.0, -30.0, -10.0, -10.0]
    hi = [ 30.0,  30.0, 100.0,  40.0]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for k in range(4):
        sweep = np.linspace(lo[k], hi[k], n)
        Ls = [float(loss_jit(jnp.asarray(np.where(np.arange(4) == k, v, p_base)),
                             jnp.asarray(X0), jnp.asarray(sigma_l_vec)))
              for v in sweep]
        ax = axes[k]
        ax.plot(sweep, Ls, "k-", lw=1.5)
        ax.axvline(p_base[k], color="tab:blue", ls="--", lw=1, label="base")
        ax.set_xlabel(GAIN_LABELS[k])
        ax.set_ylabel("L(p)")
        ax.grid(True, alpha=0.4)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Pre-optimisation 1-D loss scans  (X0={list(np.round(X0, 3))}, "
                 f"$\\sigma_l$={list(sigma_l_vec)}, base p={list(p_base)})")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_loss_contour_2d(loss_jit, X0, sigma_l_vec, p_base, title, savepath, n=35):
    """Contour of L over (p_theta, p_thetadot), other gains at p_base."""
    pth = np.linspace(-20.0, 100.0, n)
    ptd = np.linspace(-10.0, 40.0, n)
    PT, PD = np.meshgrid(pth, ptd)
    Z = np.zeros_like(PT)
    for i in range(n):
        for j in range(n):
            p = p_base.copy()
            p[2], p[3] = PT[i, j], PD[i, j]
            Z[i, j] = float(loss_jit(jnp.asarray(p), jnp.asarray(X0),
                                     jnp.asarray(sigma_l_vec)))
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cs = ax.contourf(PT, PD, Z, levels=25, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="L(p)")
    ax.set_xlabel(r"$p_\theta$")
    ax.set_ylabel(r"$p_{\dot\theta}$")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_timeevo(states, actions, max_force, title, savepath):
    """5 panels: 4 (wrapped) states + action/force."""
    dt = CFG["dt"]
    T = len(actions)
    t_state = np.arange(T + 1) * dt
    t_act = np.arange(T) * dt
    fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    for k in range(4):
        ax = axes[k]
        y = _mask_wraps(states[:, k]) if k == 2 else states[:, k]
        ax.plot(t_state, y, "k-", lw=1.6)
        if k == 2:
            ax.axhline( CFG["theta_settled"], color="tab:green", ls=":", lw=1)
            ax.axhline(-CFG["theta_settled"], color="tab:green", ls=":", lw=1,
                       label=f"$\\pm${CFG['theta_settled']}")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_ylabel(STATE_LABELS[k])
        ax.grid(True, alpha=0.4)
    ax = axes[4]
    forces = max_force * np.tanh(actions / max_force)
    ax.plot(t_act, actions, color="tab:red", lw=1.3, label="action")
    ax.plot(t_act, forces, color="tab:orange", lw=1.0, ls="--", label="force")
    ax.axhline( max_force, color="tab:gray", ls=":", lw=1)
    ax.axhline(-max_force, color="tab:gray", ls=":", lw=1)
    ax.set_ylabel("action / force")
    ax.set_xlabel("time / s")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.4)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_phase(states, title, savepath):
    theta = _mask_wraps(states[:, 2])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(theta, states[:, 3], "k-", lw=1.0, alpha=0.8)
    ax.scatter([states[0, 2]], [states[0, 3]], color="tab:green", s=70,
               zorder=5, label="start")
    ax.scatter([states[-1, 2]], [states[-1, 3]], color="tab:red", s=70,
               zorder=5, label="end")
    ax.scatter([0], [0], color="tab:blue", s=70, marker="x", zorder=5,
               label="target")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot\theta$")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


# =========================================================================
# 7. EXPERIMENT RUNNER
# =========================================================================
def run_experiment(name, X0, sigma_l, max_force, hand_seeds, seed,
                   positive_restarts=False) -> dict:
    """Optimise p from the given IC, verify over the long horizon, plot."""
    T, T_verify = CFG["T"], CFG["T_verify"]
    X0 = np.asarray(X0, dtype=np.float64)
    sigma_l = np.asarray(sigma_l, dtype=np.float64)

    print(f"\n=== Experiment {name}: X0={X0.tolist()}, max_force={max_force}, "
          f"sigma_l={sigma_l.tolist()} ===")
    loss_jit, _, loss_and_grad = make_loss_fn(T, max_force)
    res = optimise_policy(loss_and_grad, X0, sigma_l,
                          n_restarts=CFG["n_restarts"], seed=seed,
                          hand_seeds=hand_seeds,
                          positive_restarts=positive_restarts)
    p_best = res["p_best"]
    print(f"[{name}] best p = [{', '.join(f'{v:+.4f}' for v in p_best)}]"
          f"  L(T={T}) = {res['loss_best']:.4e}")

    # long-horizon verification
    rollout_jit = make_rollout_fn(T_verify, max_force)
    s_j, a_j = rollout_jit(jnp.asarray(p_best), jnp.asarray(X0))
    states = np.vstack([X0[None, :], np.asarray(s_j)])
    states[0, 2] = wrap_theta_np(states[0, 2])
    actions = np.asarray(a_j)
    mets = rollout_metrics(states, actions, max_force)
    print(f"[{name}] VERIFY over {T_verify} steps: held={mets['held_to_end']}  "
          f"settle={mets['settle_time_s']}s  final|theta|={mets['final_abs_theta']:.4f}  "
          f"peak|theta|={mets['peak_abs_theta']:.3f}  "
          f"max|F|={mets['max_abs_force']:.2f} (sat {mets['force_sat_frac']:.1%})")

    # independent numpy cross-check of the full closed loop
    xdiff = crosscheck_numpy_vs_jax(p_best, X0, rollout_jit, T_verify, max_force)
    ok = xdiff < CFG["xcheck_tol"]
    print(f"[{name}] numpy-vs-JAX closed-loop max diff = {xdiff:.3e}  "
          f"{'OK' if ok else 'MISMATCH -- investigate!'}")

    plot_timeevo(states, actions, max_force,
                 f"Experiment {name} — $p$={[f'{v:.2f}' for v in p_best]}, "
                 f"max_force={max_force}, $\\sigma_l$={sigma_l.tolist()}",
                 fig_path(f"timeevo_{name}"))
    plot_phase(states, f"Experiment {name} — phase portrait",
               fig_path(f"phase_{name}"))

    return {"X0": X0.tolist(), "sigma_l": sigma_l.tolist(),
            "max_force": max_force, "p_best": p_best.tolist(),
            "loss_best": res["loss_best"], "metrics": mets,
            "xcheck_max_diff": xdiff, "xcheck_ok": ok,
            "runs": res["runs"], "_loss_jit": loss_jit}


# =========================================================================
# 8. MAIN
# =========================================================================
def main():
    t0 = time.time()
    print(f"=== Task 3.2 v2 (QUICK={QUICK}) ===")
    print(f"    jax {jax.__version__} on {jax.devices()}")
    print(f"    T={CFG['T']}  T_verify={CFG['T_verify']}  "
          f"max_force up/down = {CFG['max_force_up']}/{CFG['max_force_down']}")

    rng = np.random.default_rng(CFG["seed"])

    # --- gate: JAX physics == numpy CartPole ---
    max_diff, passed = gating_check_dynamics(rng, CFG["n_gate"], CFG["gate_tol"])
    print(f"\n[gate] JAX vs numpy single-step on {CFG['n_gate']} random pairs: "
          f"max diff = {max_diff:.3e}  {'PASS' if passed else 'FAIL'}")
    if not passed:
        sys.exit("[gate] FAIL -- aborting, downstream results would be invalid.")

    # --- loss reduces to the builtin at sigma_l = 0.5 ---
    assert loss_check_against_builtin(), "loss does not match cartpole._loss"
    print("[loss-check] eq.17 loss == cartpole._loss at sigma_l=0.5: PASS")

    X0_up   = np.asarray(CFG["X0_up"])
    X0_down = np.asarray(CFG["X0_down"])
    sig_up   = np.asarray(CFG["sigma_l_up"])
    sig_down = np.asarray(CFG["sigma_l_down"])

    # --- pre-optimisation loss scans (handout requirement) ---
    print("\n[scans] 1-D scans + 2-D contours of L(p) before optimisation...")
    loss_jit_up, _, _ = make_loss_fn(CFG["T"], CFG["max_force_up"])
    loss_jit_dn, _, _ = make_loss_fn(CFG["T"], CFG["max_force_down"])
    p_base = np.array([0.0, 0.0, 30.0, 5.0])
    n_scan = 30 if QUICK else 60
    n_grid = 18 if QUICK else 35
    plot_loss_scans_1d(loss_jit_up, X0_up, sig_up, p_base,
                       fig_path("loss_scans_1d"), n=n_scan)
    plot_loss_contour_2d(loss_jit_up, X0_up, sig_up, p_base,
                         "L over ($p_\\theta$, $p_{\\dot\\theta}$) — upright IC",
                         fig_path("loss_contour_2d_upright"), n=n_grid)
    plot_loss_contour_2d(loss_jit_dn, X0_down, sig_down, p_base,
                         "L over ($p_\\theta$, $p_{\\dot\\theta}$) — downward IC",
                         fig_path("loss_contour_2d_downward"), n=n_grid)

    # --- Experiment A: slightly displaced from upright ---
    seeds_up = [np.zeros(4),
                np.array([-1.0, -2.0,  5.0, 1.0]),
                np.array([-1.0, -2.0, 20.0, 4.0]),
                np.array([-2.0, -3.0, 40.0, 8.0])]
    expA = run_experiment("upright", X0_up, sig_up, CFG["max_force_up"],
                          seeds_up, seed=CFG["seed"] + 100)

    # --- Experiment B: downward (swing-up + balance) ---
    # warm-start candidates: the upright optimum plus generic positive seeds
    # (positive gains + the theta wrap act as an energy pump). The peer policy
    # is NOT seeded -- it is only evaluated afterwards as a cross-check.
    seeds_down = [np.asarray(expA["p_best"]),
                  np.array([3.0, 4.0, 30.0, 5.0]),
                  np.array([5.0, 5.0, 60.0, 10.0])]
    expB = run_experiment("downward", X0_down, sig_down, CFG["max_force_down"],
                          seeds_down, seed=CFG["seed"] + 200,
                          positive_restarts=True)

    # --- peer policy cross-check (post-hoc only) ---
    print("\n=== Peer policy cross-check (NOT used in optimisation) ===")
    p_friend = np.asarray(CFG["p_friend"], dtype=np.float64)
    rollout_jit = make_rollout_fn(CFG["T_verify"], CFG["max_force_down"])
    s_j, a_j = rollout_jit(jnp.asarray(p_friend), jnp.asarray(X0_down))
    s_f = np.vstack([X0_down[None, :], np.asarray(s_j)])
    s_f[0, 2] = wrap_theta_np(s_f[0, 2])
    mets_f = rollout_metrics(s_f, np.asarray(a_j), CFG["max_force_down"])
    L_f = float(expB["_loss_jit"](jnp.asarray(p_friend), jnp.asarray(X0_down),
                                  jnp.asarray(sig_down)))
    print(f"[friend] p={p_friend.tolist()}  L(T={CFG['T']})={L_f:.4e}  "
          f"held={mets_f['held_to_end']}  settle={mets_f['settle_time_s']}s  "
          f"final|theta|={mets_f['final_abs_theta']:.4f}")

    # --- JSON dump ---
    for e in (expA, expB):
        e.pop("_loss_jit")
    out = {
        "config": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in CFG.items()},
        "quick": QUICK,
        "gate": {"max_diff": max_diff, "passed": passed},
        "upright": expA,
        "downward": expB,
        "friend_crosscheck": {"p": p_friend.tolist(), "loss_T": L_f,
                              "metrics": mets_f},
        "runtime_s": round(time.time() - t0, 1),
    }
    out_path = FIG_DIR / f"{FIG_PREFIX}_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] results -> {out_path.relative_to(ROOT)}  "
          f"({out['runtime_s']} s)")


if __name__ == "__main__":
    main()
