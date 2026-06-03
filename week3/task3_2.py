"""
SF3 Machine Learning -- Week 3, Task 3.2
========================================

Optimise a linear feedback policy  a = p . X  to balance the cartpole at the
upright unstable equilibrium under the TRUE dynamics (no learned model). The
true dynamics are re-implemented in JAX (semi-implicit Euler, same update
order as cartpole.py) so that gradients flow from the policy gains through
the rollout to the trajectory loss.

Stages:
  1. JAX performAction copy + GATING CHECK against numpy CartPole (<1e-6).
  2. Loss = handout eq. 17 with per-component sigma_l; assert it matches the
     built-in cartpole._loss when sigma_l = 0.5 scalar.
  3. Pre-optimisation 1-D / 2-D loss scans from X0_up = [0,0,0.2,0].
  4. sigma_l SELECTION by metric sweep (settling time / steady-state |theta|
     / peak |theta| / control effort / stabilised fraction) evaluated on a
     fixed held-out IC spread. Per-component refinement on top.
  5. Experiment A: optimise p from X0_up (expect success); time-evo + phase.
  6. Experiment B: optimise from X0_down=[0,0,pi,0] (expect failure);
     evidence the failure (||grad L|| at downward IC vs upright IC, plus a
     max_force-rescue check that should STILL fail).
  7. Robustness: evaluate the final p on the held-out IC spread.
  8. JSON dump + figures.

Run:
    python week3/task3_2.py             # full run
    TASK3_2_QUICK=1 python week3/task3_2.py    # smoke test
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
from cartpole import CartPole, _remap_angle, _loss as cp_loss   # noqa: E402

# =========================================================================
# CONFIG
# =========================================================================
QUICK = os.environ.get("TASK3_2_QUICK", "0") == "1"

CFG = {
    # horizon
    "T":               40,          # optimisation horizon (steps)
    "T_verify":        80,          # verification rollout (steps) for metrics
    "dt":              0.1,
    # optimisation
    "n_restarts":      8,
    "maxiter":         200,
    "ftol":            1e-10,
    "gtol":            1e-7,
    "p_bound":         100.0,       # |p_i| <= p_bound for L-BFGS-B
    # sigma sweep
    "sigma_grid":      [0.1, 0.18, 0.32, 0.56, 1.0, 1.78],   # log-spaced ~[0.1, 3]
    # held-out IC spread (for sigma selection AND robustness)
    "n_holdout":       20,
    "holdout_x":       0.5,
    "holdout_xdot":    0.5,
    "holdout_theta":   0.3,
    "holdout_thetadot":0.5,
    # ICs
    "X0_up":           [0.0, 0.0, 0.2, 0.0],
    "X0_down":         [0.0, 0.0, np.pi, 0.0],
    # downward max_force rescue check
    "downward_max_force_check": [40.0, 80.0],
    # stabilisation criterion
    "theta_settled":   0.05,
    # gating check
    "n_gate":          200,
    "gate_tol":        1e-6,
    # seeds
    "seed":            0,
}

if QUICK:
    CFG.update({
        "T":           20,
        "T_verify":    40,
        "n_restarts":  2,
        "maxiter":     60,
        "sigma_grid":  [0.18, 0.5, 1.5],
        "n_holdout":   5,
        "n_gate":      40,
        "downward_max_force_check": [80.0],
    })

# =========================================================================
# PHYSICS CONSTANTS (copied from cartpole.py; do not deviate)
# =========================================================================
_REF_SIM = CartPole(visual=False)
MAX_FORCE   = float(_REF_SIM.max_force)
SIM_STEPS   = int(_REF_SIM.sim_steps)
DELTA_TIME  = float(_REF_SIM.delta_time)
GRAVITY     = float(_REF_SIM.gravity)
CART_MASS   = float(_REF_SIM.cart_mass)
POLE_MASS   = float(_REF_SIM.pole_mass)
POLE_LENGTH = float(_REF_SIM.pole_length)
MU_C        = float(_REF_SIM.mu_c)
MU_P        = float(_REF_SIM.mu_p)
del _REF_SIM

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
FIG_PREFIX = "task3_2"

STATE_LABELS = [r"$x$", r"$\dot x$", r"$\theta$", r"$\dot\theta$"]
GAIN_LABELS  = [r"$p_x$", r"$p_{\dot x}$", r"$p_\theta$", r"$p_{\dot\theta}$"]


def fig_path(name: str) -> Path:
    return FIG_DIR / f"{FIG_PREFIX}_{name}.png"


# =========================================================================
# 1. JAX DYNAMICS  (semi-implicit Euler, exact copy of cartpole.performAction)
# =========================================================================
def _performAction_jax_with_force(state, force):
    """Inner version: 50 semi-implicit Euler substeps for a given force.
    The update order (cart_vel -> pole_vel -> pole_angle -> cart_loc) matches
    cartpole.py exactly so that this is symplectic and bit-comparable."""
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


def performAction_jax(state, action, max_force=MAX_FORCE):
    """One control step. tanh-bound the action then run the substeps."""
    force = max_force * jnp.tanh(action / max_force)
    return _performAction_jax_with_force(state, force)


_performAction_jit = jit(performAction_jax)


def gating_check_dynamics(rng, n=200, tol=1e-6) -> tuple[float, bool]:
    """Compare jit'd JAX performAction to the numpy reference on n random
    (state, action) pairs. Returns (max_abs_diff, passed)."""
    sim = CartPole(visual=False)
    max_diff = 0.0
    for _ in range(n):
        x  = rng.uniform(-2.0, 2.0)
        xd = rng.uniform(-3.0, 3.0)
        th = rng.uniform(-np.pi, np.pi)
        td = rng.uniform(-3.0, 3.0)
        a  = rng.uniform(-30.0, 30.0)
        sim.setState([x, xd, th, td])
        sim.performAction(float(a))
        ref = sim.getState()
        jax_out = np.asarray(_performAction_jit(jnp.array([x, xd, th, td]),
                                                jnp.array(a)))
        d = float(np.max(np.abs(ref - jax_out)))
        if d > max_diff:
            max_diff = d
    return max_diff, (max_diff < tol)


# =========================================================================
# 2. LOSS  (handout eq. 17 with per-component sigma_l, X_target = 0)
# =========================================================================
def pointwise_loss(state, sigma_l_vec, X_target):
    """l(X) = 1 - exp(-sum_j (X_j - X0_j)^2 / (2 sigma_l_j^2))."""
    diff = state - X_target
    inv2s2 = 1.0 / (2.0 * sigma_l_vec ** 2)
    return 1.0 - jnp.exp(-jnp.sum(diff ** 2 * inv2s2))


def loss_assertion_against_builtin():
    """Assert that with sigma_l = 0.5 scalar and X_target = 0 we exactly
    reproduce cartpole._loss."""
    sigma = jnp.array([0.5, 0.5, 0.5, 0.5])
    X0 = jnp.zeros(4)
    rng = np.random.default_rng(123)
    for _ in range(50):
        st = rng.uniform(-2.0, 2.0, size=4)
        ours = float(pointwise_loss(jnp.asarray(st), sigma, X0))
        ref  = float(cp_loss(st))
        assert abs(ours - ref) < 1e-12, (ours, ref, st)
    return True


# =========================================================================
# 3. ROLLOUT, TRAJECTORY LOSS, GRADIENT  (the Task-2.2 pattern, jitted)
# =========================================================================
def make_rollout_fn(T: int, max_force: float = MAX_FORCE):
    """Return jit'd rollout(p, X0) -> (states[T,4], actions[T])."""
    def rollout(p, X0):
        def step(state, _):
            a = jnp.dot(p, state)
            next_state = performAction_jax(state, a, max_force=max_force)
            return next_state, (next_state, a)
        _, (states, actions) = jax.lax.scan(step, X0, None, length=T)
        return states, actions
    return jit(rollout)


def make_loss_fn(T: int, max_force: float = MAX_FORCE):
    """Return jit'd loss_and_grad(p_np, X0_np, sigma_np) -> (L, grad_L)."""
    def loss_p(p, X0, sigma_l_vec):
        def step(state, _):
            a = jnp.dot(p, state)
            next_state = performAction_jax(state, a, max_force=max_force)
            inv2s2 = 1.0 / (2.0 * sigma_l_vec ** 2)
            l = 1.0 - jnp.exp(-jnp.sum(next_state ** 2 * inv2s2))
            return next_state, l
        _, losses = jax.lax.scan(step, X0, None, length=T)
        return jnp.sum(losses)
    loss_jit = jit(loss_p)
    grad_jit = jit(grad(loss_p, argnums=0))

    def loss_and_grad(p_np, X0_np, sigma_np):
        p_j     = jnp.asarray(p_np, dtype=jnp.float64)
        X0_j    = jnp.asarray(X0_np, dtype=jnp.float64)
        sig_j   = jnp.asarray(sigma_np, dtype=jnp.float64)
        L  = float(loss_jit(p_j, X0_j, sig_j))
        gL = np.asarray(grad_jit(p_j, X0_j, sig_j), dtype=np.float64)
        return L, gL

    return loss_jit, grad_jit, loss_and_grad


# =========================================================================
# 4. OPTIMISATION  (multi-restart L-BFGS-B)
# =========================================================================
def optimise_policy(loss_and_grad, X0, sigma_l_vec, n_restarts: int,
                    seed: int, hand_seeds: list[np.ndarray] | None = None,
                    bounds=None, verbose: bool = True) -> dict:
    """Multi-restart L-BFGS-B on p. `n_restarts` is the count of RANDOM
    restarts -- the hand_seeds (if any) are always included on top. Returns
    dict with best p, loss, all run summaries, restart spread."""
    rng = np.random.default_rng(seed)

    inits = []
    if hand_seeds:
        for hs in hand_seeds:
            inits.append(np.asarray(hs, dtype=np.float64))
    for _ in range(n_restarts):
        inits.append(rng.normal(0.0, 5.0, size=4))

    if bounds is None:
        b = CFG["p_bound"]
        bounds = [(-b, b)] * 4

    def f_and_g(p_np):
        try:
            L, gL = loss_and_grad(p_np, X0, sigma_l_vec)
        except Exception:
            return 1e20, np.zeros_like(p_np)
        if not np.isfinite(L) or not np.isfinite(gL).all():
            return 1e20, np.zeros_like(p_np)
        return L, gL

    runs = []
    for k, p0 in enumerate(inits):
        res = minimize(f_and_g, p0, jac=True, method="L-BFGS-B",
                       bounds=bounds,
                       options={"maxiter": CFG["maxiter"],
                                "ftol": CFG["ftol"],
                                "gtol": CFG["gtol"]})
        runs.append({"p0": p0.copy(), "p": res.x.copy(),
                     "loss": float(res.fun), "nit": int(res.nit),
                     "success": bool(res.success), "message": str(res.message)})
        if verbose:
            print(f"  restart {k}: p0=[{', '.join(f'{v:+.2f}' for v in p0)}]"
                  f" -> p=[{', '.join(f'{v:+.2f}' for v in res.x)}]"
                  f"  L={res.fun:.4e}  iters={res.nit}")

    losses = [r["loss"] for r in runs]
    best = runs[int(np.argmin(losses))]
    lmin, lmax = float(min(losses)), float(max(losses))
    spread = (lmax - lmin) / max(abs(lmin), 1e-30)
    return {
        "p_best":   best["p"],
        "loss_best": best["loss"],
        "runs":     runs,
        "loss_min": lmin,
        "loss_max": lmax,
        "spread_rel": spread,
    }


# =========================================================================
# 5. METRICS  (sigma_l-independent quality of a controller)
# =========================================================================
def settling_step(theta_traj: np.ndarray, thresh: float) -> int | None:
    """Smallest t such that |theta(t')| < thresh for all t' >= t. None if never."""
    below = np.abs(theta_traj) < thresh
    if not below.any():
        return None
    T = len(theta_traj)
    last_above = np.where(~below)[0]
    if len(last_above) == 0:
        return 0
    t_settle = int(last_above[-1] + 1)
    if t_settle >= T:
        return None
    return t_settle


def metrics_one_traj(states: np.ndarray, actions: np.ndarray,
                     dt: float, theta_settled: float) -> dict:
    """Quality metrics for a single rollout, sigma_l-INDEPENDENT."""
    theta = states[:, 2]
    T = len(theta)
    # final-second window (last 1s = last int(1/dt) steps)
    nfin = max(1, int(round(1.0 / dt)))
    ss_abs_theta  = float(np.mean(np.abs(theta[-nfin:])))
    peak_abs_theta = float(np.max(np.abs(theta)))
    effort = float(np.sum(actions ** 2))
    st = settling_step(theta, theta_settled)
    settled = (st is not None)
    settle_time = (st * dt) if settled else float("nan")
    held = bool(np.all(np.abs(theta[-nfin:]) < theta_settled))
    return {
        "settle_time":     settle_time,   # seconds, NaN if never
        "settled":         settled,
        "ss_abs_theta":    ss_abs_theta,
        "peak_abs_theta":  peak_abs_theta,
        "effort":          effort,
        "stabilised":      held,         # |theta|<thresh held to the end
    }


def evaluate_controller(p, rollout_fn, ICs: np.ndarray, dt: float,
                        theta_settled: float) -> dict:
    """Run the policy on each IC, aggregate metrics."""
    p_j = jnp.asarray(p, dtype=jnp.float64)
    per_ic = []
    for ic in ICs:
        states_j, actions_j = rollout_fn(p_j, jnp.asarray(ic, dtype=jnp.float64))
        states  = np.asarray(states_j)
        actions = np.asarray(actions_j)
        per_ic.append(metrics_one_traj(states, actions, dt, theta_settled))
    stab_frac = float(np.mean([m["stabilised"] for m in per_ic]))
    # aggregates over the stabilised subset for settle_time (NaN-safe mean)
    set_times = np.array([m["settle_time"] for m in per_ic], dtype=float)
    finite    = np.isfinite(set_times)
    mean_set  = float(np.nanmean(set_times)) if finite.any() else float("nan")
    return {
        "per_ic":           per_ic,
        "stab_frac":        stab_frac,
        "mean_settle_time": mean_set,
        "median_ss_theta":  float(np.median([m["ss_abs_theta"]   for m in per_ic])),
        "median_peak_theta":float(np.median([m["peak_abs_theta"] for m in per_ic])),
        "median_effort":    float(np.median([m["effort"]         for m in per_ic])),
    }


def make_holdout_ICs(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x   = rng.uniform(-CFG["holdout_x"],        CFG["holdout_x"],        n)
    xd  = rng.uniform(-CFG["holdout_xdot"],     CFG["holdout_xdot"],     n)
    th  = rng.uniform(-CFG["holdout_theta"],    CFG["holdout_theta"],    n)
    td  = rng.uniform(-CFG["holdout_thetadot"], CFG["holdout_thetadot"], n)
    return np.stack([x, xd, th, td], axis=1)


# =========================================================================
# 6. PLOTS
# =========================================================================
def plot_loss_scans_1d(loss_jit, X0, sigma_l_vec, p_base, savepath, n=80):
    """Sweep each gain individually over a sensible range, others held at p_base."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    grid_lo = [-30.0, -30.0, -10.0,  -10.0]
    grid_hi = [ 30.0,  30.0, 100.0,   40.0]
    for k in range(4):
        sweep = np.linspace(grid_lo[k], grid_hi[k], n)
        Ls = np.zeros(n)
        for i, v in enumerate(sweep):
            p = p_base.copy(); p[k] = v
            Ls[i] = float(loss_jit(jnp.asarray(p), jnp.asarray(X0),
                                   jnp.asarray(sigma_l_vec)))
        ax = axes[k]
        ax.plot(sweep, Ls, "k-", lw=1.5)
        ax.axvline(p_base[k], color="tab:blue", ls="--", lw=1, label="base")
        ax.set_xlabel(GAIN_LABELS[k])
        ax.set_ylabel("L(p)")
        ax.set_title(f"1-D scan of {GAIN_LABELS[k]}")
        ax.grid(True, alpha=0.4)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Pre-optimisation 1-D loss scans  (X0={list(X0)}, "
                 f"$\\sigma_l$={sigma_l_vec.tolist()})")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_loss_contours_2d(loss_jit, X0, sigma_l_vec, p_base, savepath, n=40):
    """Contour over (p_theta, p_thetadot) with the other gains at p_base."""
    p_th_grid  = np.linspace(-20.0, 100.0, n)
    p_td_grid  = np.linspace(-10.0,  40.0, n)
    PT, PD = np.meshgrid(p_th_grid, p_td_grid, indexing="xy")
    Z = np.zeros_like(PT)
    for i in range(n):
        for j in range(n):
            p = p_base.copy()
            p[2] = PT[i, j]
            p[3] = PD[i, j]
            Z[i, j] = float(loss_jit(jnp.asarray(p), jnp.asarray(X0),
                                     jnp.asarray(sigma_l_vec)))
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    levels = np.linspace(np.min(Z), np.max(Z), 25)
    cs = ax.contourf(PT, PD, Z, levels=levels, cmap="viridis")
    ax.contour(PT, PD, Z, levels=levels[::4], colors="white",
               linewidths=0.5, alpha=0.6)
    fig.colorbar(cs, ax=ax, label="L(p)")
    ax.scatter([p_base[2]], [p_base[3]], color="red", s=40,
               edgecolor="white", zorder=5, label="base")
    ax.set_xlabel(r"$p_\theta$")
    ax.set_ylabel(r"$p_{\dot\theta}$")
    ax.set_title(f"Loss contour over ($p_\\theta$, $p_{{\\dot\\theta}}$) "
                 f"with $p_x={p_base[0]:.2f}$, $p_{{\\dot x}}={p_base[1]:.2f}$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_sigma_sweep(sweep_data: list[dict], chosen_sigma_scalar: float,
                     savepath):
    sigmas = np.array([d["sigma"] for d in sweep_data])
    stab   = np.array([d["stab_frac"]         for d in sweep_data])
    set_t  = np.array([d["mean_settle_time"]  for d in sweep_data])
    ss_th  = np.array([d["median_ss_theta"]   for d in sweep_data])
    pk_th  = np.array([d["median_peak_theta"] for d in sweep_data])
    eff    = np.array([d["median_effort"]     for d in sweep_data])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        (stab,  "stabilised fraction",      "fraction",  False),
        (set_t, "mean settling time (s)",   "s",         False),
        (ss_th, "median steady-state $|\\theta|$ (last 1s)",   "rad", True),
        (pk_th, "median peak $|\\theta|$",  "rad",       True),
        (eff,   "median control effort $\\Sigma a^2$", "",      True),
    ]
    for ax, (y, title, ylabel, log) in zip(axes.flat, panels):
        ax.plot(sigmas, y, "o-", color="tab:red", lw=1.6)
        ax.axvline(chosen_sigma_scalar, color="tab:blue", ls="--", lw=1,
                   label=f"chosen $\\sigma_l$={chosen_sigma_scalar:.2g}")
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma_l$ (scalar)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=8)
    axes[1, 2].axis("off")
    fig.suptitle("$\\sigma_l$ sweep — per-controller quality metrics on the "
                 "held-out IC spread (each point is a full re-optimisation of p)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_timeevo(states: np.ndarray, actions: np.ndarray, dt: float,
                 title: str, savepath, holdout_states: np.ndarray | None = None):
    """5 panels: 4 states + action. States: T_verify+1 rows (incl X0).
    actions: T_verify rows."""
    T = actions.shape[0]
    t_state = np.arange(T + 1) * dt
    t_act   = np.arange(T) * dt
    fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    series = [states[:, 0], states[:, 1], states[:, 2], states[:, 3]]
    for k in range(4):
        ax = axes[k]
        if holdout_states is not None:
            for hs in holdout_states:
                ax.plot(t_state, hs[:, k], color="0.7", lw=0.8, alpha=0.6)
        ax.plot(t_state, series[k], "k-", lw=1.8, label="main rollout")
        if k == 2:
            ax.axhline( CFG["theta_settled"], color="tab:green", ls=":", lw=1)
            ax.axhline(-CFG["theta_settled"], color="tab:green", ls=":", lw=1,
                       label=f"$\\pm${CFG['theta_settled']}")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_ylabel(STATE_LABELS[k])
        ax.grid(True, alpha=0.4)
    ax = axes[4]
    ax.plot(t_act, actions, color="tab:red", lw=1.4)
    ax.axhline( MAX_FORCE, color="tab:gray", ls=":", lw=1, label=f"$\\pm$max_force")
    ax.axhline(-MAX_FORCE, color="tab:gray", ls=":", lw=1)
    ax.set_ylabel("action $a(t)$")
    ax.set_xlabel("time / s")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.4)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_phase(states: np.ndarray, title: str, savepath):
    theta  = states[:, 2]
    thetad = states[:, 3]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(theta, thetad, "k-", lw=1.0, alpha=0.8)
    ax.scatter([theta[0]],  [thetad[0]],  color="tab:green", s=70, zorder=5,
               label="start")
    ax.scatter([theta[-1]], [thetad[-1]], color="tab:red", s=70, zorder=5,
               label="end")
    ax.scatter([0], [0], color="tab:blue", s=70, marker="x", zorder=5,
               label="target")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot\theta$")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


# =========================================================================
# 7. MAIN
# =========================================================================
def main():
    t_start = time.time()
    print(f"=== Task 3.2 (QUICK={QUICK}) ===")
    print(f"    jax {jax.__version__} on {jax.devices()}")
    print(f"    MAX_FORCE={MAX_FORCE}  SIM_STEPS={SIM_STEPS}  "
          f"DELTA_TIME={DELTA_TIME}  T={CFG['T']}  T_verify={CFG['T_verify']}")

    rng = np.random.default_rng(CFG["seed"])

    # --- 1. GATING CHECK: JAX dynamics == numpy CartPole ---
    print("\n[gate] JAX vs numpy dynamics on "
          f"{CFG['n_gate']} random (state, action) pairs...")
    max_diff, passed = gating_check_dynamics(
        rng, n=CFG["n_gate"], tol=CFG["gate_tol"])
    status = "PASS" if passed else "FAIL"
    print(f"[gate] max abs diff = {max_diff:.3e}   tol = {CFG['gate_tol']:.0e}   "
          f"{status}")
    if not passed:
        print("[gate] FAIL -- aborting, every downstream result would be invalid.")
        sys.exit(1)

    # --- 2. Loss matches the built-in at sigma_l = 0.5 ---
    print("\n[loss-check] asserting our JAX loss reduces to cartpole._loss "
          "when sigma_l = 0.5 ...")
    assert loss_assertion_against_builtin(), "loss does not match builtin"
    print("[loss-check] PASS")

    # --- 3. Build the jitted loss/rollout for T and T_verify ---
    T        = CFG["T"]
    T_verify = CFG["T_verify"]
    dt       = CFG["dt"]
    X0_up    = np.asarray(CFG["X0_up"], dtype=np.float64)
    X0_down  = np.asarray(CFG["X0_down"], dtype=np.float64)

    loss_jit_opt, grad_jit_opt, loss_and_grad_opt = make_loss_fn(T, MAX_FORCE)
    rollout_jit_verify = make_rollout_fn(T_verify, MAX_FORCE)

    # Pre-jit warm-up (so timings make sense)
    _ = loss_and_grad_opt(np.zeros(4), X0_up, np.full(4, 0.5))
    _ = rollout_jit_verify(jnp.zeros(4), jnp.asarray(X0_up))

    # --- 4. Pre-optimisation loss scans ---
    # Take a sensible non-zero base policy so the scans pass through the basin.
    p_base = np.array([0.0, 0.0, 30.0, 5.0])
    sigma_scalar_init = 0.5
    sig_init = np.full(4, sigma_scalar_init, dtype=np.float64)
    print(f"\n[scans] 1-D and 2-D pre-opt loss scans from X0_up={X0_up.tolist()}, "
          f"sigma_l={sigma_scalar_init}, p_base={p_base.tolist()}")
    plot_loss_scans_1d(loss_jit_opt, X0_up, sig_init, p_base,
                       fig_path("loss_scans_1d"), n=60 if not QUICK else 30)
    plot_loss_contours_2d(loss_jit_opt, X0_up, sig_init, p_base,
                          fig_path("loss_contours_2d"),
                          n=35 if not QUICK else 18)

    # --- 5. SIGMA_L SELECTION  (metric sweep) ---
    print("\n=== sigma_l selection (metric sweep) ===")
    holdout_ICs = make_holdout_ICs(seed=CFG["seed"] + 7, n=CFG["n_holdout"])
    print(f"[sigma] {len(holdout_ICs)} held-out ICs, "
          f"theta in +/- {CFG['holdout_theta']}, "
          f"thetadot in +/- {CFG['holdout_thetadot']}")

    # Mix of small-, mid-, and high-gain hand seeds. Small-gain seeds give the
    # optimiser a non-saturated starting point if sigma_l is tight (the loss
    # is convex-ish near p=0). High-gain seeds anchor in the LQR-style basin.
    hand_seeds = [
        np.array([ 0.0,  0.0,  0.0,  0.0]),    # zero -- pure gradient probe
        np.array([-1.0, -2.0,  5.0,  1.0]),    # small balanced
        np.array([-1.0, -2.0, 20.0,  4.0]),    # mid LQR-style
        np.array([-2.0, -3.0, 40.0,  8.0]),    # high LQR-style
    ]

    sweep_data = []
    for k, sig in enumerate(CFG["sigma_grid"]):
        sig_vec = np.full(4, sig, dtype=np.float64)
        print(f"\n[sigma {k}] sigma_l = {sig} (scalar) -- optimising p from X0_up")
        res = optimise_policy(
            loss_and_grad_opt, X0_up, sig_vec,
            n_restarts=CFG["n_restarts"], seed=CFG["seed"] + 10 * k + 1,
            hand_seeds=hand_seeds, verbose=False)
        p_opt = res["p_best"]
        mets = evaluate_controller(p_opt, rollout_jit_verify, holdout_ICs,
                                    dt, CFG["theta_settled"])
        sweep_data.append({
            "sigma":             float(sig),
            "p_opt":             p_opt.tolist(),
            "loss_best":         res["loss_best"],
            "spread_rel":        res["spread_rel"],
            **mets,
        })
        print(f"   loss_best={res['loss_best']:.3e}  "
              f"stab_frac={mets['stab_frac']:.2f}  "
              f"mean_settle={mets['mean_settle_time']:.2f}s  "
              f"med_ss|theta|={mets['median_ss_theta']:.3f}  "
              f"med_effort={mets['median_effort']:.2f}")

    # Pick best scalar: require stabilisation, then a composite that trades
    # settling time, steady-state |theta|, and control effort -- per the
    # context md ("best trades settling time / steady-state error / effort").
    # Normalisation: 1s of settle ~ 0.05 rad ss|theta| ~ effort of 100.
    def composite(d) -> float:
        st = d["mean_settle_time"] if np.isfinite(d["mean_settle_time"]) else 5.0
        return st + 0.01 * d["median_effort"] + 20.0 * d["median_ss_theta"]

    def rank_key(d):
        return (-d["stab_frac"], composite(d))

    sweep_sorted = sorted(sweep_data, key=rank_key)
    best_scalar = sweep_sorted[0]
    sigma_scalar = float(best_scalar["sigma"])
    print(f"\n[sigma] best scalar sigma_l = {sigma_scalar}  "
          f"(stab_frac={best_scalar['stab_frac']:.2f}, "
          f"settle={best_scalar['mean_settle_time']:.2f}s, "
          f"ss|theta|={best_scalar['median_ss_theta']:.3f})")

    # Per-component refinement: tighten theta a bit, loosen x. Apply a floor
    # on sigma_theta so we don't drive it into the saturated-loss regime where
    # the optimiser cannot move.
    sigma_theta_floor = 0.25
    sigma_refined = np.array(
        [sigma_scalar * 1.5,                          # x      loosen
         sigma_scalar * 1.2,                          # x_dot
         max(sigma_theta_floor, sigma_scalar * 0.7),  # theta  tighten (floored)
         sigma_scalar * 1.0],                         # th_dot
        dtype=np.float64)
    print(f"\n[sigma] testing per-component refinement: {sigma_refined.tolist()}")
    res_ref = optimise_policy(
        loss_and_grad_opt, X0_up, sigma_refined,
        n_restarts=CFG["n_restarts"], seed=CFG["seed"] + 999,
        hand_seeds=hand_seeds, verbose=False)
    mets_ref = evaluate_controller(res_ref["p_best"], rollout_jit_verify,
                                   holdout_ICs, dt, CFG["theta_settled"])
    print(f"[sigma refine] stab_frac={mets_ref['stab_frac']:.2f}  "
          f"settle={mets_ref['mean_settle_time']:.2f}s  "
          f"ss|theta|={mets_ref['median_ss_theta']:.3f}  "
          f"med_effort={mets_ref['median_effort']:.2f}")

    # Compare: keep refinement only if it wins on the same composite key.
    refine_better = rank_key({"stab_frac":         mets_ref["stab_frac"],
                              "mean_settle_time":  mets_ref["mean_settle_time"],
                              "median_ss_theta":   mets_ref["median_ss_theta"],
                              "median_peak_theta": mets_ref["median_peak_theta"],
                              "median_effort":     mets_ref["median_effort"]}) \
                    < rank_key(best_scalar)
    if refine_better:
        sigma_l_final = sigma_refined
        sigma_choice = "per-component refinement"
        print(f"[sigma] kept refinement (better on key metrics)")
    else:
        sigma_l_final = np.full(4, sigma_scalar, dtype=np.float64)
        sigma_choice = f"scalar {sigma_scalar}"
        print(f"[sigma] kept scalar (refinement did not improve key metrics)")

    print(f"[sigma] FINAL sigma_l vector = {sigma_l_final.tolist()}")
    plot_sigma_sweep(sweep_data, sigma_scalar, fig_path("sigma_sweep"))

    # --- 6. EXPERIMENT A: optimise p from X0_up ---
    print("\n=== Experiment A: upright (X0 = [0,0,0.2,0]) ===")
    resA = optimise_policy(
        loss_and_grad_opt, X0_up, sigma_l_final,
        n_restarts=CFG["n_restarts"], seed=CFG["seed"] + 5000,
        hand_seeds=hand_seeds, verbose=True)
    p_upright = resA["p_best"]
    print(f"[A] best p = {p_upright.tolist()}  L={resA['loss_best']:.4e}  "
          f"restart spread={resA['spread_rel']:.2%}")

    # Verification rollout (T_verify steps)
    states_up_j, actions_up_j = rollout_jit_verify(
        jnp.asarray(p_upright), jnp.asarray(X0_up))
    states_up_full = np.vstack([X0_up[None, :], np.asarray(states_up_j)])
    actions_up = np.asarray(actions_up_j)
    forces_up  = MAX_FORCE * np.tanh(actions_up / MAX_FORCE)
    sat_frac   = float(np.mean(np.abs(forces_up) > 0.95 * MAX_FORCE))
    print(f"[A] final |theta| = {abs(states_up_full[-1, 2]):.4f}  "
          f"max|F|={np.max(np.abs(forces_up)):.2f}  "
          f"force-saturation frac (>0.95*F_max) = {sat_frac:.2%}")

    # Robustness on held-out ICs (also collect a few trajectories for overlay)
    mets_up = evaluate_controller(p_upright, rollout_jit_verify, holdout_ICs,
                                  dt, CFG["theta_settled"])
    n_overlay = min(6, len(holdout_ICs))
    overlay_traj = []
    for ic in holdout_ICs[:n_overlay]:
        s_j, _ = rollout_jit_verify(jnp.asarray(p_upright), jnp.asarray(ic))
        overlay_traj.append(np.vstack([ic[None, :], np.asarray(s_j)]))
    overlay_traj = np.stack(overlay_traj, axis=0)
    print(f"[A] held-out stabilised fraction = {mets_up['stab_frac']:.2%}  "
          f"mean settle = {mets_up['mean_settle_time']:.2f}s  "
          f"med ss|theta|={mets_up['median_ss_theta']:.3f}")

    plot_timeevo(states_up_full, actions_up, dt,
                 f"Experiment A — upright stabilisation under optimised "
                 f"$p$={[f'{v:.2f}' for v in p_upright]}",
                 fig_path("timeevo_upright"),
                 holdout_states=overlay_traj)
    plot_phase(states_up_full,
               "Experiment A — phase portrait $\\theta$ vs $\\dot\\theta$",
               fig_path("phase_upright"))

    # --- 7. EXPERIMENT B: optimise p from X0_down ---
    print("\n=== Experiment B: downward (X0 = [0,0,pi,0]) ===")
    resB = optimise_policy(
        loss_and_grad_opt, X0_down, sigma_l_final,
        n_restarts=CFG["n_restarts"], seed=CFG["seed"] + 6000,
        hand_seeds=hand_seeds, verbose=True)
    p_downward = resB["p_best"]
    print(f"[B] best p = {p_downward.tolist()}  L={resB['loss_best']:.4e}")

    # Evidence: ||grad L|| at the initial X0 (i.e. at p=0) — the L-BFGS-B
    # starting condition's effective signal. Compare upright vs downward.
    grad_up   = np.asarray(grad_jit_opt(jnp.zeros(4),
                                        jnp.asarray(X0_up),
                                        jnp.asarray(sigma_l_final)))
    grad_down = np.asarray(grad_jit_opt(jnp.zeros(4),
                                        jnp.asarray(X0_down),
                                        jnp.asarray(sigma_l_final)))
    gnorm_up   = float(np.linalg.norm(grad_up))
    gnorm_down = float(np.linalg.norm(grad_down))
    print(f"[B] ||grad L(p=0)||  upright={gnorm_up:.3e}   "
          f"downward={gnorm_down:.3e}   ratio = {gnorm_down/max(gnorm_up,1e-30):.3e}")

    # Verification rollout for the downward case
    states_dn_j, actions_dn_j = rollout_jit_verify(
        jnp.asarray(p_downward), jnp.asarray(X0_down))
    states_dn_full = np.vstack([X0_down[None, :], np.asarray(states_dn_j)])
    actions_dn = np.asarray(actions_dn_j)
    print(f"[B] final |theta| = {abs(states_dn_full[-1, 2]):.4f}  "
          f"(target 0; expect ~pi, i.e. still hanging down)")

    # max_force rescue check: re-optimise with larger max_force; should STILL fail
    rescue = []
    for mf in CFG["downward_max_force_check"]:
        print(f"[B max_force={mf}] rebuilding loss/rollout and re-optimising...")
        loss_jit_mf, grad_jit_mf, loss_and_grad_mf = make_loss_fn(T, mf)
        rollout_jit_mf = make_rollout_fn(T_verify, mf)
        _ = loss_and_grad_mf(np.zeros(4), X0_down, sigma_l_final)
        res_mf = optimise_policy(
            loss_and_grad_mf, X0_down, sigma_l_final,
            n_restarts=CFG["n_restarts"], seed=CFG["seed"] + 7000 + int(mf),
            hand_seeds=hand_seeds, verbose=False)
        s_j, a_j = rollout_jit_mf(jnp.asarray(res_mf["p_best"]),
                                  jnp.asarray(X0_down))
        s_full = np.vstack([X0_down[None, :], np.asarray(s_j)])
        final_abs_th = float(abs(s_full[-1, 2]))
        rescued = bool(final_abs_th < CFG["theta_settled"])
        rescue.append({
            "max_force":         float(mf),
            "p_opt":             res_mf["p_best"].tolist(),
            "loss_best":         float(res_mf["loss_best"]),
            "final_abs_theta":   final_abs_th,
            "rescued":           rescued,
        })
        print(f"[B max_force={mf}] final |theta| = {final_abs_th:.4f}   "
              f"rescued = {rescued}")

    plot_timeevo(states_dn_full, actions_dn, dt,
                 f"Experiment B — downward start under optimised "
                 f"$p$={[f'{v:.2f}' for v in p_downward]} (linear policy "
                 "cannot swing up)",
                 fig_path("timeevo_downward"))

    # --- 8. JSON RESULTS ---
    print("\n=== JSON ===")
    out = {
        "tag":                 FIG_PREFIX,
        "quick":               QUICK,
        "elapsed_total_s":     time.time() - t_start,
        "cfg":                 CFG,
        "physics": {
            "max_force":   MAX_FORCE,
            "sim_steps":   SIM_STEPS,
            "delta_time":  DELTA_TIME,
            "gravity":     GRAVITY,
            "cart_mass":   CART_MASS,
            "pole_mass":   POLE_MASS,
            "pole_length": POLE_LENGTH,
            "mu_c":        MU_C,
            "mu_p":        MU_P,
        },
        "gating": {
            "n":         CFG["n_gate"],
            "tol":       CFG["gate_tol"],
            "max_diff":  max_diff,
            "passed":    passed,
        },
        "loss_match_builtin_at_sigma_0p5": True,
        "T":                   T,
        "T_verify":            T_verify,
        "dt":                  dt,
        "X0_up":               X0_up.tolist(),
        "X0_down":             X0_down.tolist(),
        "sigma_l_final":       sigma_l_final.tolist(),
        "sigma_choice":        sigma_choice,
        "sigma_sweep":         sweep_data,
        "sigma_refinement": {
            "sigma_l":         sigma_refined.tolist(),
            "p_opt":           res_ref["p_best"].tolist(),
            "loss_best":       float(res_ref["loss_best"]),
            "mets":            mets_ref,
            "kept":            bool(refine_better),
        },
        "experiment_A_upright": {
            "p_opt":              p_upright.tolist(),
            "loss_best":          float(resA["loss_best"]),
            "restart_loss_min":   resA["loss_min"],
            "restart_loss_max":   resA["loss_max"],
            "restart_spread_rel": resA["spread_rel"],
            "final_abs_theta":    float(abs(states_up_full[-1, 2])),
            "max_abs_force":      float(np.max(np.abs(forces_up))),
            "force_saturation_frac": sat_frac,
            "holdout_metrics":    mets_up,
            "per_ic_final_loss":  None,  # set below
        },
        "experiment_B_downward": {
            "p_opt":             p_downward.tolist(),
            "loss_best":         float(resB["loss_best"]),
            "restart_loss_min":  resB["loss_min"],
            "restart_loss_max":  resB["loss_max"],
            "restart_spread_rel": resB["spread_rel"],
            "final_abs_theta":   float(abs(states_dn_full[-1, 2])),
            "grad_norm_at_p0_upright":  gnorm_up,
            "grad_norm_at_p0_downward": gnorm_down,
            "grad_norm_ratio_down_over_up": gnorm_down / max(gnorm_up, 1e-30),
            "max_force_rescue_check": rescue,
        },
    }

    # per-IC final trajectory loss for experiment A
    p_j = jnp.asarray(p_upright)
    per_ic_L = []
    for ic in holdout_ICs:
        L_ic = float(loss_jit_opt(p_j, jnp.asarray(ic),
                                  jnp.asarray(sigma_l_final)))
        per_ic_L.append(L_ic)
    out["experiment_A_upright"]["per_ic_final_loss"] = per_ic_L
    out["experiment_A_upright"]["holdout_ICs"]       = holdout_ICs.tolist()

    results_path = FIG_DIR / f"{FIG_PREFIX}_results.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[json] saved -> {results_path.name}")

    # --- 9. CONSOLE SUMMARY ---
    print("\n" + "=" * 72)
    print("TASK 3.2 SUMMARY")
    print("=" * 72)
    print(f"Dynamics match (JAX vs numpy): max diff = {max_diff:.2e}  "
          f"(< {CFG['gate_tol']:.0e})  PASS")
    print(f"Final sigma_l vector: {sigma_l_final.tolist()}  ({sigma_choice})")
    print(f"\nExperiment A (upright):")
    print(f"  p_opt   = {[f'{v:+.3f}' for v in p_upright]}")
    print(f"  L*      = {resA['loss_best']:.4e}   "
          f"final |theta| = {abs(states_up_full[-1, 2]):.4f}")
    print(f"  held-out stab_frac = {mets_up['stab_frac']:.2%}   "
          f"mean settle = {mets_up['mean_settle_time']:.2f}s")
    print(f"  max|F| = {np.max(np.abs(forces_up)):.2f}  "
          f"saturation_frac = {sat_frac:.2%}  "
          f"({'no saturation' if sat_frac < 0.01 else 'saturating'})")
    print(f"\nExperiment B (downward):")
    print(f"  p_opt   = {[f'{v:+.3f}' for v in p_downward]}")
    print(f"  L*      = {resB['loss_best']:.4e}   "
          f"final |theta| = {abs(states_dn_full[-1, 2]):.4f} "
          f"(target 0; expect ~{np.pi:.3f})")
    print(f"  ||grad L(p=0)|| upright   = {gnorm_up:.3e}")
    print(f"  ||grad L(p=0)|| downward  = {gnorm_down:.3e}   "
          f"({'flat -- L saturated, expected' if gnorm_down < gnorm_up * 0.1 else 'NOT flat -- unexpected'})")
    for r in rescue:
        print(f"  max_force={r['max_force']:>5.0f}: final |theta|="
              f"{r['final_abs_theta']:.3f}  rescued={r['rescued']}  "
              f"({'still failed -- structural' if not r['rescued'] else 'rescued?!'})")
    print(f"\nElapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
