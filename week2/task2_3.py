"""
SF3 Machine Learning -- Week 2, Task 2.3
========================================

Replace the raw angle theta with the pair (sin theta, cos theta) as input
features, drop the periodic sin^2((theta - theta')/2) kernel substitution,
and remove angle remapping inside iterated rollouts. The new feature space
is 5-D: [x, x_dot, sin(theta), cos(theta), theta_dot]. Targets remain the
raw 4-D state delta, with the angle component still remapped into (-pi, pi]
when computing dY at data-gathering time.

Pipeline:
  1. Gather/cache 2000 train + 500 val (X, dX) one-step pairs (force=0).
  2. Linear baseline dX = C * features(X), fit by np.linalg.lstsq.
  3. Nonlinear sparse kernel regression on 5-D features (M = 200 centres),
     with one length scale per feature dim and per output dim (24 params).
     Use jnp.linalg.solve for (K_MN K_NM + lam K_MM) alpha = K_MN y.
  4. Per-output L-BFGS-B in JAX, single start, initialised from Task 2.2's
     optimised hyperparameters (with the theta log-sigma duplicated into
     both sin/cos slots). Sanity check warns if any dim's MSE went up.
  5. Iterated rollouts (no internal remap; remap only when plotting theta).
  6. Six figures, console summary, hyperparams.json, results.json.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, value_and_grad

from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week2"))
from cartpole import CartPole, _remap_angle                          # noqa: E402
from task2_1 import _angle_wrap, _estimate_period                    # noqa: E402
import task2_2 as t22                                                # noqa: E402


# =========================================================================
# CONFIG
# =========================================================================
CFG = {
    "N_train":           2000,
    "N_val":             500,
    "M":                 200,

    "lambda_init":       1e-4,
    "log_lambda_bounds": (-18.0, -2.0),
    "log_sigma_bounds":  (-6.0,   5.0),
    "maxiter":           200,
    "ftol":              1e-10,
    "gtol":              1e-8,

    "seed":              0,
    "rollout_steps":     100,
    "dt":                0.1,
    "horizon_threshold": 0.5,

    "t2_2_hp_path": "figures/task2_2_delta_NM_N2000_M200_hyperparams.json",
}

SAMPLE_RANGES = {
    "x":         (-5.0,    5.0),
    "x_dot":     (-10.0,  10.0),
    "theta":     (-np.pi,  np.pi),
    "theta_dot": (-15.0,  15.0),
}
FEATURE_LABELS = [r"$x$", r"$\dot x$", r"$\sin\theta$", r"$\cos\theta$", r"$\dot\theta$"]
DELTA_LABELS   = [r"$\Delta x$", r"$\Delta \dot x$",
                  r"$\Delta \theta$", r"$\Delta \dot\theta$"]
STATE_LABELS   = [r"$x$", r"$\dot x$", r"$\theta$", r"$\dot\theta$"]
DIM_NAMES      = ["x", "xdot", "theta", "thetadot"]

FIG_DIR   = ROOT / "figures"
CACHE_DIR = ROOT / "week2"
FIG_DIR.mkdir(exist_ok=True)

FIG_PREFIX = "task2_3"


def fig_path(name: str) -> Path:
    return FIG_DIR / f"{FIG_PREFIX}_{name}.png"


def data_cache_path() -> Path:
    return CACHE_DIR / (
        f"{FIG_PREFIX}_data_N{CFG['N_train']}_V{CFG['N_val']}_seed{CFG['seed']}.npz"
    )


# =========================================================================
# 1. DATA GATHERING
# =========================================================================
def sample_states(n: int, rng: np.random.Generator) -> np.ndarray:
    keys = ("x", "x_dot", "theta", "theta_dot")
    lo = np.array([SAMPLE_RANGES[k][0] for k in keys])
    hi = np.array([SAMPLE_RANGES[k][1] for k in keys])
    return rng.uniform(lo, hi, size=(n, 4))


def gather_dataset(n: int, rng: np.random.Generator):
    """n pairs (X, dX) with one-step performAction(0.0); dX[:, 2] is remapped
    into (-pi, pi] so the target is the actual short-time angle change."""
    sim = CartPole(visual=False)
    X = sample_states(n, rng)
    Y = np.zeros_like(X)
    for i in range(n):
        sim.setState(X[i].tolist())
        sim.performAction(0.0)
        dY = sim.getState() - X[i]
        dY[2] = _remap_angle(dY[2])
        Y[i] = dY
    return X, Y


def load_or_gather_data() -> dict:
    cache = data_cache_path()
    n_tr, n_va, seed = CFG["N_train"], CFG["N_val"], CFG["seed"]

    if cache.exists():
        print(f"[data] reusing {cache.name}")
        d = np.load(cache)
        X_tr, Y_tr = d["X_tr"], d["Y_tr"]
        X_va, Y_va = d["X_va"], d["Y_va"]
        assert X_tr.shape[0] == n_tr and X_va.shape[0] == n_va
    else:
        print(f"[data] generating {n_tr + n_va} samples (seed={seed})...")
        t0 = time.time()
        rng = np.random.default_rng(seed)
        X_all, Y_all = gather_dataset(n_tr + n_va, rng)
        X_tr, Y_tr = X_all[:n_tr], Y_all[:n_tr]
        X_va, Y_va = X_all[n_tr:], Y_all[n_tr:]
        np.savez(cache, X_tr=X_tr, Y_tr=Y_tr, X_va=X_va, Y_va=Y_va)
        print(f"[data] cached -> {cache.name}  ({time.time()-t0:.1f}s)")

    rng_b = np.random.default_rng(seed + 1)
    M = min(CFG["M"], n_tr)
    basis_idx = rng_b.choice(n_tr, size=M, replace=False)

    return {
        "X_tr": X_tr, "Y_tr": Y_tr,
        "X_va": X_va, "Y_va": Y_va,
        "X_basis": X_tr[basis_idx].copy(),
    }


# =========================================================================
# 2. FEATURE TRANSFORM
# =========================================================================
def to_features_np(X: np.ndarray) -> np.ndarray:
    """(..., 4) -> (..., 5):  [x, x_dot, sin theta, cos theta, theta_dot]."""
    X = np.atleast_2d(X)
    return np.stack([X[..., 0], X[..., 1],
                     np.sin(X[..., 2]), np.cos(X[..., 2]),
                     X[..., 3]], axis=-1)


def to_features_jax(X):
    return jnp.stack([X[..., 0], X[..., 1],
                      jnp.sin(X[..., 2]), jnp.cos(X[..., 2]),
                      X[..., 3]], axis=-1)


# =========================================================================
# 3. KERNEL + FIT (JAX, plain anisotropic Gaussian on 5-D features)
# =========================================================================
def kernel_matrix(F1, F2, log_sigmas):
    """K[i,j] = exp(-sum_k (F1[i,k] - F2[j,k])^2 / (2 sigma_k^2))."""
    sigmas = jnp.exp(log_sigmas)
    inv2s2 = 1.0 / (2.0 * sigmas ** 2)
    diff = F1[:, None, :] - F2[None, :, :]
    return jnp.exp(-jnp.sum(diff ** 2 * inv2s2, axis=-1))


def fit_alpha_NM(F_tr, y_col, F_basis, log_lambda, log_sigmas):
    """Sparse subset, eq. 15: alpha = (K_MN K_NM + lam K_MM)^{-1} K_MN y."""
    K_NM = kernel_matrix(F_tr, F_basis, log_sigmas)
    K_MM = kernel_matrix(F_basis, F_basis, log_sigmas)
    K_MN = K_NM.T
    lam = jnp.exp(log_lambda)
    A = K_MN @ K_NM + lam * K_MM
    return jnp.linalg.solve(A, K_MN @ y_col)


def kernel_predict(F_test, F_centres, alpha, log_sigmas):
    return kernel_matrix(F_test, F_centres, log_sigmas) @ alpha


# Numpy mirrors for plotting / rollouts (don't need autodiff).
def kernel_matrix_np(F1, F2, log_sigmas):
    sigmas = np.exp(log_sigmas)
    inv2s2 = 1.0 / (2.0 * sigmas ** 2)
    diff = F1[:, None, :] - F2[None, :, :]
    return np.exp(-np.sum(diff ** 2 * inv2s2, axis=-1))


def fit_alpha_NM_np(F_tr, y_col, F_basis, log_lambda, log_sigmas):
    K_NM = kernel_matrix_np(F_tr, F_basis, log_sigmas)
    K_MM = kernel_matrix_np(F_basis, F_basis, log_sigmas)
    K_MN = K_NM.T
    lam = float(np.exp(log_lambda))
    A = K_MN @ K_NM + lam * K_MM
    b = K_MN @ y_col
    alpha, *_ = np.linalg.lstsq(A, b, rcond=None)
    return alpha


@dataclass
class KernelEnsemble:
    label: str
    F_basis: np.ndarray              # (M, 5)
    alphas: np.ndarray               # (M, 4)
    log_sigmas_per_out: np.ndarray   # (4, 5)
    log_lambdas: np.ndarray          # (4,)

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = to_features_np(X)
        n = F.shape[0]
        out = np.zeros((n, 4))
        for j in range(4):
            K = kernel_matrix_np(F, self.F_basis, self.log_sigmas_per_out[j])
            out[:, j] = K @ self.alphas[:, j]
        return out


def fit_ensemble(F_tr, Y_tr, F_basis, log_sigmas_per_out, log_lambdas, label):
    M = F_basis.shape[0]
    alphas = np.zeros((M, 4))
    for j in range(4):
        alphas[:, j] = fit_alpha_NM_np(
            F_tr, Y_tr[:, j], F_basis,
            float(log_lambdas[j]), log_sigmas_per_out[j],
        )
    return KernelEnsemble(
        label=label, F_basis=F_basis, alphas=alphas,
        log_sigmas_per_out=log_sigmas_per_out.copy(),
        log_lambdas=log_lambdas.copy(),
    )


# =========================================================================
# 4. LINEAR BASELINE on 5-D features
# =========================================================================
def fit_linear_features(F, Y):
    """Solve Y = F C^T by lstsq. Returns C of shape (4, 5)."""
    CT, *_ = np.linalg.lstsq(F, Y, rcond=None)
    return CT.T


def mse_per_dim(Y_true, Y_pred):
    return ((Y_true - Y_pred) ** 2).mean(axis=0)


# =========================================================================
# 5. JAX HYPERPARAMETER OPTIMISATION
# =========================================================================
def make_val_mse_fn(F_tr_j, Y_tr_col, F_va_j, Y_va_col, F_basis_j):
    def val_mse(log_hp):
        log_lambda = log_hp[0]
        log_sigmas = log_hp[1:]
        alpha = fit_alpha_NM(F_tr_j, Y_tr_col, F_basis_j,
                             log_lambda, log_sigmas)
        preds = kernel_predict(F_va_j, F_basis_j, alpha, log_sigmas)
        return jnp.mean((preds - Y_va_col) ** 2)
    return jit(value_and_grad(val_mse))


def initial_hyperparams_from_t22(F_tr) -> tuple[np.ndarray, str]:
    """Build the 6-vector log-hp init for each of 4 output dims, from Task 2.2.

    Returns (log_hp_init_per_dim, source_tag) with shape (4, 6).
    Order of the 6-vector: [log_lambda, log_sigma_{x, xdot, sin, cos, thetadot}].

    Task 2.2's log_hp_opt is a 5-vector: [log_lambda, log_sigma_{x, xdot, theta, thetadot}].
    The theta log-sigma is duplicated into both sin and cos slots; everything
    else is copied directly.

    Falls back to a std-based init if the Task 2.2 file is missing/malformed.
    """
    path = ROOT / CFG["t2_2_hp_path"]
    if path.exists():
        try:
            with open(path) as f:
                t22 = json.load(f)
            results = t22["results"]
            assert len(results) == 4
            init = np.zeros((4, 6), dtype=np.float64)
            for d in range(4):
                hp22 = results[d]["log_hp_opt"]   # 5 floats
                assert len(hp22) == 5
                init[d] = [hp22[0], hp22[1], hp22[2], hp22[3], hp22[3], hp22[4]]
            return init, f"t2.2 ({path.name})"
        except (KeyError, AssertionError, ValueError) as e:
            print(f"[init] WARNING: failed to parse {path.name} ({e}); "
                  f"falling back to std-based init.")
    else:
        print(f"[init] WARNING: {path.name} not found; "
              f"falling back to std-based init.")

    # Fallback: log_sigma_k = log(std of feature k); log_lambda = log(1e-4).
    sig0 = F_tr.std(axis=0)                          # (5,)
    sig0 = np.maximum(sig0, 1e-3)
    log_lambda_init = float(np.log(CFG["lambda_init"]))
    init_one = np.concatenate([[log_lambda_init], np.log(sig0)])
    init = np.tile(init_one, (4, 1))                 # same init for all dims
    return init, "std-based fallback"


def optimise_one_dim(val_and_grad_fn, log_hp_init, bounds):
    trace = []

    def f_and_g(x_np):
        v, g = val_and_grad_fn(jnp.asarray(x_np))
        v_f = float(v)
        trace.append(v_f)
        return v_f, np.asarray(g, dtype=np.float64)

    res = minimize(
        f_and_g,
        np.asarray(log_hp_init, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": CFG["maxiter"],
                 "ftol":    CFG["ftol"],
                 "gtol":    CFG["gtol"]},
    )
    return res, trace


def run_optimisation(data):
    """Single L-BFGS-B per output dim, init from Task 2.2 hyperparameters."""
    F_tr = to_features_np(data["X_tr"])
    F_va = to_features_np(data["X_va"])
    F_basis = to_features_np(data["X_basis"])
    Y_tr, Y_va = data["Y_tr"], data["Y_va"]

    init_per_dim, init_source = initial_hyperparams_from_t22(F_tr)
    print(f"[init] log_hp_init source: {init_source}")

    log_lo,  log_hi  = CFG["log_lambda_bounds"]
    log_slo, log_shi = CFG["log_sigma_bounds"]
    bounds = [(log_lo, log_hi)] + [(log_slo, log_shi)] * 5    # 6 entries

    F_tr_j    = jnp.asarray(F_tr)
    F_va_j    = jnp.asarray(F_va)
    F_basis_j = jnp.asarray(F_basis)

    results = []
    for j in range(4):
        Y_tr_col = jnp.asarray(Y_tr[:, j])
        Y_va_col = jnp.asarray(Y_va[:, j])

        vg = make_val_mse_fn(F_tr_j, Y_tr_col, F_va_j, Y_va_col, F_basis_j)

        log_hp0 = init_per_dim[j]
        t0 = time.time()
        v0, g0 = vg(jnp.asarray(log_hp0))
        v0 = float(v0)
        _ = float(jnp.linalg.norm(g0))   # block on device
        print(f"[opt dim {j} {DIM_NAMES[j]:>8s}] compile+init {time.time()-t0:.1f}s   "
              f"init val MSE: {v0:.4e}")

        t0 = time.time()
        res, trace = optimise_one_dim(vg, log_hp0, bounds)
        dt = time.time() - t0

        # Re-evaluate at the returned optimum for a clean gradient norm.
        _, g_final = vg(jnp.asarray(res.x))
        g_norm = float(jnp.linalg.norm(g_final))
        v_opt = float(res.fun)
        ratio = v0 / max(v_opt, 1e-30)

        print(f"                     opt done in {dt:.1f}s   iters={res.nit:3d}   "
              f"final val MSE: {v_opt:.4e}   grad norm: {g_norm:.2e}   "
              f"(factor {ratio:.2f}x improvement)")
        sig_opt = res.x[1:]
        print(f"                     log_hp_opt: lam={res.x[0]:.3f}  "
              f"sig=[{sig_opt[0]:.3f}, {sig_opt[1]:.3f}, {sig_opt[2]:.3f}, "
              f"{sig_opt[3]:.3f}, {sig_opt[4]:.3f}]")

        if v_opt > v0:
            print(f"*** WARNING: dim {j} ({DIM_NAMES[j]}) val MSE went UP: "
                  f"init {v0:.4e} -> opt {v_opt:.4e}. "
                  f"Optimisation failed; a restart strategy may be needed. ***")

        results.append({
            "dim":          j,
            "dim_name":     DIM_NAMES[j],
            "log_hp_init":  log_hp0.tolist(),
            "log_hp_opt":   res.x.tolist(),
            "val_mse_init": v0,
            "val_mse_opt":  v_opt,
            "iters":        int(res.nit),
            "n_evals":      len(trace),
            "success":      bool(res.success),
            "message":      str(res.message),
            "trace":        trace,
            "elapsed_s":    dt,
            "improvement_factor": ratio,
        })

    return results, init_source


# =========================================================================
# 6. ROLLOUTS
# =========================================================================
def true_rollout(X0, n_steps):
    sim = CartPole(visual=False)
    sim.setState(X0.tolist())
    traj = np.zeros((n_steps + 1, 4))
    traj[0] = sim.getState()
    for t in range(n_steps):
        sim.performAction(0.0)
        traj[t + 1] = sim.getState()
    return traj


def model_rollout(ensemble: KernelEnsemble, X0: np.ndarray, n_steps: int):
    """Iterate X_{n+1} = X_n + f(X_n). NO theta remapping inside the loop --
    sin/cos in the feature transform absorb periodicity, and remapping would
    create a step in the (theta_dot, ...) neighbourhood the kernel hasn't
    seen during training. For visualisation we remap after the fact."""
    traj = np.zeros((n_steps + 1, 4))
    X = X0.copy()
    traj[0] = X
    for t in range(n_steps):
        dX = ensemble.predict(X[None, :])[0]
        X = X + dX
        traj[t + 1] = X
    return traj


def rollout_horizon(model: KernelEnsemble, X0, sigmas_state, n_steps, dt,
                    threshold=0.5):
    """Normalised per-step state error; break time and oscillation cycles."""
    traj_t = true_rollout(X0, n_steps)
    traj_m = model_rollout(model, X0, n_steps)
    diff = traj_m - traj_t
    diff[:, 2] = _angle_wrap(diff[:, 2])
    err = np.sqrt(np.sum((diff / sigmas_state) ** 2, axis=1))
    above = err > threshold
    if above.any():
        step_break = int(np.argmax(above))
        t_break = step_break * dt
    else:
        step_break = n_steps + 1
        t_break = float("inf")
    period = _estimate_period(traj_t, dt)
    if np.isfinite(t_break) and np.isfinite(period) and period > 0:
        cycles = t_break / period
    else:
        cycles = None
    return {
        "traj_true":  traj_t,
        "traj_model": traj_m,
        "err":        err,
        "step_break": step_break,
        "t_break":    t_break,
        "period_s":   period if np.isfinite(period) else None,
        "cycles":     cycles,
    }


# =========================================================================
# 6b. RECONSTRUCT TASK 2.2 MODELS (for the rollout-horizon overlay)
# =========================================================================
def build_t22_models():
    """Rebuild Task 2.2's 't2.1' and optimised ('opt') KernelEnsembles so their
    iterated rollouts can be overlaid against the Task 2.3 sin/cos model.

    Returns (t21_model, t22_opt_model, t21_lambda); any may be None if the
    Task 2.2 hyperparameter cache (or the Task 2.1 lambda) is unavailable.

    The returned objects are task2_2.KernelEnsemble instances and MUST be rolled
    out with task2_2.model_rollout (which remaps theta in-loop) -- unlike the
    Task 2.3 model, whose sin/cos features absorb periodicity without remap."""
    hp_path = t22.hp_cache_path()
    if not hp_path.exists():
        print(f"[t2.2] {hp_path.name} not found -- skipping t2.1/t2.2 overlay")
        return None, None, None
    data22 = t22.load_or_gather_data()
    lin_C = t22.fit_linear(data22["X_tr"], data22["Y_tr"])
    with open(hp_path) as f:
        opt_results = json.load(f)["results"]
    _, t21_model, t22_opt_model, t21_lambda = t22.build_ensembles(
        data22, opt_results, lin_C)
    return t21_model, t22_opt_model, t21_lambda


# =========================================================================
# 7. PLOTS
# =========================================================================
def plot_pred_vs_true(Y_true, Y_pred, title, savepath):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for k in range(4):
        ax = axes[k]
        ax.scatter(Y_true[:, k], Y_pred[:, k], s=5, alpha=0.4)
        lo = float(min(Y_true[:, k].min(), Y_pred[:, k].min()))
        hi = float(max(Y_true[:, k].max(), Y_pred[:, k].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        mse_k = float(np.mean((Y_true[:, k] - Y_pred[:, k]) ** 2))
        ax.set_title(f"{DELTA_LABELS[k]}   MSE={mse_k:.2e}", fontsize=10)
        ax.set_xlabel("true")
        if k == 0:
            ax.set_ylabel("predicted")
        ax.set_aspect("equal", "box")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_theta_scan(model: KernelEnsemble, savepath,
                    fixed=(0.5, 1.0, 2.0), n=400):
    """Sweep theta in [-pi, pi] with the other state vars fixed; compare true
    one-step dX against the model. The point is to show smooth behaviour at
    theta = +/- pi (which the old periodic-kernel formulation could distort).
    """
    x0, xdot0, thdot0 = fixed
    thetas = np.linspace(-np.pi, np.pi, n)
    X_scan = np.stack([np.full(n, x0), np.full(n, xdot0),
                       thetas, np.full(n, thdot0)], axis=1)

    sim = CartPole(visual=False)
    Y_true = np.zeros_like(X_scan)
    for i in range(n):
        sim.setState(X_scan[i].tolist())
        sim.performAction(0.0)
        dY = sim.getState() - X_scan[i]
        dY[2] = _remap_angle(dY[2])
        Y_true[i] = dY

    Y_pred = model.predict(X_scan)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for k in range(4):
        ax = axes[k]
        ax.plot(thetas, Y_true[:, k], "k-", lw=1.5, label="true")
        ax.plot(thetas, Y_pred[:, k], "r--", lw=1.4, label="model")
        ax.axvline(-np.pi, color="gray", lw=0.5, ls=":")
        ax.axvline( np.pi, color="gray", lw=0.5, ls=":")
        ax.set_xlabel(r"$\theta$ (rad)")
        ax.set_title(DELTA_LABELS[k])
        if k == 0:
            ax.set_ylabel(r"one-step $\Delta$")
            ax.legend(loc="best", fontsize=9)
    fig.suptitle(
        rf"$\theta$ scan at $(x, \dot x, \dot\theta) = ({x0}, {xdot0}, {thdot0})$"
        "  -- smooth crossing at $\\pm\\pi$ is the qualitative goal of Task 2.3"
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_rollout(roll, title, savepath, dt):
    traj_t, traj_m = roll["traj_true"], roll["traj_model"]
    # Remap theta to (-pi, pi] for both, just for clearer visualisation.
    th_t = np.array([_remap_angle(a) for a in traj_t[:, 2]])
    th_m = np.array([_remap_angle(a) for a in traj_m[:, 2]])
    t = np.arange(traj_t.shape[0]) * dt

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    series_t = [traj_t[:, 0], traj_t[:, 1], th_t, traj_t[:, 3]]
    series_m = [traj_m[:, 0], traj_m[:, 1], th_m, traj_m[:, 3]]
    for k in range(4):
        ax = axes[k]
        ax.plot(t, series_t[k], "k-",  lw=1.5, label="true")
        ax.plot(t, series_m[k], "r--", lw=1.4, label="model")
        ax.set_ylabel(STATE_LABELS[k])
        if k == 0:
            ax.legend(loc="best", fontsize=9)
    if np.isfinite(roll["t_break"]):
        for ax in axes:
            ax.axvline(roll["t_break"], color="tab:orange", ls=":", lw=1)
        axes[0].text(roll["t_break"], axes[0].get_ylim()[1],
                     f"  T_break = {roll['t_break']:.2g}s",
                     va="top", fontsize=9, color="tab:orange")
    axes[-1].set_xlabel("time (s)")
    cyc = roll["cycles"]
    cyc_str = "n/a" if cyc is None else f"{cyc:.1f}"
    fig.suptitle(f"{title}   (T_break={roll['t_break']:.2g}s, cycles~{cyc_str})")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_mse_comparison(mse_lin, mse_nl_init, mse_nl_opt, savepath):
    """Per-output-dim val MSE: linear vs nonlinear init vs nonlinear optimised."""
    x = np.arange(4)
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, mse_lin,     w, label="linear (5-D feats)",   color="tab:gray")
    ax.bar(x,     mse_nl_init, w, label="nonlinear @ init",     color="tab:blue")
    ax.bar(x + w, mse_nl_opt,  w, label="nonlinear @ optimised", color="tab:red")
    ax.set_yscale("log")
    ax.set_xticks(x, DELTA_LABELS)
    ax.set_ylabel("validation MSE")
    ax.set_title("Validation MSE per output dim (lower is better)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_rollout_horizon_comparison(model_entries, ICs, ic_labels,
                                    sigmas_state, n_steps, dt, threshold,
                                    savepath):
    """Overlay normalised rollout error vs time for several models on a common
    set of ICs -- the Task 2.2 rollout-horizon figure, plus the Task 2.3 model.

    model_entries: list of dicts {label, model, rollout_fn, colour}. Each model
    is rolled out with its own rollout_fn (Task 2.2 remaps theta in-loop, Task
    2.3 does not), then compared against the shared true rollout. Returns
    per-model lists of {IC, T_break_s, period_s, cycles_at_break}."""
    n_ic = len(ICs)
    n_cols = min(n_ic, 3)
    n_rows = int(np.ceil(n_ic / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.0 * n_cols, 4.0 * n_rows),
                             sharex=True, sharey=True)
    axes_flat = axes.ravel() if n_ic > 1 else [axes]

    summaries = {e["label"]: [] for e in model_entries}
    tarr = np.arange(n_steps + 1) * dt

    for r, X0 in enumerate(ICs):
        ax = axes_flat[r]
        tr_true = true_rollout(X0, n_steps)
        period_s = _estimate_period(tr_true, dt)
        for e in model_entries:
            model = e["model"]
            if model is None:
                continue
            tr_m = e["rollout_fn"](model, X0, n_steps)
            diff = tr_m - tr_true
            diff[:, 2] = _angle_wrap(diff[:, 2])
            err = np.sqrt(np.sum((diff / sigmas_state) ** 2, axis=1))
            above = err > threshold
            t_break = (float(tarr[int(np.argmax(above))])
                       if above.any() else float("inf"))
            if np.isfinite(t_break) and np.isfinite(period_s) and period_s > 0:
                cycles = t_break / period_s
            else:
                cycles = None
            t_str = (f"T={t_break:.1f}s" if np.isfinite(t_break)
                     else f"T>{tarr[-1]:.0f}s")
            ax.semilogy(tarr, np.maximum(err, 1e-6), "-",
                        color=e["colour"], lw=1.4,
                        label=f"{e['label']} [{t_str}]")
            if np.isfinite(t_break):
                ax.axvline(t_break, color=e["colour"], ls=":", lw=1, alpha=0.6)
            summaries[e["label"]].append({
                "IC":              ic_labels[r].splitlines()[0],
                "T_break_s":       t_break if np.isfinite(t_break) else None,
                "period_s":        period_s if np.isfinite(period_s) else None,
                "cycles_at_break": cycles,
            })
        ax.axhline(threshold, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_title(ic_labels[r], fontsize=9)
        ax.set_xlabel("time / s")
        ax.set_ylabel(r"$\|(X_m - X_t)/\sigma\|_2$")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=7, loc="lower right")

    for r in range(n_ic, n_rows * n_cols):
        axes_flat[r].axis("off")

    fig.suptitle(f"Rollout horizon comparison (threshold={threshold}): "
                 f"t2.1 vs t2.2-opt vs t2.3")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)
    return summaries


# =========================================================================
# 8. MAIN
# =========================================================================
def main():
    np.random.seed(0)
    t_start = time.time()

    # ---- 1. DATA ----
    data = load_or_gather_data()
    X_tr, Y_tr = data["X_tr"], data["Y_tr"]
    X_va, Y_va = data["X_va"], data["Y_va"]
    X_basis    = data["X_basis"]
    F_tr      = to_features_np(X_tr)
    F_va      = to_features_np(X_va)
    F_basis   = to_features_np(X_basis)
    print(f"[data] N_train={X_tr.shape[0]}  N_val={X_va.shape[0]}  "
          f"M={F_basis.shape[0]}  feature_dim={F_tr.shape[1]}")

    # State-error normalisation for rollout-horizon: use feature stds for x,
    # x_dot, theta_dot directly; for theta use std of train angles.
    sigmas_state = np.array([X_tr[:, 0].std(),
                             X_tr[:, 1].std(),
                             X_tr[:, 2].std(),
                             X_tr[:, 3].std()])

    # ---- 2. LINEAR BASELINE ----
    C = fit_linear_features(F_tr, Y_tr)
    Y_lin_tr = F_tr @ C.T
    Y_lin_va = F_va @ C.T
    mse_lin_tr = mse_per_dim(Y_tr, Y_lin_tr)
    mse_lin_va = mse_per_dim(Y_va, Y_lin_va)
    print(f"[linear] train MSE per dim: {mse_lin_tr.round(6)}  total={mse_lin_tr.mean():.4e}")
    print(f"[linear] val   MSE per dim: {mse_lin_va.round(6)}  total={mse_lin_va.mean():.4e}")

    # ---- 3. HYPERPARAMETER OPTIMISATION ----
    print("\n=== Hyperparameter optimisation ===")
    opt_results, init_source = run_optimisation(data)

    # ---- 4. BUILD INIT vs OPTIMISED NONLINEAR ENSEMBLES (for plotting) ----
    log_sigmas_init_per_out = np.stack(
        [np.asarray(r["log_hp_init"][1:]) for r in opt_results], axis=0)
    log_lambdas_init        = np.asarray(
        [r["log_hp_init"][0] for r in opt_results])
    log_sigmas_opt_per_out  = np.stack(
        [np.asarray(r["log_hp_opt"][1:])  for r in opt_results], axis=0)
    log_lambdas_opt         = np.asarray(
        [r["log_hp_opt"][0]  for r in opt_results])

    nl_init = fit_ensemble(F_tr, Y_tr, F_basis,
                           log_sigmas_init_per_out, log_lambdas_init, "init")
    nl_opt  = fit_ensemble(F_tr, Y_tr, F_basis,
                           log_sigmas_opt_per_out,  log_lambdas_opt,  "opt")

    Y_init_va = nl_init.predict(X_va)
    Y_opt_va  = nl_opt.predict(X_va)
    mse_nl_init_va = mse_per_dim(Y_va, Y_init_va)
    mse_nl_opt_va  = mse_per_dim(Y_va, Y_opt_va)
    mse_nl_opt_tr  = mse_per_dim(Y_tr, nl_opt.predict(X_tr))

    print(f"\n[nl-init] val MSE per dim: {mse_nl_init_va}  total={mse_nl_init_va.mean():.4e}")
    print(f"[nl-opt]  val MSE per dim: {mse_nl_opt_va}  total={mse_nl_opt_va.mean():.4e}")
    print(f"[nl-opt]  train MSE per dim: {mse_nl_opt_tr}  total={mse_nl_opt_tr.mean():.4e}")

    # ---- 5. PLOTS: pred vs true (linear and nonlinear) ----
    plot_pred_vs_true(Y_va, Y_lin_va,
                      "Linear baseline on 5-D features (validation)",
                      fig_path("predicted_vs_true_linear"))
    plot_pred_vs_true(Y_va, Y_opt_va,
                      "Optimised nonlinear kernel model (validation)",
                      fig_path("predicted_vs_true_nonlinear"))

    # ---- 6. PLOT: theta scan (the headline qualitative result) ----
    plot_theta_scan(nl_opt, fig_path("theta_scan"))

    # ---- 7. ROLLOUTS ----
    dt = CFG["dt"]
    n_steps = CFG["rollout_steps"]

    X0_osc = np.array([0.0, 0.0, np.pi - 0.3, 0.0])     # near downward eq
    X0_rot = np.array([0.0, 0.0, 0.0, 10.0])            # full-rotation regime

    roll_osc = rollout_horizon(nl_opt, X0_osc, sigmas_state,
                               n_steps, dt, CFG["horizon_threshold"])
    roll_rot = rollout_horizon(nl_opt, X0_rot, sigmas_state,
                               n_steps, dt, CFG["horizon_threshold"])
    plot_rollout(roll_osc,
                 "Rollout: small perturbation near downward equilibrium",
                 fig_path("rollout_oscillation"), dt)
    plot_rollout(roll_rot,
                 "Rollout: full-rotation regime (theta_dot=10)",
                 fig_path("rollout_fullrotation"), dt)

    # ---- 7b. ROLLOUT-HORIZON COMPARISON: t2.1 vs t2.2-opt vs t2.3 ----
    # Two regimes side by side, overlaying t2.1 / t2.2-opt / t2.3 to compare
    # how long each model tracks the true dynamics.
    cmp_ICs = [
        np.array([0.0, 0.0, 0.1, 3.0]),    # near upright + rapid spin
        np.array([1.0, 1.0, -1.5, 0.0]),   # off-centre downswing
    ]
    cmp_IC_labels = [
        "near upright + rapid spin\nX=[0, 0, 0.1, 3.0]",
        "off-centre, downswing\nX=[1, 1, -1.5, 0]",
    ]
    cmp_steps = int(t22.CFG["rollout_steps"])   # match Task 2.2's horizon length
    t21_model, t22_opt_model, t21_lambda = build_t22_models()
    model_entries = []
    if t21_model is not None:
        model_entries.append({"label": "t2.1", "model": t21_model,
                              "rollout_fn": t22.model_rollout, "colour": "tab:green"})
    if t22_opt_model is not None:
        model_entries.append({"label": "t2.2-opt", "model": t22_opt_model,
                              "rollout_fn": t22.model_rollout, "colour": "tab:red"})
    model_entries.append({"label": "t2.3", "model": nl_opt,
                          "rollout_fn": model_rollout, "colour": "tab:purple"})
    horizon_cmp = plot_rollout_horizon_comparison(
        model_entries, cmp_ICs, cmp_IC_labels, sigmas_state,
        cmp_steps, dt, CFG["horizon_threshold"],
        fig_path("rollout_horizon_comparison"))
    print(f"[fig] rollout-horizon comparison -> "
          f"{fig_path('rollout_horizon_comparison').name}")

    # ---- 8. PLOT: MSE comparison bar chart ----
    plot_mse_comparison(mse_lin_va, mse_nl_init_va, mse_nl_opt_va,
                        fig_path("mse_comparison"))

    # ---- 9. SAVE JSON: hyperparams + full results ----
    hp_json_path = FIG_DIR / f"{FIG_PREFIX}_hyperparams.json"
    with open(hp_json_path, "w") as f:
        json.dump({
            "tag":          FIG_PREFIX,
            "init_source":  init_source,
            "feature_order": ["x", "xdot", "sin(theta)", "cos(theta)", "thetadot"],
            "cfg":          {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in CFG.items()},
            "results":      opt_results,
        }, f, indent=2)
    print(f"[json] saved -> {hp_json_path.name}")

    results_path = FIG_DIR / f"{FIG_PREFIX}_results.json"
    summary = {
        "elapsed_total_s": time.time() - t_start,
        "init_source":     init_source,
        "linear": {
            "train_mse_per_dim": mse_lin_tr.tolist(),
            "val_mse_per_dim":   mse_lin_va.tolist(),
            "train_mse_total":   float(mse_lin_tr.mean()),
            "val_mse_total":     float(mse_lin_va.mean()),
        },
        "nl_init": {
            "val_mse_per_dim":   mse_nl_init_va.tolist(),
            "val_mse_total":     float(mse_nl_init_va.mean()),
        },
        "nl_opt": {
            "val_mse_per_dim":   mse_nl_opt_va.tolist(),
            "val_mse_total":     float(mse_nl_opt_va.mean()),
            "train_mse_per_dim": mse_nl_opt_tr.tolist(),
            "train_mse_total":   float(mse_nl_opt_tr.mean()),
            "sigmas_per_dim":    np.exp(log_sigmas_opt_per_out).tolist(),
            "lambdas_per_dim":   np.exp(log_lambdas_opt).tolist(),
        },
        "rollouts": {
            "oscillation": {
                "IC": X0_osc.tolist(),
                "t_break_s":  roll_osc["t_break"] if np.isfinite(roll_osc["t_break"]) else None,
                "step_break": int(roll_osc["step_break"]),
                "period_s":   roll_osc["period_s"],
                "cycles":     roll_osc["cycles"],
            },
            "fullrotation": {
                "IC": X0_rot.tolist(),
                "t_break_s":  roll_rot["t_break"] if np.isfinite(roll_rot["t_break"]) else None,
                "step_break": int(roll_rot["step_break"]),
                "period_s":   roll_rot["period_s"],
                "cycles":     roll_rot["cycles"],
            },
        },
        "sigmas_state_normalisation": sigmas_state.tolist(),
        "rollout_horizon_comparison": {
            "n_steps":   cmp_steps,
            "threshold": CFG["horizon_threshold"],
            "t21_lambda": t21_lambda,
            "per_model": horizon_cmp,
        },
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[json] saved -> {results_path.name}")

    # ---- 10. CONSOLE SUMMARY ----
    print("\n" + "=" * 72)
    print("TASK 2.3 SUMMARY")
    print("=" * 72)

    print("\nValidation MSE per output dim (lower is better):")
    print(f"  {'output':>10s}  {'linear':>12s}  {'NL-init':>12s}  {'NL-opt':>12s}")
    for k in range(4):
        print(f"  {DIM_NAMES[k]:>10s}  {mse_lin_va[k]:>12.4e}  "
              f"{mse_nl_init_va[k]:>12.4e}  {mse_nl_opt_va[k]:>12.4e}")
    print(f"  {'TOTAL':>10s}  {mse_lin_va.mean():>12.4e}  "
          f"{mse_nl_init_va.mean():>12.4e}  {mse_nl_opt_va.mean():>12.4e}")

    print("\nOptimised hyperparameters (natural units):")
    print(f"  {'dim':>10s}  {'lambda':>10s}  "
          + "  ".join(f"{n:>8s}" for n in
                      ["sig_x", "sig_xdot", "sig_sin", "sig_cos", "sig_thdot"]))
    sig_opt_natural = np.exp(log_sigmas_opt_per_out)
    lam_opt_natural = np.exp(log_lambdas_opt)
    for k in range(4):
        sigs = "  ".join(f"{s:>8.3f}" for s in sig_opt_natural[k])
        print(f"  {DIM_NAMES[k]:>10s}  {lam_opt_natural[k]:>10.3e}  {sigs}")

    print("\nRollout-horizon results (threshold "
          f"||(X_m - X_t)/sigma_state||_2 = {CFG['horizon_threshold']}):")
    for tag, roll in [("oscillation", roll_osc), ("full rotation", roll_rot)]:
        cyc_str = ("n/a" if roll["cycles"] is None
                   else f"{roll['cycles']:.2f} cycles")
        per_str = ("n/a" if roll["period_s"] is None
                   else f"{roll['period_s']:.2f}s")
        if np.isfinite(roll["t_break"]):
            ts = f"{roll['t_break']:.2g}s ({roll['step_break']} steps)"
        else:
            ts = f">{n_steps * dt:.0f}s (no break in {n_steps} steps)"
        print(f"  {tag:>14s}: T_break = {ts}, period {per_str}, {cyc_str}")

    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
