"""
SF3 Machine Learning -- Week 3, Task 3.1
========================================

Add the applied force F (post-tanh) as a 6th input feature to the dynamics
model and verify it on one-step predictions, rollouts, and an accuracy-vs-|F|
sweep. Targets remain the 4-D state delta dX = X(after performAction) - X(before).

Feature vector (6-D):
    phi(X, F) = [x, x_dot, sin theta, cos theta, theta_dot, F]

To realise a chosen F in the simulator we invert tanh:
    a = max_force * arctanh(F / max_force)        max_force = 20

Pipeline:
  1. Gather/cache train/val/test (X, F, dX) pairs (train/val: F~U(-15, 15);
     test: F~U(-19.5, 19.5) so the same set populates the wider sweep bins).
  2. Linear baseline dX ~ Phi @ C^T fit by np.linalg.lstsq.
  3. Kernel regression (delta-NM by default; "NN" available via CFG):
        alpha = solve(K_MN K_NM + lam K_MM + jitter I,  K_MN y)
     Per-output, 7 hyperparameters (1 lambda + 6 sigma).
  4. Tune hyperparameters from scratch with JAX gradients + scipy L-BFGS-B in
     log-space; multi-start (canonical std-based init + 5 perturbed restarts);
     tuning uses a 1000-pt subset for speed, then alpha is refit on full train.
  5. Matched-force rollouts: same F sequence (zero / constant / sinusoidal)
     fed to plant and to each model.
  6. Six figures + JSON results.

Run:
    python week3/task3_1.py             # full run (~5-15 min on CPU)
    TASK3_1_QUICK=1 python week3/task3_1.py   # smoke test (~1 min)
"""

from __future__ import annotations
from dataclasses import dataclass
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
from jax import jit, value_and_grad

from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cartpole import CartPole, _remap_angle                              # noqa: E402


# =========================================================================
# CONFIG
# =========================================================================
QUICK = os.environ.get("TASK3_1_QUICK", "0") == "1"

CFG = {
    "N_train":           2000,
    "N_val":              500,
    "N_test":            1500,
    "N_tune":            1000,
    "M":                  200,

    "F_train":           15.0,     # train + val F sampled U(-F_train, F_train)
    "F_test":            19.5,     # test F sampled U(-F_test, F_test)  (wider for sweep)

    "lambda_init":       1e-4,
    "log_lambda_bounds": (-18.0, -2.0),
    "log_sigma_bounds":  (-6.0,   7.0),  # widened from week2 (sig_F expected large)
    "maxiter":           200,
    "ftol":              1e-10,
    "gtol":              1e-8,
    "n_restarts":        5,
    "restart_pert_std":  1.0,

    "seed":              0,
    "basis":             "NM",     # "NM" (sparse subset) | "NN" (full Tikhonov)
    "rollout_steps":     100,
    "dt":                0.1,

    "sweep_nbins":       12,       # bins in |F| from 0 to F_test for accuracy plot
    "jitter":            1e-8,     # added to diagonal of normal-equation matrix
}

if QUICK:
    CFG.update({
        "N_train": 300, "N_val": 100, "N_test": 200,
        "N_tune": 300, "M": 40,
        "n_restarts": 1, "maxiter": 40,
    })

SAMPLE_RANGES = {
    "x":         (-5.0,    5.0),
    "x_dot":     (-10.0,  10.0),
    "theta":     (-np.pi,  np.pi),
    "theta_dot": (-15.0,  15.0),
}

FEATURE_LABELS = [r"$x$", r"$\dot x$", r"$\sin\theta$", r"$\cos\theta$",
                  r"$\dot\theta$", r"$F$"]
FEATURE_NAMES  = ["x", "xdot", "sin", "cos", "thdot", "F"]
DELTA_LABELS   = [r"$\Delta x$", r"$\Delta \dot x$",
                  r"$\Delta \theta$", r"$\Delta \dot\theta$"]
STATE_LABELS   = [r"$x$", r"$\dot x$", r"$\theta$", r"$\dot\theta$"]
DIM_NAMES      = ["x", "xdot", "theta", "thetadot"]
SCAN_LABELS    = [r"$x$", r"$\dot x$", r"$\theta$", r"$\dot\theta$", r"$F$"]
SCAN_NAMES     = ["x", "xdot", "theta", "thetadot", "F"]
SCAN_RANGES    = [SAMPLE_RANGES["x"], SAMPLE_RANGES["x_dot"],
                  SAMPLE_RANGES["theta"], SAMPLE_RANGES["theta_dot"],
                  (-CFG["F_train"], CFG["F_train"])]

FIG_DIR   = ROOT / "figures"
CACHE_DIR = ROOT / "week3"
FIG_DIR.mkdir(exist_ok=True)
FIG_PREFIX = "task3_1"

# Read max_force from the simulator so the arctanh inversion always matches.
MAX_FORCE = float(CartPole(visual=False).max_force)


def fig_path(name: str) -> Path:
    return FIG_DIR / f"{FIG_PREFIX}_{name}.png"


def data_cache_path() -> Path:
    q = "_quick" if QUICK else ""
    return CACHE_DIR / (
        f"{FIG_PREFIX}_data{q}_N{CFG['N_train']}_V{CFG['N_val']}_"
        f"T{CFG['N_test']}_seed{CFG['seed']}.npz"
    )


# =========================================================================
# 1. DATA GATHERING
# =========================================================================
def sample_states(n: int, rng: np.random.Generator) -> np.ndarray:
    keys = ("x", "x_dot", "theta", "theta_dot")
    lo = np.array([SAMPLE_RANGES[k][0] for k in keys])
    hi = np.array([SAMPLE_RANGES[k][1] for k in keys])
    return rng.uniform(lo, hi, size=(n, 4))


def force_to_action(F: np.ndarray | float) -> np.ndarray | float:
    """Invert the simulator's tanh: a = max_force * arctanh(F/max_force).
    Clip F to (-max_force, max_force) with a tiny margin so arctanh stays finite."""
    eps = 1e-6
    F_cl = np.clip(F, -MAX_FORCE * (1.0 - eps), MAX_FORCE * (1.0 - eps))
    return MAX_FORCE * np.arctanh(F_cl / MAX_FORCE)


def gather_dataset(n: int, rng: np.random.Generator, F_low: float, F_high: float):
    """n triples (X, F, dX). For each i:
      a = arctanh-inverse of F[i]; performAction(a); dX = state - X; remap dX[2]."""
    sim = CartPole(visual=False)
    X = sample_states(n, rng)
    F = rng.uniform(F_low, F_high, size=n)
    A = force_to_action(F)
    Y = np.zeros_like(X)
    for i in range(n):
        sim.setState(X[i].tolist())
        sim.performAction(float(A[i]))
        dY = sim.getState() - X[i]
        dY[2] = _remap_angle(dY[2])
        Y[i] = dY
    return X, F, Y


def load_or_gather_data() -> dict:
    cache = data_cache_path()
    n_tr, n_va, n_te = CFG["N_train"], CFG["N_val"], CFG["N_test"]
    seed = CFG["seed"]
    F_tr_hi = CFG["F_train"]
    F_te_hi = CFG["F_test"]

    if cache.exists():
        print(f"[data] reusing {cache.name}")
        d = np.load(cache)
        X_tr, F_tr, Y_tr = d["X_tr"], d["F_tr"], d["Y_tr"]
        X_va, F_va, Y_va = d["X_va"], d["F_va"], d["Y_va"]
        X_te, F_te, Y_te = d["X_te"], d["F_te"], d["Y_te"]
        assert X_tr.shape[0] == n_tr and X_va.shape[0] == n_va and X_te.shape[0] == n_te
    else:
        print(f"[data] generating {n_tr + n_va + n_te} samples (seed={seed})...")
        t0 = time.time()
        rng = np.random.default_rng(seed)
        X_tr, F_tr, Y_tr = gather_dataset(n_tr, rng, -F_tr_hi,  F_tr_hi)
        X_va, F_va, Y_va = gather_dataset(n_va, rng, -F_tr_hi,  F_tr_hi)
        X_te, F_te, Y_te = gather_dataset(n_te, rng, -F_te_hi,  F_te_hi)
        np.savez(cache,
                 X_tr=X_tr, F_tr=F_tr, Y_tr=Y_tr,
                 X_va=X_va, F_va=F_va, Y_va=Y_va,
                 X_te=X_te, F_te=F_te, Y_te=Y_te)
        print(f"[data] cached -> {cache.name}  ({time.time()-t0:.1f}s)")

    rng_b = np.random.default_rng(seed + 1)
    M = min(CFG["M"], n_tr)
    basis_idx = rng_b.choice(n_tr, size=M, replace=False)

    rng_tune = np.random.default_rng(seed + 2)
    n_tune = min(CFG["N_tune"], n_tr)
    tune_idx = rng_tune.choice(n_tr, size=n_tune, replace=False)

    return {
        "X_tr": X_tr, "F_tr": F_tr, "Y_tr": Y_tr,
        "X_va": X_va, "F_va": F_va, "Y_va": Y_va,
        "X_te": X_te, "F_te": F_te, "Y_te": Y_te,
        "basis_idx": basis_idx,
        "tune_idx":  tune_idx,
    }


# =========================================================================
# 2. FEATURE TRANSFORM
# =========================================================================
def to_features_np(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """(..., 4), (...,) -> (..., 6): [x, x_dot, sin theta, cos theta, theta_dot, F]."""
    X = np.atleast_2d(X)
    F = np.atleast_1d(F)
    return np.stack([X[..., 0], X[..., 1],
                     np.sin(X[..., 2]), np.cos(X[..., 2]),
                     X[..., 3], F], axis=-1)


def to_features_jax(X, F):
    return jnp.stack([X[..., 0], X[..., 1],
                      jnp.sin(X[..., 2]), jnp.cos(X[..., 2]),
                      X[..., 3], F], axis=-1)


# =========================================================================
# 3. KERNEL + FIT  (JAX, anisotropic Gaussian on 6-D features)
# =========================================================================
def kernel_matrix(F1, F2, log_sigmas):
    """K[i, j] = exp(-sum_d (F1[i,d] - F2[j,d])^2 / (2 sigma_d^2)).
    F1: (n1, 6), F2: (n2, 6), log_sigmas: (6,). Plain Euclidean -- no periodic
    correction (sin/cos handle periodicity)."""
    sigmas = jnp.exp(log_sigmas)
    inv2s2 = 1.0 / (2.0 * sigmas ** 2)
    diff = F1[:, None, :] - F2[None, :, :]
    return jnp.exp(-jnp.sum(diff ** 2 * inv2s2, axis=-1))


def fit_alpha_NM(F_tr, y_col, F_basis, log_lambda, log_sigmas, jitter):
    """Sparse subset (eq. 15) with diagonal jitter for numerical safety:
       alpha = solve(K_MN K_NM + lam K_MM + jitter I,  K_MN y)."""
    K_NM = kernel_matrix(F_tr, F_basis, log_sigmas)
    K_MM = kernel_matrix(F_basis, F_basis, log_sigmas)
    K_MN = K_NM.T
    lam = jnp.exp(log_lambda)
    M = F_basis.shape[0]
    A = K_MN @ K_NM + lam * K_MM + jitter * jnp.eye(M)
    return jnp.linalg.solve(A, K_MN @ y_col)


def fit_alpha_NN(F_tr, y_col, log_lambda, log_sigmas, jitter):
    """Full Tikhonov: alpha = solve(K_NN + lam I + jitter I, y)."""
    K_NN = kernel_matrix(F_tr, F_tr, log_sigmas)
    lam = jnp.exp(log_lambda)
    N = F_tr.shape[0]
    A = K_NN + (lam + jitter) * jnp.eye(N)
    return jnp.linalg.solve(A, y_col)


def kernel_predict(F_test, F_centres, alpha, log_sigmas):
    return kernel_matrix(F_test, F_centres, log_sigmas) @ alpha


# ---- NumPy mirrors (no autograd needed for plotting / rollouts) ----
def kernel_matrix_np(F1, F2, log_sigmas):
    sigmas = np.exp(log_sigmas)
    inv2s2 = 1.0 / (2.0 * sigmas ** 2)
    diff = F1[:, None, :] - F2[None, :, :]
    return np.exp(-np.sum(diff ** 2 * inv2s2, axis=-1))


def fit_alpha_NM_np(F_tr, y_col, F_basis, log_lambda, log_sigmas, jitter):
    K_NM = kernel_matrix_np(F_tr, F_basis, log_sigmas)
    K_MM = kernel_matrix_np(F_basis, F_basis, log_sigmas)
    K_MN = K_NM.T
    lam = float(np.exp(log_lambda))
    M = F_basis.shape[0]
    A = K_MN @ K_NM + lam * K_MM + jitter * np.eye(M)
    b = K_MN @ y_col
    alpha, *_ = np.linalg.lstsq(A, b, rcond=None)
    return alpha


def fit_alpha_NN_np(F_tr, y_col, log_lambda, log_sigmas, jitter):
    K_NN = kernel_matrix_np(F_tr, F_tr, log_sigmas)
    lam = float(np.exp(log_lambda))
    N = F_tr.shape[0]
    A = K_NN + (lam + jitter) * np.eye(N)
    alpha, *_ = np.linalg.lstsq(A, y_col, rcond=None)
    return alpha


@dataclass
class KernelEnsemble:
    """Holds 4 per-output kernel models sharing a single set of centres."""
    label:               str
    F_basis:             np.ndarray            # (M, 6)
    alphas:              np.ndarray            # (M, 4)
    log_sigmas_per_out:  np.ndarray            # (4, 6)
    log_lambdas:         np.ndarray            # (4,)

    def predict(self, X: np.ndarray, F: np.ndarray) -> np.ndarray:
        Phi = to_features_np(X, F)
        n = Phi.shape[0]
        out = np.zeros((n, 4))
        for j in range(4):
            K = kernel_matrix_np(Phi, self.F_basis, self.log_sigmas_per_out[j])
            out[:, j] = K @ self.alphas[:, j]
        return out


def fit_ensemble(Phi_tr, Y_tr, F_basis, log_sigmas_per_out, log_lambdas,
                 label, basis: str, jitter: float) -> KernelEnsemble:
    M = F_basis.shape[0]
    alphas = np.zeros((M, 4))
    if basis == "NM":
        for j in range(4):
            alphas[:, j] = fit_alpha_NM_np(
                Phi_tr, Y_tr[:, j], F_basis,
                float(log_lambdas[j]), log_sigmas_per_out[j], jitter,
            )
    elif basis == "NN":
        assert M == Phi_tr.shape[0], "NN form requires M == N"
        for j in range(4):
            alphas[:, j] = fit_alpha_NN_np(
                Phi_tr, Y_tr[:, j],
                float(log_lambdas[j]), log_sigmas_per_out[j], jitter,
            )
    else:
        raise ValueError(f"unknown basis: {basis}")
    return KernelEnsemble(
        label=label, F_basis=F_basis, alphas=alphas,
        log_sigmas_per_out=log_sigmas_per_out.copy(),
        log_lambdas=log_lambdas.copy(),
    )


# =========================================================================
# 4. LINEAR BASELINE  (6-D features, no bias)
# =========================================================================
def fit_linear_features(Phi: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Solve Y = Phi C^T by lstsq. Returns C of shape (4, 6)."""
    CT, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    return CT.T


def mse_per_dim(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    return ((Y_true - Y_pred) ** 2).mean(axis=0)


# =========================================================================
# 5. JAX HYPERPARAMETER OPTIMISATION  (multi-start L-BFGS-B in log-space)
# =========================================================================
def make_val_mse_fn(Phi_tune_j, y_tune_col, Phi_va_j, y_va_col, Phi_basis_j,
                    basis: str, jitter: float):
    """jit(value_and_grad(val_mse)) for one output dim. Inputs are JAX arrays.
    log_hp ordering: [log_lambda, log_sigma_0, ..., log_sigma_5]."""
    if basis == "NM":
        centres = Phi_basis_j

        def val_mse(log_hp):
            log_lambda = log_hp[0]
            log_sigmas = log_hp[1:]
            alpha = fit_alpha_NM(Phi_tune_j, y_tune_col, centres,
                                 log_lambda, log_sigmas, jitter)
            preds = kernel_predict(Phi_va_j, centres, alpha, log_sigmas)
            mse = jnp.mean((preds - y_va_col) ** 2)
            return jnp.nan_to_num(mse, nan=1e20, posinf=1e20, neginf=1e20)

    elif basis == "NN":
        centres = Phi_tune_j

        def val_mse(log_hp):
            log_lambda = log_hp[0]
            log_sigmas = log_hp[1:]
            alpha = fit_alpha_NN(Phi_tune_j, y_tune_col,
                                 log_lambda, log_sigmas, jitter)
            preds = kernel_predict(Phi_va_j, centres, alpha, log_sigmas)
            mse = jnp.mean((preds - y_va_col) ** 2)
            return jnp.nan_to_num(mse, nan=1e20, posinf=1e20, neginf=1e20)

    else:
        raise ValueError(f"unknown basis: {basis}")

    return jit(value_and_grad(val_mse))


def initial_hyperparams(Phi_tune: np.ndarray) -> np.ndarray:
    """Canonical init: log_lambda = log(1e-4); log_sigma_d = log(std of feature d)."""
    sig0 = Phi_tune.std(axis=0)
    sig0 = np.maximum(sig0, 1e-3)
    log_lambda_init = float(np.log(CFG["lambda_init"]))
    return np.concatenate([[log_lambda_init], np.log(sig0)])


def optimise_one_dim(val_and_grad_fn, log_hp_init, bounds):
    """One L-BFGS-B run. f_and_g handles NaN/Inf defensively."""
    trace = []

    def f_and_g(x_np):
        try:
            v, g = val_and_grad_fn(jnp.asarray(x_np))
            v_f = float(v)
            g_np = np.asarray(g, dtype=np.float64)
        except Exception:
            return 1e20, np.zeros_like(x_np)
        if not np.isfinite(v_f) or not np.isfinite(g_np).all():
            return 1e20, np.zeros_like(x_np)
        trace.append(v_f)
        return v_f, g_np

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


def check_bounds(opt_results):
    """Warn for any optimised log-hyperparameter within 0.5 of a bound."""
    log_lo,  log_hi  = CFG["log_lambda_bounds"]
    log_slo, log_shi = CFG["log_sigma_bounds"]
    labels = [("lambda", 0, log_lo, log_hi)] + [
        (f"sigma_{name}", idx + 1, log_slo, log_shi)
        for idx, name in enumerate(FEATURE_NAMES)
    ]
    for r in opt_results:
        x = r["log_hp_opt"]
        for name, idx, lo, hi in labels:
            v = float(x[idx])
            if abs(v - hi) < 0.5:
                print(f"[check] dim {r['dim']} ({r['dim_name']}): "
                      f"{name} hit upper bound (log={v:.2f}, value={np.exp(v):.3g}); "
                      f"widen the bound and re-run.")
            elif abs(v - lo) < 0.5:
                print(f"[check] dim {r['dim']} ({r['dim_name']}): "
                      f"{name} near lower bound (log={v:.2f}, value={np.exp(v):.3g}).")


def run_optimisation(data) -> list[dict]:
    """Tune hyperparameters per output dim with multi-start L-BFGS-B.
    Tuning uses the 1000-pt subset of train. Returns one dict per output dim."""
    X_tr, F_tr, Y_tr = data["X_tr"], data["F_tr"], data["Y_tr"]
    X_va, F_va, Y_va = data["X_va"], data["F_va"], data["Y_va"]
    basis_idx        = data["basis_idx"]
    tune_idx         = data["tune_idx"]

    Phi_tr      = to_features_np(X_tr, F_tr)
    Phi_va      = to_features_np(X_va, F_va)
    Phi_basis   = Phi_tr[basis_idx]
    Phi_tune    = Phi_tr[tune_idx]
    Y_tune      = Y_tr[tune_idx]

    log_hp_init = initial_hyperparams(Phi_tune)
    log_lo,  log_hi  = CFG["log_lambda_bounds"]
    log_slo, log_shi = CFG["log_sigma_bounds"]
    bounds   = [(log_lo, log_hi)] + [(log_slo, log_shi)] * 6
    bounds_lo = np.array([b[0] for b in bounds])
    bounds_hi = np.array([b[1] for b in bounds])

    Phi_tune_j  = jnp.asarray(Phi_tune)
    Phi_va_j    = jnp.asarray(Phi_va)
    Phi_basis_j = jnp.asarray(Phi_basis)

    n_restarts = int(CFG["n_restarts"])
    pert_std   = float(CFG["restart_pert_std"])
    basis      = CFG["basis"]
    jitter     = float(CFG["jitter"])

    print(f"[tune] N_tune={Phi_tune.shape[0]}  M={Phi_basis.shape[0]}  "
          f"basis={basis}  n_restarts={n_restarts}+1  feature_dim={Phi_tune.shape[1]}")

    results = []
    for j in range(4):
        Y_tune_col = jnp.asarray(Y_tune[:, j])
        Y_va_col   = jnp.asarray(Y_va[:, j])
        vg = make_val_mse_fn(Phi_tune_j, Y_tune_col, Phi_va_j, Y_va_col,
                             Phi_basis_j, basis, jitter)

        t_compile = time.time()
        v0, g0 = vg(jnp.asarray(log_hp_init))
        v0 = float(v0); _ = float(jnp.linalg.norm(g0))   # block on device
        print(f"[opt dim {j} {DIM_NAMES[j]:>8s}] compile+init {time.time()-t_compile:.1f}s   "
              f"init val MSE: {v0:.4e}")

        rng_j = np.random.default_rng(CFG["seed"] + 100 + j)
        inits = [np.asarray(log_hp_init, dtype=np.float64)]
        for _r in range(n_restarts):
            pert  = rng_j.normal(0.0, pert_std, size=7)
            start = np.clip(log_hp_init + pert, bounds_lo, bounds_hi)
            inits.append(start)

        t0 = time.time()
        runs = []
        for start in inits:
            res_k, trace_k = optimise_one_dim(vg, start, bounds)
            runs.append({"loss":  float(res_k.fun),
                         "res":   res_k, "trace": trace_k,
                         "start": np.asarray(start, dtype=np.float64)})
        dt = time.time() - t0

        best_idx = int(np.argmin([rk["loss"] for rk in runs]))
        best     = runs[best_idx]
        res      = best["res"]
        trace    = best["trace"]
        _, g_final = vg(jnp.asarray(res.x))
        g_norm = float(jnp.linalg.norm(g_final))
        ratio = v0 / max(res.fun, 1e-30)

        print(f"                     opt done in {dt:.1f}s   iters={res.nit:3d}   "
              f"final val MSE: {res.fun:.4e}   grad norm: {g_norm:.2e}   "
              f"({ratio:.1f}x improvement)")
        sig_opt = res.x[1:]
        sig_str = ", ".join(f"{s:.2f}" for s in sig_opt)
        print(f"                     log_hp_opt: lam={res.x[0]:.3f}  sig=[{sig_str}]")

        restart_losses = sorted(float(rk["loss"]) for rk in runs)
        lmin, lmax = restart_losses[0], restart_losses[-1]
        rel_spread_pct = (lmax - lmin) / max(lmin, 1e-30) * 100.0
        print(f"[multi-start] dim {j} ({DIM_NAMES[j]}): {1+n_restarts} runs, "
              f"loss min={lmin:.2e} max={lmax:.2e} (rel-spread={rel_spread_pct:.1f}%)")

        results.append({
            "dim":          j,
            "dim_name":     DIM_NAMES[j],
            "log_hp_init":  log_hp_init.tolist(),
            "log_hp_opt":   res.x.tolist(),
            "val_mse_init": v0,
            "val_mse_opt":  float(res.fun),
            "iters":        int(res.nit),
            "n_evals":      len(trace),
            "success":      bool(res.success),
            "message":      str(res.message),
            "elapsed_s":    dt,
            "improvement_factor": ratio,
            "restart_losses": restart_losses,
            "rel_spread_pct": rel_spread_pct,
            "n_restarts":   n_restarts,
        })
    return results


# =========================================================================
# 6. ROLLOUTS  (matched-force: same F sequence to plant and to models)
# =========================================================================
def true_rollout_with_force(X0: np.ndarray, F_seq: np.ndarray) -> np.ndarray:
    sim = CartPole(visual=False)
    sim.setState(X0.tolist())
    n = F_seq.shape[0]
    traj = np.zeros((n + 1, 4))
    traj[0] = sim.getState()
    A = force_to_action(F_seq)
    for t in range(n):
        sim.performAction(float(A[t]))
        traj[t + 1] = sim.getState()
    return traj


def model_rollout_kernel(model: KernelEnsemble, X0: np.ndarray,
                         F_seq: np.ndarray) -> np.ndarray:
    """Iterate X_{t+1} = X_t + model.predict(X_t, F_t). NO theta remap (sin/cos
    absorb periodicity)."""
    n = F_seq.shape[0]
    traj = np.zeros((n + 1, 4))
    X = X0.copy()
    traj[0] = X
    for t in range(n):
        dX = model.predict(X[None, :], np.array([F_seq[t]]))[0]
        X = X + dX
        traj[t + 1] = X
    return traj


def model_rollout_linear(C: np.ndarray, X0: np.ndarray,
                         F_seq: np.ndarray) -> np.ndarray:
    n = F_seq.shape[0]
    traj = np.zeros((n + 1, 4))
    X = X0.copy()
    traj[0] = X
    for t in range(n):
        phi = to_features_np(X[None, :], np.array([F_seq[t]]))[0]
        X = X + C @ phi
        traj[t + 1] = X
    return traj


def build_force_sequences(n_steps: int, dt: float) -> list[tuple[str, np.ndarray]]:
    t = np.arange(n_steps) * dt
    seqs = [
        ("zero",       np.zeros(n_steps)),
        ("constant=5", np.full(n_steps, 5.0)),
        ("sinusoidal", 10.0 * np.sin(2.0 * np.pi * t / 2.0)),
    ]
    return seqs


# =========================================================================
# 7. FIGURES
# =========================================================================
def plot_scatter(Y_true, Y_lin, Y_ker, savepath):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    rows = [(Y_lin, "linear"), (Y_ker, "kernel")]
    for r, (Y_pred, name) in enumerate(rows):
        for k in range(4):
            ax = axes[r, k]
            ax.scatter(Y_true[:, k], Y_pred[:, k], s=4, alpha=0.35)
            lo = float(min(Y_true[:, k].min(), Y_pred[:, k].min()))
            hi = float(max(Y_true[:, k].max(), Y_pred[:, k].max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            mse_k = float(np.mean((Y_true[:, k] - Y_pred[:, k]) ** 2))
            ax.set_title(f"{name}  {DELTA_LABELS[k]}  MSE={mse_k:.2e}", fontsize=10)
            ax.set_xlabel("true")
            if k == 0:
                ax.set_ylabel(f"{name}: predicted")
            ax.set_aspect("equal", "box")
    fig.suptitle("Predicted vs true on the test set (all |F| in [0, F_test])")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_scans_1d(C_lin, ker_model, savepath, n=200):
    """Sweep one input at a time, others at origin (and F=0 except in the F scan).
    4 rows (output dim) x 5 cols (sweep var)."""
    fig, axes = plt.subplots(4, 5, figsize=(20, 12), sharex=False)
    for c, (lo, hi) in enumerate(SCAN_RANGES):
        sweep = np.linspace(lo, hi, n)
        X_scan = np.zeros((n, 4))
        F_scan = np.zeros(n)
        if c < 4:
            X_scan[:, c] = sweep
        else:
            F_scan[:] = sweep

        # True
        sim = CartPole(visual=False)
        Y_true = np.zeros((n, 4))
        A_scan = force_to_action(F_scan)
        for i in range(n):
            sim.setState(X_scan[i].tolist())
            sim.performAction(float(A_scan[i]))
            dY = sim.getState() - X_scan[i]
            dY[2] = _remap_angle(dY[2])
            Y_true[i] = dY

        Phi_scan = to_features_np(X_scan, F_scan)
        Y_lin = Phi_scan @ C_lin.T
        Y_ker = ker_model.predict(X_scan, F_scan)

        for r in range(4):
            ax = axes[r, c]
            ax.plot(sweep, Y_true[:, r], "k-",  lw=1.6, label="true")
            ax.plot(sweep, Y_lin[:,  r], "--", color="tab:gray", lw=1.2, label="linear")
            ax.plot(sweep, Y_ker[:,  r], "--", color="tab:red",  lw=1.4, label="kernel")
            if r == 0:
                ax.set_title(f"sweep {SCAN_LABELS[c]}")
            if r == 3:
                ax.set_xlabel(SCAN_LABELS[c])
            if c == 0:
                ax.set_ylabel(DELTA_LABELS[r])
            ax.grid(True, alpha=0.4)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle("1-D scans of each input feature at origin (other features = 0)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_force_scan(C_lin, ker_model, savepath, n=400):
    """Focused F sweep at origin state; one panel per output dim. Should be linear."""
    F_scan = np.linspace(-CFG["F_test"], CFG["F_test"], n)
    X_scan = np.zeros((n, 4))
    sim = CartPole(visual=False)
    A_scan = force_to_action(F_scan)
    Y_true = np.zeros((n, 4))
    for i in range(n):
        sim.setState(X_scan[i].tolist())
        sim.performAction(float(A_scan[i]))
        dY = sim.getState() - X_scan[i]
        dY[2] = _remap_angle(dY[2])
        Y_true[i] = dY
    Phi_scan = to_features_np(X_scan, F_scan)
    Y_lin = Phi_scan @ C_lin.T
    Y_ker = ker_model.predict(X_scan, F_scan)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for k in range(4):
        ax = axes[k]
        ax.plot(F_scan, Y_true[:, k], "k-",  lw=1.6, label="true")
        ax.plot(F_scan, Y_lin[:,  k], "--", color="tab:gray", lw=1.2, label="linear")
        ax.plot(F_scan, Y_ker[:,  k], "--", color="tab:red",  lw=1.4, label="kernel")
        ax.axvline(-CFG["F_train"], color="tab:blue", ls=":", lw=1)
        ax.axvline( CFG["F_train"], color="tab:blue", ls=":", lw=1,
                    label=f"|F|=F_train={CFG['F_train']}")
        ax.set_xlabel(r"$F$")
        ax.set_title(DELTA_LABELS[k])
        ax.grid(True, alpha=0.4)
        if k == 0:
            ax.legend(fontsize=8, loc="best")
            ax.set_ylabel("one-step $\\Delta$ at origin state")
    fig.suptitle("Focused F-only scan at X = 0 (should be near-linear in F)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_contours_2d(ker_model, savepath, n_grid=50):
    """Kernel model predictions on 2D grids. One row per state component, all
    paired against F on the x-axis. Columns = output dim."""
    F_grid = np.linspace(-CFG["F_test"], CFG["F_test"], n_grid)
    panels = [
        (np.linspace(*SAMPLE_RANGES["x"],         n_grid), 0, r"$x$"),
        (np.linspace(*SAMPLE_RANGES["x_dot"],     n_grid), 1, r"$\dot x$"),
        (np.linspace(*SAMPLE_RANGES["theta"],     n_grid), 2, r"$\theta$"),
        (np.linspace(*SAMPLE_RANGES["theta_dot"], n_grid), 3, r"$\dot\theta$"),
    ]
    fig, axes = plt.subplots(len(panels), 4, figsize=(20, 4.2 * len(panels)))
    for r, (yg, varying_state_idx, y_label) in enumerate(panels):
        XG, YG = np.meshgrid(F_grid, yg, indexing="xy")
        flat_X = np.zeros((n_grid * n_grid, 4))
        flat_F = XG.ravel()
        flat_X[:, varying_state_idx] = YG.ravel()
        Y_ker = ker_model.predict(flat_X, flat_F)
        sim = CartPole(visual=False)
        A_flat = force_to_action(flat_F)
        Y_true = np.zeros_like(Y_ker)
        for i in range(flat_X.shape[0]):
            sim.setState(flat_X[i].tolist())
            sim.performAction(float(A_flat[i]))
            dY = sim.getState() - flat_X[i]
            dY[2] = _remap_angle(dY[2])
            Y_true[i] = dY

        for k in range(4):
            ax = axes[r, k]
            Z = Y_ker[:, k].reshape(n_grid, n_grid)
            Z_true = Y_true[:, k].reshape(n_grid, n_grid)
            vmin = min(Z.min(), Z_true.min())
            vmax = max(Z.max(), Z_true.max())
            levels = np.linspace(vmin, vmax, 21)
            cs = ax.contourf(XG, YG, Z, levels=levels, cmap="viridis")
            ax.contour(XG, YG, Z_true, levels=levels[::4], colors="white",
                       linewidths=0.6, alpha=0.7)
            fig.colorbar(cs, ax=ax)
            ax.set_xlabel(r"$F$")
            ax.set_ylabel(y_label)
            ax.set_title(f"kernel {DELTA_LABELS[k]}", fontsize=10)
    fig.suptitle("Kernel model contours over (F, state-component); "
                 "white lines = true one-step delta")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_rollouts(C_lin, ker_model, savepath, n_steps=None, dt=None):
    n_steps = n_steps or CFG["rollout_steps"]
    dt      = dt or CFG["dt"]
    seqs = build_force_sequences(n_steps, dt)
    X0 = np.array([0.0, 0.0, np.pi, 0.0])  # pole down, still, centred
    t = np.arange(n_steps + 1) * dt

    n_cols = len(seqs)
    fig, axes = plt.subplots(5, n_cols, figsize=(5.0 * n_cols, 12), sharex=True)
    for c, (name, F_seq) in enumerate(seqs):
        traj_t = true_rollout_with_force(X0, F_seq)
        traj_l = model_rollout_linear(C_lin, X0, F_seq)
        traj_k = model_rollout_kernel(ker_model, X0, F_seq)
        # Remap theta only for visualisation
        th_t = np.array([_remap_angle(a) for a in traj_t[:, 2]])
        th_l = np.array([_remap_angle(a) for a in traj_l[:, 2]])
        th_k = np.array([_remap_angle(a) for a in traj_k[:, 2]])
        series = [
            (traj_t[:, 0], traj_l[:, 0], traj_k[:, 0]),
            (traj_t[:, 1], traj_l[:, 1], traj_k[:, 1]),
            (th_t,        th_l,        th_k),
            (traj_t[:, 3], traj_l[:, 3], traj_k[:, 3]),
        ]
        for r in range(4):
            ax = axes[r, c]
            ax.plot(t,           series[r][0], "k-",  lw=1.5, label="true")
            ax.plot(t,           series[r][1], "--", color="tab:gray", lw=1.1, label="linear")
            ax.plot(t,           series[r][2], "--", color="tab:red",  lw=1.3, label="kernel")
            if c == 0:
                ax.set_ylabel(STATE_LABELS[r])
            if r == 0:
                ax.set_title(f"F = {name}")
                if c == 0:
                    ax.legend(fontsize=8, loc="best")
        # F(t) panel
        ax_f = axes[4, c]
        ax_f.plot(np.arange(n_steps) * dt, F_seq, "b-", lw=1.2)
        ax_f.set_ylabel(r"$F(t)$")
        ax_f.set_xlabel("time / s")
        ax_f.grid(True, alpha=0.4)
    fig.suptitle(
        f"Matched-force rollouts from X0 = [0, 0, pi, 0] over {n_steps} steps"
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def compute_force_sweep(C_lin, ker_model, X_te, F_te, Y_te, n_bins: int):
    """Bin test data by |F|; per-bin MSE summed over 4 output dims, for each model."""
    Phi_te = to_features_np(X_te, F_te)
    Y_lin  = Phi_te @ C_lin.T
    Y_ker  = ker_model.predict(X_te, F_te)
    absF = np.abs(F_te)
    edges = np.linspace(0.0, CFG["F_test"], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mse_lin = np.full(n_bins, np.nan)
    mse_ker = np.full(n_bins, np.nan)
    counts  = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = (absF >= edges[b]) & (absF < edges[b + 1])
        if b == n_bins - 1:   # include the right edge
            mask = (absF >= edges[b]) & (absF <= edges[b + 1])
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            mse_lin[b] = float(np.mean((Y_te[mask] - Y_lin[mask]) ** 2))
            mse_ker[b] = float(np.mean((Y_te[mask] - Y_ker[mask]) ** 2))
    return {
        "bin_edges":          edges.tolist(),
        "bin_centers":        centers.tolist(),
        "bin_counts":         counts.tolist(),
        "linear_mse_per_bin": mse_lin.tolist(),
        "kernel_mse_per_bin": mse_ker.tolist(),
    }


def plot_force_accuracy(sweep, savepath):
    centers = np.asarray(sweep["bin_centers"])
    counts  = np.asarray(sweep["bin_counts"])
    mse_lin = np.asarray(sweep["linear_mse_per_bin"])
    mse_ker = np.asarray(sweep["kernel_mse_per_bin"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(centers, mse_lin, "o-", color="tab:gray", lw=1.5, label="linear")
    ax.semilogy(centers, mse_ker, "o-", color="tab:red",  lw=1.5, label="kernel")
    ax.axvline(CFG["F_train"], color="tab:blue", ls="--", lw=1.2,
               label=f"F_train = {CFG['F_train']}")
    ax.set_xlabel(r"$|F|$ (bin centre)")
    ax.set_ylabel("test MSE (mean over 4 output dims)")
    ax.set_title("Accuracy vs |F|: trained on |F| <= F_train, extrapolating beyond")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    # Annotate bin counts above the x-axis
    y_lo, y_hi = ax.get_ylim()
    for c, n in zip(centers, counts):
        ax.text(c, y_lo, f"n={n}", ha="center", va="bottom",
                fontsize=7, color="0.4", rotation=90)
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


# =========================================================================
# 8. MAIN
# =========================================================================
def main():
    print(f"=== Task 3.1 (QUICK={QUICK}) ===")
    print(f"    jax {jax.__version__} on {jax.devices()}   "
          f"max_force={MAX_FORCE}")
    t_start = time.time()

    # --- 1. DATA ---
    data = load_or_gather_data()
    X_tr, F_tr, Y_tr = data["X_tr"], data["F_tr"], data["Y_tr"]
    X_va, F_va, Y_va = data["X_va"], data["F_va"], data["Y_va"]
    X_te, F_te, Y_te = data["X_te"], data["F_te"], data["Y_te"]
    basis_idx        = data["basis_idx"]

    Phi_tr    = to_features_np(X_tr, F_tr)
    Phi_va    = to_features_np(X_va, F_va)
    Phi_te    = to_features_np(X_te, F_te)
    Phi_basis = Phi_tr[basis_idx]
    print(f"[data] N_train={X_tr.shape[0]}  N_val={X_va.shape[0]}  "
          f"N_test={X_te.shape[0]}  M={Phi_basis.shape[0]}  feature_dim={Phi_tr.shape[1]}")
    print(f"       F ranges: train=[{F_tr.min():.1f}, {F_tr.max():.1f}]  "
          f"test=[{F_te.min():.1f}, {F_te.max():.1f}]")

    # --- 2. LINEAR BASELINE ---
    C = fit_linear_features(Phi_tr, Y_tr)
    Y_lin_tr = Phi_tr @ C.T
    Y_lin_va = Phi_va @ C.T
    Y_lin_te = Phi_te @ C.T
    mse_lin_tr = mse_per_dim(Y_tr, Y_lin_tr)
    mse_lin_va = mse_per_dim(Y_va, Y_lin_va)
    mse_lin_te = mse_per_dim(Y_te, Y_lin_te)
    print(f"[linear] train MSE per dim: {mse_lin_tr.round(6)}  total={mse_lin_tr.mean():.4e}")
    print(f"[linear] val   MSE per dim: {mse_lin_va.round(6)}  total={mse_lin_va.mean():.4e}")
    print(f"[linear] test  MSE per dim: {mse_lin_te.round(6)}  total={mse_lin_te.mean():.4e}")

    # --- 3. KERNEL HYPERPARAMETER OPTIMISATION ---
    print("\n=== Hyperparameter optimisation (from-scratch multi-start) ===")
    opt_results = run_optimisation(data)
    check_bounds(opt_results)

    # --- 4. REFIT KERNEL ENSEMBLE ON FULL TRAIN ---
    log_sigmas_opt = np.stack([np.asarray(r["log_hp_opt"][1:])
                               for r in opt_results], axis=0)   # (4, 6)
    log_lambdas_opt = np.asarray([r["log_hp_opt"][0]
                                  for r in opt_results])         # (4,)
    print(f"\n[refit] refitting alpha on full train (N={X_tr.shape[0]}, M={Phi_basis.shape[0]})...")
    t0 = time.time()
    ker_model = fit_ensemble(Phi_tr, Y_tr, Phi_basis,
                             log_sigmas_opt, log_lambdas_opt,
                             label="kernel-opt", basis=CFG["basis"],
                             jitter=CFG["jitter"])
    print(f"[refit] done in {time.time()-t0:.1f}s")

    Y_ker_te = ker_model.predict(X_te, F_te)
    Y_ker_va = ker_model.predict(X_va, F_va)
    Y_ker_tr = ker_model.predict(X_tr, F_tr)
    mse_ker_tr = mse_per_dim(Y_tr, Y_ker_tr)
    mse_ker_va = mse_per_dim(Y_va, Y_ker_va)
    mse_ker_te = mse_per_dim(Y_te, Y_ker_te)
    print(f"[kernel] train MSE per dim: {mse_ker_tr.round(6)}  total={mse_ker_tr.mean():.4e}")
    print(f"[kernel] val   MSE per dim: {mse_ker_va.round(6)}  total={mse_ker_va.mean():.4e}")
    print(f"[kernel] test  MSE per dim: {mse_ker_te.round(6)}  total={mse_ker_te.mean():.4e}")

    # In-distribution test MSE (|F| <= F_train) — the headline number
    mask_in = np.abs(F_te) <= CFG["F_train"]
    if mask_in.any():
        mse_lin_in = mse_per_dim(Y_te[mask_in], Y_lin_te[mask_in])
        mse_ker_in = mse_per_dim(Y_te[mask_in], Y_ker_te[mask_in])
        print(f"[linear] in-dist test MSE (|F|<={CFG['F_train']}):  "
              f"{mse_lin_in.round(6)}  total={mse_lin_in.mean():.4e}  "
              f"(N={int(mask_in.sum())})")
        print(f"[kernel] in-dist test MSE (|F|<={CFG['F_train']}):  "
              f"{mse_ker_in.round(6)}  total={mse_ker_in.mean():.4e}  "
              f"(N={int(mask_in.sum())})")
    else:
        mse_lin_in = mse_lin_te
        mse_ker_in = mse_ker_te

    # --- 5. FIGURES ---
    print("\n=== Figures ===")
    print("[fig] scatter (linear vs kernel)")
    plot_scatter(Y_te, Y_lin_te, Y_ker_te, fig_path("scatter"))
    print("[fig] scans_1d")
    plot_scans_1d(C, ker_model, fig_path("scans_1d"))
    print("[fig] force_scan")
    plot_force_scan(C, ker_model, fig_path("force_scan"))
    print("[fig] contours_2d")
    plot_contours_2d(ker_model, fig_path("contours_2d"))
    print("[fig] rollouts")
    plot_rollouts(C, ker_model, fig_path("rollouts"))
    print("[fig] force_accuracy")
    sweep = compute_force_sweep(C, ker_model, X_te, F_te, Y_te, CFG["sweep_nbins"])
    plot_force_accuracy(sweep, fig_path("force_accuracy"))

    # --- 6. JSON RESULTS ---
    print("\n=== Results JSON ===")
    out = {
        "tag":         FIG_PREFIX,
        "quick":       QUICK,
        "elapsed_total_s": time.time() - t_start,
        "cfg":         {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in CFG.items()},
        "max_force":   MAX_FORCE,
        "feature_order": FEATURE_NAMES,
        "linear": {
            "C_shape":             list(C.shape),
            "train_mse_per_dim":   mse_lin_tr.tolist(),
            "val_mse_per_dim":     mse_lin_va.tolist(),
            "test_mse_per_dim":    mse_lin_te.tolist(),
            "test_mse_total":      float(mse_lin_te.mean()),
            "in_dist_test_mse_per_dim": mse_lin_in.tolist(),
            "in_dist_test_mse_total":   float(mse_lin_in.mean()),
        },
        "kernel": {
            "basis":              CFG["basis"],
            "M":                  int(Phi_basis.shape[0]),
            "train_mse_per_dim":  mse_ker_tr.tolist(),
            "val_mse_per_dim":    mse_ker_va.tolist(),
            "test_mse_per_dim":   mse_ker_te.tolist(),
            "test_mse_total":     float(mse_ker_te.mean()),
            "in_dist_test_mse_per_dim": mse_ker_in.tolist(),
            "in_dist_test_mse_total":   float(mse_ker_in.mean()),
            "sigmas_per_dim":     np.exp(log_sigmas_opt).tolist(),
            "lambdas_per_dim":    np.exp(log_lambdas_opt).tolist(),
            "opt_summary_per_dim": [
                {k: r[k] for k in ("dim", "dim_name", "val_mse_init", "val_mse_opt",
                                   "iters", "n_evals", "success", "elapsed_s",
                                   "improvement_factor", "restart_losses",
                                   "rel_spread_pct", "n_restarts",
                                   "log_hp_init", "log_hp_opt")}
                for r in opt_results
            ],
        },
        "force_sweep": sweep,
    }
    results_path = FIG_DIR / f"{FIG_PREFIX}_results.json"
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[json] saved -> {results_path.name}")

    # --- 7. CONSOLE SUMMARY ---
    print("\n" + "=" * 72)
    print("TASK 3.1 SUMMARY")
    print("=" * 72)
    print(f"\nTest MSE on full test set (|F| up to {CFG['F_test']}):")
    print(f"  {'output':>10s}  {'linear':>12s}  {'kernel':>12s}  "
          f"{'lin/ker':>10s}")
    for k in range(4):
        ratio = mse_lin_te[k] / max(mse_ker_te[k], 1e-30)
        print(f"  {DIM_NAMES[k]:>10s}  {mse_lin_te[k]:>12.4e}  "
              f"{mse_ker_te[k]:>12.4e}  {ratio:>10.1f}x")
    rt = mse_lin_te.mean() / max(mse_ker_te.mean(), 1e-30)
    print(f"  {'TOTAL':>10s}  {mse_lin_te.mean():>12.4e}  "
          f"{mse_ker_te.mean():>12.4e}  {rt:>10.1f}x")

    print(f"\nIn-distribution test MSE (|F| <= {CFG['F_train']}):")
    print(f"  linear total = {mse_lin_in.mean():.4e}")
    print(f"  kernel total = {mse_ker_in.mean():.4e}   "
          f"(linear/kernel = {mse_lin_in.mean()/max(mse_ker_in.mean(), 1e-30):.1f}x)")

    print("\nOptimal hyperparameters (natural units):")
    print(f"  {'dim':>10s}  {'lambda':>11s}  "
          + "  ".join(f"{n:>8s}" for n in
                      ["sig_x", "sig_xdot", "sig_sin", "sig_cos", "sig_thdot", "sig_F"]))
    sig_n = np.exp(log_sigmas_opt)
    lam_n = np.exp(log_lambdas_opt)
    for k in range(4):
        sigs = "  ".join(f"{s:>8.3f}" for s in sig_n[k])
        print(f"  {DIM_NAMES[k]:>10s}  {lam_n[k]:>11.3e}  {sigs}")

    print(f"\nElapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
