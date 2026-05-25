"""
SF3 Machine Learning -- Week 2, Task 2.2
========================================

JAX-based gradient optimisation of kernel-regression hyperparameters.

Pipeline:
  1. Gather/cache 2000 train + 500 val (X, Y) one-step pairs (force=0).
  2. Build JAX kernel: gaussian + sin^2((theta - theta')/2) periodic angle.
  3. Per output dim j in {0..3}, define
         val_mse(log_hp) = MSE on validation set,
     with log_hp = [log_lambda, log_sigma_0..3].
     Compile jax.jit(jax.value_and_grad(val_mse)) once per output.
  4. scipy.optimize.minimize(method='L-BFGS-B', jac=True) for each j, in
     log-space, with sensible bounds.
  5. Compare three model variants on the val set and on iterated rollouts:
       - init       : sigma_j = std(X[:,j]) / (2 for theta), lambda = 1e-4
       - t2.1-grid  : same sigmas, lambda from task2_1_results.json (per tag)
       - opt        : per-output (sigma_j, lambda) from gradient optimisation
  6. Save hyperparams.json, results.json and 7 figures.

CFG (top of file) exposes 4 toggles for experimentation:
  - target : "delta" | "resid"   (predict Y, or Y - X C^T from linear baseline)
  - form   : "NM"    | "NN"      (sparse-subset eq.15, or full Tikhonov)
  - N_train, M

All cached artefacts (data .npz, hyperparams.json, figures) embed
(target, form, N, M) in their filename so flipping the knobs doesn't clobber
earlier runs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import json
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# JAX
import jax
jax.config.update("jax_enable_x64", True)   # 64-bit precision for opt accuracy
import jax.numpy as jnp
from jax import jit, value_and_grad

# scipy
from scipy.optimize import minimize

# Make cartpole.py and task2_1.py importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "week2"))
from cartpole import CartPole, _remap_angle                           # noqa: E402
from task2_1 import (                                                 # noqa: E402
    gram_matrix as np_gram_matrix,
    fit_linear,
    predict_linear,
    mse,
    true_rollout,
    compute_rollout_horizon,
    _angle_wrap,
)


# =========================================================================
# CONFIG -- the four knobs you'll want to experiment with sit at the top.
# =========================================================================
CFG = {
    # --- model knobs ---
    "target":   "delta",        # "delta" (fit Y) | "resid" (fit Y - X C^T)
    "form":     "NM",           # "NM" sparse subset | "NN" full Tikhonov
    "N_train":  2000,
    "N_val":    500,
    "M":        200,            # ignored when form == "NN" (centres = X_train)

    # --- optimisation (lambda in original space; bounds in NATURAL log space) ---
    "lambda_init":       1e-4,                 # log_lambda_init = ln(1e-4) ~= -9.21
    "log_lambda_bounds": (-18.0, -2.0),        # lambda in (~1.5e-8, ~0.135)
    "log_sigma_bounds":  (-6.0,  5.0),         # sigma  in (~0.0025, ~148.4)
    "maxiter":           200,
    "ftol":              1e-10,
    "gtol":              1e-8,

    # --- data / misc ---
    "seed":              0,
    "refit":             False,    # set True to ignore cached hyperparams
    "rollout_steps":     200,
    "dt":                0.1,
    "horizon_threshold": 0.5,
}

ANGLE_IDX = 2
FIG_DIR   = ROOT / "figures"
CACHE_DIR = ROOT / "week2"
FIG_DIR.mkdir(exist_ok=True)

SAMPLE_RANGES = {
    "x":         (-5.0,    5.0),
    "x_dot":     (-10.0,  10.0),
    "theta":     (-np.pi,  np.pi),
    "theta_dot": (-15.0,  15.0),
}
COMPONENT_LABELS = [r"$x$", r"$\dot x$", r"$\theta$", r"$\dot\theta$"]
DELTA_LABELS     = [r"$\Delta x$", r"$\Delta \dot x$",
                    r"$\Delta \theta$", r"$\Delta \dot\theta$"]
DIM_NAMES        = ["x", "xdot", "theta", "thetadot"]


def tag() -> str:
    """Unique tag for filenames; encodes the four experiment knobs."""
    return (f"{CFG['target']}_{CFG['form']}"
            f"_N{CFG['N_train']}_M{CFG['M']}")


def data_cache_path() -> Path:
    return CACHE_DIR / (
        f"task2_2_data_N{CFG['N_train']}_V{CFG['N_val']}_seed{CFG['seed']}.npz"
    )


def hp_cache_path() -> Path:
    return FIG_DIR / f"task2_2_{tag()}_hyperparams.json"


def results_path() -> Path:
    return FIG_DIR / f"task2_2_{tag()}_results.json"


def fig_path(name: str) -> Path:
    return FIG_DIR / f"task2_2_{tag()}_{name}.png"


# =========================================================================
# 1. DATA GATHERING (NumPy) + CACHING
# =========================================================================
def sample_states(n: int, rng: np.random.Generator) -> np.ndarray:
    keys = ("x", "x_dot", "theta", "theta_dot")
    lo = np.array([SAMPLE_RANGES[k][0] for k in keys])
    hi = np.array([SAMPLE_RANGES[k][1] for k in keys])
    return rng.uniform(lo, hi, size=(n, 4))


def gather_dataset(n: int, rng: np.random.Generator
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Generate n (X, Y) pairs with Y = X(T) - X(0), force=0.
    Y[:, 2] is remapped to (-pi, pi] before differencing so it's the actual
    short-time angle change (no wrap)."""
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
    """Load cached train/val pairs (and pick basis subset) or generate afresh.

    Cache file is keyed on (N_train, N_val, seed). Switching M alone reuses
    the data and re-samples the basis indices deterministically from seed+1.
    """
    cache = data_cache_path()
    n_tr, n_va = CFG["N_train"], CFG["N_val"]
    seed = CFG["seed"]

    if cache.exists():
        print(f"[data] reusing {cache.name}")
        d = np.load(cache)
        X_tr, Y_tr = d["X_tr"], d["Y_tr"]
        X_va, Y_va = d["X_va"], d["Y_va"]
        assert X_tr.shape[0] == n_tr, f"cache N_train mismatch: {X_tr.shape[0]} vs {n_tr}"
        assert X_va.shape[0] == n_va, f"cache N_val mismatch: {X_va.shape[0]} vs {n_va}"
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
        "basis_idx": basis_idx,
        "X_basis": X_tr[basis_idx].copy(),
    }


# =========================================================================
# 2. JAX KERNEL + FIT_ALPHA + PREDICT
# =========================================================================
def kernel_matrix(X1, X2, log_sigmas):
    """K[i, j] = K(X1[i], X2[j]) under the periodic Gaussian kernel.
    X1: (n1, 4), X2: (n2, 4), log_sigmas: (4,)
    Periodic substitution: (theta-theta')^2 -> sin^2((theta-theta')/2)."""
    sigmas = jnp.exp(log_sigmas)
    inv2s2 = 1.0 / (2.0 * sigmas ** 2)
    diff = X1[:, None, :] - X2[None, :, :]                  # (n1, n2, 4)
    sq = diff ** 2
    angle_sq = jnp.sin(diff[..., ANGLE_IDX] / 2.0) ** 2
    sq = sq.at[..., ANGLE_IDX].set(angle_sq)
    exponent = -jnp.sum(sq * inv2s2, axis=-1)
    return jnp.exp(exponent)


def _fit_alpha_NM(X_train, Y_col, X_basis, log_lambda, log_sigmas):
    """Sparse subset (eq.15): solve (K_MN K_NM + lam K_MM) alpha = K_MN y."""
    K_NM = kernel_matrix(X_train, X_basis, log_sigmas)      # (N, M)
    K_MM = kernel_matrix(X_basis,  X_basis, log_sigmas)     # (M, M)
    K_MN = K_NM.T                                           # (M, N)
    lam = jnp.exp(log_lambda)
    A = K_MN @ K_NM + lam * K_MM
    b = K_MN @ Y_col
    return jnp.linalg.lstsq(A, b)[0]                        # (M,) -- lstsq per handout


def _fit_alpha_NN(X_train, Y_col, log_lambda, log_sigmas):
    """Full Tikhonov: solve (K_NN + lam I) alpha = y."""
    K_NN = kernel_matrix(X_train, X_train, log_sigmas)      # (N, N)
    lam = jnp.exp(log_lambda)
    N = X_train.shape[0]
    A = K_NN + lam * jnp.eye(N)
    return jnp.linalg.lstsq(A, Y_col)[0]                    # (N,) -- lstsq per handout


def _predict(X_test, centres, alpha, log_sigmas):
    K = kernel_matrix(X_test, centres, log_sigmas)
    return K @ alpha


# =========================================================================
# 3. val_mse + jit(value_and_grad)
# =========================================================================
def make_val_mse_fn(X_tr_j, Y_tr_col, X_va_j, Y_va_col, X_basis_j, form):
    """Build a jit-compiled value_and_grad(val_mse) for a single output dim.

    Returns a callable: log_hp(5,) -> (loss, grad(5,)).

    log_hp ordering: [log_lambda, log_sigma_0, log_sigma_1, log_sigma_2, log_sigma_3]

    The training inputs, basis subset and form are *closed over* so JIT can
    specialise; only the 5-vector log_hp varies between calls.
    """
    if form == "NM":
        centres = X_basis_j

        def val_mse(log_hp):
            log_lambda = log_hp[0]
            log_sigmas = log_hp[1:]
            alpha = _fit_alpha_NM(X_tr_j, Y_tr_col, centres,
                                  log_lambda, log_sigmas)
            preds = _predict(X_va_j, centres, alpha, log_sigmas)
            return jnp.mean((preds - Y_va_col) ** 2)

    elif form == "NN":
        # centres are the full training set for NN
        centres = X_tr_j

        def val_mse(log_hp):
            log_lambda = log_hp[0]
            log_sigmas = log_hp[1:]
            alpha = _fit_alpha_NN(X_tr_j, Y_tr_col,
                                  log_lambda, log_sigmas)
            preds = _predict(X_va_j, centres, alpha, log_sigmas)
            return jnp.mean((preds - Y_va_col) ** 2)

    else:
        raise ValueError(f"unknown form: {form}")

    return jit(value_and_grad(val_mse))


# =========================================================================
# 4. scipy L-BFGS-B WRAPPER
# =========================================================================
def optimise_one_dim(val_and_grad_fn, log_hp_init, bounds):
    """Run L-BFGS-B for one output dim. Returns (scipy_result, loss_trace)."""
    trace = []

    def f_and_g(x_np):
        v, g = val_and_grad_fn(jnp.asarray(x_np))
        v_f = float(v)
        g_np = np.asarray(g, dtype=np.float64)
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
    """Print a one-line warning for each optimised log-hyperparameter that lies
    within 0.5 (in log-space) of either bound. Diagnostic only."""
    log_lo,  log_hi  = CFG["log_lambda_bounds"]
    log_slo, log_shi = CFG["log_sigma_bounds"]
    labels = [("lambda", 0, log_lo, log_hi)] + [
        (f"sigma_{name}", idx + 1, log_slo, log_shi)
        for idx, name in enumerate(DIM_NAMES)
    ]
    for r in opt_results:
        x = r["log_hp_opt"]
        for name, idx, lo, hi in labels:
            v = float(x[idx])
            if abs(v - hi) < 0.5:
                print(f"[check] dim {r['dim']} ({r['dim_name']}): "
                      f"{name} hit upper bound (log={v:.2f}, value={np.exp(v):.3g}); "
                      f"optimum may be higher.")
            elif abs(v - lo) < 0.5:
                print(f"[check] dim {r['dim']} ({r['dim_name']}): "
                      f"{name} near lower bound (log={v:.2f}, value={np.exp(v):.3g}).")


def initial_hyperparams(X_tr) -> np.ndarray:
    """log_hp_init = [log lambda_init, log std(X[:,0]), ..., log std(X[:,3])].
    For theta dim we halve the std to roughly match the sin^2 substitution."""
    sig0 = X_tr.std(axis=0)
    sig0[ANGLE_IDX] = sig0[ANGLE_IDX] / 2.0
    sig0 = np.maximum(sig0, 1e-3)
    log_lambda_init = float(np.log(CFG["lambda_init"]))
    return np.concatenate([[log_lambda_init], np.log(sig0)])


def run_optimisation(data, lin_C):
    """Optimise per-output hyperparams. Returns (results_per_dim, log_hp_init)."""
    X_tr, Y_tr = data["X_tr"], data["Y_tr"]
    X_va, Y_va = data["X_va"], data["Y_va"]
    X_basis    = data["X_basis"]
    form       = CFG["form"]
    target     = CFG["target"]

    # Target transformation: residual vs delta
    if target == "delta":
        Y_tr_use, Y_va_use = Y_tr, Y_va
    elif target == "resid":
        Y_tr_use = Y_tr - X_tr @ lin_C.T
        Y_va_use = Y_va - X_va @ lin_C.T
    else:
        raise ValueError(f"unknown target: {target}")

    log_hp_init = initial_hyperparams(X_tr)
    log_lo,  log_hi  = CFG["log_lambda_bounds"]
    log_slo, log_shi = CFG["log_sigma_bounds"]
    bounds = [(log_lo, log_hi)] + [(log_slo, log_shi)] * 4

    X_tr_j    = jnp.asarray(X_tr)
    X_va_j    = jnp.asarray(X_va)
    X_basis_j = jnp.asarray(X_basis)

    results = []
    for j in range(4):
        Y_tr_col = jnp.asarray(Y_tr_use[:, j])
        Y_va_col = jnp.asarray(Y_va_use[:, j])

        vg = make_val_mse_fn(X_tr_j, Y_tr_col, X_va_j, Y_va_col,
                             X_basis_j, form)

        t_compile = time.time()
        v0, g0 = vg(jnp.asarray(log_hp_init))
        v0 = float(v0); _ = float(jnp.linalg.norm(g0))   # block on device
        print(f"[opt dim {j} {DIM_NAMES[j]:>8s}] compile+eval0 {time.time()-t_compile:.1f}s   "
              f"init val MSE: {v0:.4e}")

        # For dim 0 only: time a second post-compile eval and abort if it's
        # too slow to be practical at the current M. Each L-BFGS-B iteration
        # needs a handful of these calls per output dim.
        if j == 0:
            t_post = time.time()
            v_chk, g_chk = vg(jnp.asarray(log_hp_init))
            _ = float(v_chk); _ = float(jnp.linalg.norm(g_chk))
            post_dt = time.time() - t_post
            print(f"[timing] M={CFG['M']}: post-compile value-and-grad evaluation = {post_dt:.1f} seconds")
            if post_dt > 5.0:
                print(f"[abort] post-compile eval {post_dt:.1f}s > 5s budget at M={CFG['M']}. "
                      f"Reduce CFG['M'] and re-run.")
                sys.exit(1)

        t0 = time.time()
        res, trace = optimise_one_dim(vg, log_hp_init, bounds)
        dt = time.time() - t0
        # Re-evaluate at the returned optimum to get a clean final grad norm.
        _, g_final = vg(jnp.asarray(res.x))
        g_norm = float(jnp.linalg.norm(g_final))
        print(f"                     opt done in {dt:.1f}s   iters={res.nit:3d}   "
              f"final val MSE: {res.fun:.4e}   grad norm: {g_norm:.2e}   "
              f"(factor {v0/max(res.fun,1e-30):.1f}x)")
        print(f"                     log_hp_opt: lam={res.x[0]:.3f}  "
              f"sig=[{res.x[1]:.3f}, {res.x[2]:.3f}, {res.x[3]:.3f}, {res.x[4]:.3f}]")

        results.append({
            "dim":            j,
            "dim_name":       DIM_NAMES[j],
            "log_hp_init":    log_hp_init.tolist(),
            "log_hp_opt":     res.x.tolist(),
            "val_mse_init":   v0,
            "val_mse_opt":    float(res.fun),
            "iters":          int(res.nit),
            "n_evals":        len(trace),
            "success":        bool(res.success),
            "message":        str(res.message),
            "trace":          trace,
            "elapsed_s":      dt,
        })

    return results, log_hp_init.tolist()


# =========================================================================
# 5. ENSEMBLE -- NumPy model for plotting & rollouts
# =========================================================================
@dataclass
class KernelEnsemble:
    """Holds 4 per-output kernel models sharing a single set of centres."""
    label:               str
    centres:             np.ndarray            # (M, 4)
    alphas:              np.ndarray            # (M, 4)
    log_sigmas_per_out:  np.ndarray            # (4, 4)
    log_lambdas:         np.ndarray            # (4,)
    C_baseline:          np.ndarray | None = None  # (4, 4) for resid target

    def predict(self, X):
        n = X.shape[0]
        out = np.zeros((n, 4))
        for j in range(4):
            sig = np.exp(self.log_sigmas_per_out[j])
            K = np_gram_matrix(X, self.centres, sig)
            out[:, j] = K @ self.alphas[:, j]
        if self.C_baseline is not None:
            out = out + X @ self.C_baseline.T
        return out


def model_rollout(ensemble: KernelEnsemble, X0: np.ndarray, n_steps: int):
    """Iterate X_{n+1} = X_n + f(X_n), remap theta into (-pi, pi] each step."""
    traj = np.zeros((n_steps + 1, 4))
    X = X0.copy()
    traj[0] = X
    for t in range(n_steps):
        dX = ensemble.predict(X[None, :])[0]
        X = X + dX
        X[2] = _remap_angle(X[2])
        traj[t + 1] = X
    return traj


def fit_ensemble_numpy(X_tr, Y_tr_use, centres, log_sigmas_per_out,
                       log_lambdas, form):
    """Fit each output dim independently with NumPy lstsq."""
    N = X_tr.shape[0]
    M = centres.shape[0]
    alphas = np.zeros((M, 4))

    if form == "NM":
        for j in range(4):
            sig_j = np.exp(log_sigmas_per_out[j])
            lam_j = float(np.exp(log_lambdas[j]))
            K_NM = np_gram_matrix(X_tr, centres, sig_j)
            K_MM = np_gram_matrix(centres, centres, sig_j)
            K_MN = K_NM.T
            A = K_MN @ K_NM + lam_j * K_MM
            b = K_MN @ Y_tr_use[:, j]
            alpha, *_ = np.linalg.lstsq(A, b, rcond=None)
            alphas[:, j] = alpha

    elif form == "NN":
        assert M == N, "for NN form, centres must equal X_train (M = N)"
        for j in range(4):
            sig_j = np.exp(log_sigmas_per_out[j])
            lam_j = float(np.exp(log_lambdas[j]))
            K_NN = np_gram_matrix(X_tr, X_tr, sig_j)
            A = K_NN + lam_j * np.eye(N)
            alpha, *_ = np.linalg.lstsq(A, Y_tr_use[:, j], rcond=None)
            alphas[:, j] = alpha

    else:
        raise ValueError(form)

    return alphas


def build_ensembles(data, opt_results, lin_C):
    """Construct three KernelEnsemble objects: init, t2.1-grid-best (or None),
    and opt. All use the same set of centres for fair comparison."""
    X_tr, Y_tr = data["X_tr"], data["Y_tr"]
    X_basis    = data["X_basis"]
    form       = CFG["form"]
    target     = CFG["target"]

    if target == "resid":
        Y_tr_use = Y_tr - X_tr @ lin_C.T
        C_b = lin_C
    else:
        Y_tr_use = Y_tr
        C_b = None

    centres = X_basis if form == "NM" else X_tr

    # --- init: shared sigma_j (std), shared lambda = 1e-4 ---
    log_hp_init = initial_hyperparams(X_tr)
    log_sigmas_init = log_hp_init[1:]
    log_lambda_init = log_hp_init[0]
    log_sigmas_init_per_out = np.tile(log_sigmas_init, (4, 1))
    log_lambdas_init        = np.full(4, log_lambda_init)
    alphas_init = fit_ensemble_numpy(
        X_tr, Y_tr_use, centres,
        log_sigmas_init_per_out, log_lambdas_init, form,
    )
    init_model = KernelEnsemble(
        label="init",
        centres=centres,
        alphas=alphas_init,
        log_sigmas_per_out=log_sigmas_init_per_out,
        log_lambdas=log_lambdas_init,
        C_baseline=C_b,
    )

    # --- t2.1: same sigmas as init, lambda from task2_1_results.json ---
    t21_path = FIG_DIR / "task2_1_results.json"
    t21_tag  = f"{target}_{form}"
    t21_model = None
    t21_lambda = None
    if t21_path.exists():
        with open(t21_path) as f:
            t21 = json.load(f)
        try:
            t21_lambda = float(t21["runs"][t21_tag]["lambda_best"])
            log_lambdas_t21 = np.full(4, np.log(t21_lambda))
            alphas_t21 = fit_ensemble_numpy(
                X_tr, Y_tr_use, centres,
                log_sigmas_init_per_out, log_lambdas_t21, form,
            )
            t21_model = KernelEnsemble(
                label=f"t2.1 ($\\lambda$={t21_lambda:.0e})",
                centres=centres,
                alphas=alphas_t21,
                log_sigmas_per_out=log_sigmas_init_per_out,
                log_lambdas=log_lambdas_t21,
                C_baseline=C_b,
            )
            print(f"[t2.1] using lambda_best = {t21_lambda:.3e} from {t21_path.name} [tag={t21_tag}]")
        except KeyError:
            print(f"[t2.1] tag '{t21_tag}' not found in {t21_path.name} -- skipping baseline")
    else:
        print(f"[t2.1] {t21_path.name} not found -- skipping baseline")

    # --- opt: per-output hyperparams from L-BFGS-B ---
    log_sigmas_opt_per_out = np.stack([np.asarray(r["log_hp_opt"][1:])
                                       for r in opt_results], axis=0)  # (4, 4)
    log_lambdas_opt        = np.asarray([r["log_hp_opt"][0]
                                         for r in opt_results])        # (4,)
    alphas_opt = fit_ensemble_numpy(
        X_tr, Y_tr_use, centres,
        log_sigmas_opt_per_out, log_lambdas_opt, form,
    )
    opt_model = KernelEnsemble(
        label="opt",
        centres=centres,
        alphas=alphas_opt,
        log_sigmas_per_out=log_sigmas_opt_per_out,
        log_lambdas=log_lambdas_opt,
        C_baseline=C_b,
    )

    return init_model, t21_model, opt_model, t21_lambda


# =========================================================================
# 6. PLOTS
# =========================================================================
COLOURS = {"init": "tab:blue", "t2.1": "tab:green", "opt": "tab:red"}


def plot_optim_traces(opt_results, savepath):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
    for j, r in enumerate(opt_results):
        ax = axes[j // 2, j % 2]
        ax.semilogy(r["trace"], "-", lw=1.5, color="C3")
        ax.axhline(r["val_mse_init"], color="gray", ls="--", lw=1,
                   label=f"init={r['val_mse_init']:.2e}")
        ax.axhline(r["val_mse_opt"], color="C3", ls=":", lw=1,
                   label=f"opt={r['val_mse_opt']:.2e}")
        ax.set_title(f"dim {j}: {DELTA_LABELS[j]}   "
                     f"({r['iters']} iters, {r['n_evals']} evals)")
        ax.set_xlabel("L-BFGS-B function eval index")
        ax.set_ylabel("val MSE")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.4)
    fig.suptitle(f"L-BFGS-B optimisation traces  [{tag()}]")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_hyperparam_table(opt_results, log_hp_init, t21_lambda, savepath):
    """Show init / t2.1 / opt hyperparameters with each sigma_j in its own column.
    Three sub-blocks (init / t2.1 / opt) get distinct cell backgrounds to make
    the grouping visible at a glance."""
    fig, ax = plt.subplots(figsize=(20, 3.6))
    ax.axis("off")

    sig_init = np.exp(log_hp_init[1:])
    lam_init = float(np.exp(log_hp_init[0]))

    sigma_subs = [r"x", r"\dot x", r"\theta", r"\dot\theta"]

    def sigma_headers(suffix):
        return [rf"$\sigma_{{{s},\rm {suffix}}}$" for s in sigma_subs]

    cols = (
        ["dim"]
        + [r"$\lambda_{\rm init}$"] + sigma_headers("init")
        + [r"$\lambda_{\rm t2.1}$"] + sigma_headers("t2.1")
        + [r"$\lambda_{\rm opt}$"]  + sigma_headers("opt")
        + ["val MSE init / opt"]
    )

    # Block colour scheme: init=light grey, t2.1=white, opt=light blue.
    BG_INIT = "#f0f0f0"
    BG_T21  = "#ffffff"
    BG_OPT  = "#e6eef7"
    BG_END  = "#ffffff"

    # col index -> block bg. Column 0 (dim) stays white.
    block_bg = (
        [BG_END]
        + [BG_INIT] * 5
        + [BG_T21]  * 5
        + [BG_OPT]  * 5
        + [BG_END]
    )

    rows = []
    for j, r in enumerate(opt_results):
        sig_opt = np.exp(r["log_hp_opt"][1:])
        lam_opt = float(np.exp(r["log_hp_opt"][0]))
        sig_t21 = sig_init  # identical sigmas to init
        lam_t21 = t21_lambda if t21_lambda is not None else float("nan")
        lam_t21_str = f"{lam_t21:.2e}" if t21_lambda is not None else "n/a"
        sig_t21_strs = ([f"{s:.3g}" for s in sig_t21]
                        if t21_lambda is not None else ["n/a"] * 4)
        rows.append(
            [DIM_NAMES[j]]
            + [f"{lam_init:.2e}"] + [f"{s:.3g}" for s in sig_init]
            + [lam_t21_str]       + sig_t21_strs
            + [f"{lam_opt:.2e}"]  + [f"{s:.3g}" for s in sig_opt]
            + [f"{r['val_mse_init']:.2e} / {r['val_mse_opt']:.2e}"]
        )

    table = ax.table(cellText=rows, colLabels=cols,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    n_rows = len(rows) + 1   # +1 for header
    n_cols = len(cols)
    for (row_i, col_i), cell in table.get_celld().items():
        bg = block_bg[col_i]
        # Make headers slightly darker for legibility.
        if row_i == 0:
            cell.set_facecolor(bg)
            cell.set_text_props(weight="bold", fontsize=8)
        else:
            cell.set_facecolor(bg)
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.4)
        # Thicker vertical edges between blocks
        if col_i in (0, 5, 10, 15):
            cell.visible_edges = "BLR" if row_i > 0 else "BTLR"

    # Width tuning: dim narrow, lambda medium, sigma narrow, MSE wide.
    widths = (
        [0.04]            # dim
        + [0.07] + [0.05] * 4   # init block
        + [0.07] + [0.05] * 4   # t2.1 block
        + [0.07] + [0.05] * 4   # opt block
        + [0.12]                 # MSE
    )
    for col_i, w in enumerate(widths):
        for row_i in range(n_rows):
            table[(row_i, col_i)].set_width(w)

    table.scale(1.0, 1.6)

    ax.set_title(
        f"Hyperparameters per output dim  [{tag()}]   "
        r"  init/t2.1 share $\sigma_j$ = std(X); only $\lambda$ differs",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_true(Y_va, ens_t21, ens_opt, X_va, savepath):
    """4 cols (output dim) x N rows (model variants). Identity line + MSE.
    init is dropped here -- it numerically overlays t2.1 (same sigmas, only
    lambda differs at this data scale)."""
    variants = []
    if ens_t21 is not None:
        variants.append((ens_t21, "t2.1"))
    variants.append((ens_opt, "opt"))
    n_rows = len(variants)

    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for r, (ens, name) in enumerate(variants):
        Y_pred = ens.predict(X_va)
        for k in range(4):
            ax = axes[r, k]
            ax.scatter(Y_va[:, k], Y_pred[:, k], s=3, alpha=0.35,
                       color=COLOURS.get(name.split(" ")[0], "C0"))
            lo = min(Y_va[:, k].min(), Y_pred[:, k].min())
            hi = max(Y_va[:, k].max(), Y_pred[:, k].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            mse_k = float(np.mean((Y_va[:, k] - Y_pred[:, k]) ** 2))
            ax.set_title(f"{DELTA_LABELS[k]}  MSE={mse_k:.2e}", fontsize=9)
            if k == 0:
                ax.set_ylabel(f"{name}\npred")
            if r == n_rows - 1:
                ax.set_xlabel("true")
            ax.set_aspect("equal", "box")

    fig.suptitle(f"Pred vs true on validation set  [{tag()}]")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def _true_dY_for_states(states):
    sim = CartPole(visual=False)
    out = np.zeros_like(states)
    for i in range(states.shape[0]):
        sim.setState(states[i].tolist())
        sim.performAction(0.0)
        dY = sim.getState() - states[i]
        dY[2] = _remap_angle(dY[2])
        out[i] = dY
    return out


def plot_1d_scans(ens_t21, ens_opt, savepath):
    """Vary theta in [-pi, pi] at fixed (x=0, x_dot=0, theta_dot=0). 4 panels,
    each panel plots one output dim with true / t2.1 / opt curves.
    init is dropped here -- it overlays t2.1."""
    n = 200
    thetas = np.linspace(-np.pi, np.pi, n)
    base = np.zeros((n, 4))
    base[:, 2] = thetas

    Y_true = _true_dY_for_states(base)
    Y_opt  = ens_opt.predict(base)
    Y_t21  = ens_t21.predict(base) if ens_t21 is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for j in range(4):
        ax = axes[j // 2, j % 2]
        ax.plot(thetas, Y_true[:, j], "k-",  lw=2.0, label="true")
        if Y_t21 is not None:
            ax.plot(thetas, Y_t21[:, j], "--",  color=COLOURS["t2.1"], lw=1.4, label="t2.1")
        ax.plot(thetas, Y_opt[:, j],  "--",  color=COLOURS["opt"],  lw=1.6, label="opt")
        ax.set_title(f"{DELTA_LABELS[j]}  vs  $\\theta$  (others=0)")
        if j // 2 == 1:
            ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(DELTA_LABELS[j])
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle(f"1D scans: $\\theta$-sweep at origin  [{tag()}]")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_2d_contours(ens_t21, ens_opt, savepath, out_dim=3):
    """2D slice (theta, theta_dot) for one chosen output dim. Rows = (pred,
    pred-true); cols = (t2.1, opt). True shown as colour scale on row 1.
    init is dropped here -- it overlays t2.1."""
    n_grid = 50
    thetas = np.linspace(-np.pi, np.pi, n_grid)
    tdots  = np.linspace(-15.0,  15.0, n_grid)
    T, TD = np.meshgrid(thetas, tdots, indexing="xy")
    flat = np.zeros((n_grid * n_grid, 4))
    flat[:, 2] = T.ravel()
    flat[:, 3] = TD.ravel()

    Y_true = _true_dY_for_states(flat)[:, out_dim].reshape(n_grid, n_grid)
    preds = {
        "opt":  ens_opt.predict(flat)[:, out_dim].reshape(n_grid, n_grid),
    }
    if ens_t21 is not None:
        preds["t2.1"] = ens_t21.predict(flat)[:, out_dim].reshape(n_grid, n_grid)

    names = (["t2.1"] if ens_t21 is not None else []) + ["opt"]

    vmin = min(Y_true.min(), min(p.min() for p in preds.values()))
    vmax = max(Y_true.max(), max(p.max() for p in preds.values()))
    levels = np.linspace(vmin, vmax, 21)
    absmax = max(np.max(np.abs(preds[n] - Y_true)) for n in names)
    diff_levels = np.linspace(-absmax, absmax, 21)

    fig, axes = plt.subplots(2, len(names) + 1, figsize=(4.0 * (len(names) + 1), 8))
    # First column: true reference (and a blank below for visual symmetry)
    cs0 = axes[0, 0].contourf(T, TD, Y_true, levels=levels, cmap="viridis")
    fig.colorbar(cs0, ax=axes[0, 0])
    axes[0, 0].set_title(f"true {DELTA_LABELS[out_dim]}")
    axes[0, 0].set_ylabel(r"$\dot\theta$")
    axes[1, 0].axis("off")

    for c, name in enumerate(names):
        Z = preds[name]
        D = Z - Y_true
        cs1 = axes[0, c + 1].contourf(T, TD, Z, levels=levels, cmap="viridis")
        fig.colorbar(cs1, ax=axes[0, c + 1])
        axes[0, c + 1].set_title(f"{name}: {DELTA_LABELS[out_dim]}")
        cs2 = axes[1, c + 1].contourf(T, TD, D, levels=diff_levels, cmap="RdBu_r")
        fig.colorbar(cs2, ax=axes[1, c + 1])
        axes[1, c + 1].set_title(f"{name} - true   max|.|={np.max(np.abs(D)):.2e}")
        axes[1, c + 1].set_xlabel(r"$\theta$")
        if c == 0:
            axes[1, c + 1].set_ylabel(r"$\dot\theta$")

    fig.suptitle(f"2D ($\\theta$, $\\dot\\theta$) slice of {DELTA_LABELS[out_dim]}  [{tag()}]")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_rollouts(ens_t21, ens_opt, ICs, ic_labels, n_steps, dt, savepath):
    """N_ic rows x 4 cols. Each cell overlays true, t2.1, opt trajectories
    for one state component. init is dropped -- it overlays t2.1."""
    n_ic = len(ICs)
    fig, axes = plt.subplots(n_ic, 4, figsize=(16, 3.0 * n_ic), sharex=True)
    if n_ic == 1:
        axes = axes[None, :]
    t = np.arange(n_steps + 1) * dt

    for r, X0 in enumerate(ICs):
        tr_true = true_rollout(X0, n_steps)
        tr_opt  = model_rollout(ens_opt,  X0, n_steps)
        tr_t21  = model_rollout(ens_t21,  X0, n_steps) if ens_t21 is not None else None
        for k in range(4):
            ax = axes[r, k]
            ax.plot(t, tr_true[:, k], "k-", lw=1.6, label="true")
            if tr_t21 is not None:
                ax.plot(t, tr_t21[:, k], "--", color=COLOURS["t2.1"], lw=1.1, label="t2.1")
            ax.plot(t, tr_opt[:, k],  "--", color=COLOURS["opt"],  lw=1.3, label="opt")
            if r == 0:
                ax.set_title(COMPONENT_LABELS[k])
            if k == 0:
                ax.set_ylabel(ic_labels[r], fontsize=8)
            if r == n_ic - 1:
                ax.set_xlabel("time / s")
            if r == 0 and k == 0:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Iterated rollouts: true vs (t2.1, opt)  [{tag()}]")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_rollout_horizon(ens_t21, ens_opt, ICs, ic_labels,
                         sigmas_state, n_steps, dt, threshold, savepath):
    """For each IC, plot log-error vs t for t2.1 / opt. T_break markers.
    init is dropped from this plot (and from the returned summaries) because
    init and t2.1 share sigmas and their error curves overlay visually."""
    n_ic = len(ICs)
    n_cols = 3
    n_rows = int(np.ceil(n_ic / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.0 * n_rows),
                             sharex=True, sharey=True)
    axes_flat = axes.ravel() if n_ic > 1 else [axes]

    summaries = {"t2.1": [], "opt": []}

    for r, X0 in enumerate(ICs):
        ax = axes_flat[r]
        tr_true = true_rollout(X0, n_steps)
        period_s = _estimate_period_from(tr_true, dt)
        for label, ens, colour in [
            ("t2.1", ens_t21,  COLOURS["t2.1"]),
            ("opt",  ens_opt,  COLOURS["opt"]),
        ]:
            if ens is None:
                continue
            tr_m = model_rollout(ens, X0, n_steps)
            diff = tr_m - tr_true
            diff[:, 2] = _angle_wrap(diff[:, 2])
            err = np.sqrt(np.sum((diff / sigmas_state) ** 2, axis=1))
            tarr = np.arange(n_steps + 1) * dt
            above = err > threshold
            t_break = float(tarr[int(np.argmax(above))]) if above.any() else float("inf")
            if np.isfinite(t_break) and np.isfinite(period_s) and period_s > 0:
                cycles = t_break / period_s
            else:
                cycles = None
            if cycles is None:
                cyc_str = "fixed point" if not np.isfinite(period_s) else "cyc~n/a"
            else:
                cyc_str = f"cyc~{cycles:.1f}"
            tb_str = f"T={t_break:.1f}s, {cyc_str}"
            ax.semilogy(tarr, np.maximum(err, 1e-6),
                        "-", color=colour, lw=1.4,
                        label=f"{label} [{tb_str}]")
            if np.isfinite(t_break):
                ax.axvline(t_break, color=colour, ls=":", lw=1, alpha=0.6)
            summaries[label].append({
                "IC":              ic_labels[r].splitlines()[0],
                "T_break_s":       t_break if np.isfinite(t_break) else None,
                "period_s":        period_s if np.isfinite(period_s) else None,
                "cycles_at_break": cycles,
            })
        ax.axhline(threshold, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_title(ic_labels[r].splitlines()[0], fontsize=9)
        ax.set_xlabel("time / s")
        ax.set_ylabel(r"$\|(X_m - X_t)/\sigma\|_2$")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=7, loc="lower right")

    for r in range(n_ic, n_rows * n_cols):
        axes_flat[r].axis("off")

    fig.suptitle(f"Rollout horizon (threshold={threshold})  [{tag()}]")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)

    return summaries


def _estimate_period_from(traj_t, dt):
    """Same logic as task2_1._estimate_period, inlined here so we don't depend
    on a private name. Prints a diagnosis line so the IC's dominant motion is
    visible in the console."""
    theta_dot = traj_t[:, 3]
    om_mean = float(np.mean(theta_dot))
    om_std  = float(np.std(theta_dot))
    if abs(om_mean) > 0.5 and om_std < 0.7 * abs(om_mean):
        period = 2.0 * np.pi / abs(om_mean)
        print(f"[period] theta_dot mean={om_mean:.2f}, std={om_std:.2f} "
              f"-> rotation (period 2pi/|theta_dot| = {period:.2f}s)")
        return period
    theta_u = np.unwrap(traj_t[:, 2])
    n = len(theta_u)
    tarr = np.arange(n) * dt
    if np.std(theta_u) > 1e-3:
        slope, intercept = np.polyfit(tarr, theta_u, 1)
        detrended = theta_u - (slope * tarr + intercept)
        if np.std(detrended) > 1e-3:
            fft_v = np.fft.rfft(detrended)
            freqs = np.fft.rfftfreq(n, dt)
            if len(freqs) > 1:
                idx = 1 + int(np.argmax(np.abs(fft_v[1:])))
                f_dom = freqs[idx]
                if f_dom > 0:
                    period = 1.0 / f_dom
                    print(f"[period] theta_dot mean={om_mean:.2f}, std={om_std:.2f} "
                          f"-> oscillation (FFT-dominant period = {period:.2f}s)")
                    return period
    print(f"[period] theta_dot mean={om_mean:.2f}, std={om_std:.2f} "
          f"-> fixed point or no detectable period (returning inf)")
    return float("inf")


# =========================================================================
# 7. MAIN
# =========================================================================
def main():
    print(f"=== Task 2.2 : {tag()} ===")
    print(f"    target={CFG['target']}  form={CFG['form']}  "
          f"N_train={CFG['N_train']}  N_val={CFG['N_val']}  M={CFG['M']}")
    print(f"    jax {jax.__version__} on {jax.devices()}")

    # Data
    data = load_or_gather_data()
    sigmas_state = data["X_tr"].std(axis=0)

    # Linear baseline (needed for resid target; cheap to fit either way)
    lin_C = fit_linear(data["X_tr"], data["Y_tr"])

    # Optimisation: load hyperparams cache if present (and refit==False)
    hp_path = hp_cache_path()
    if hp_path.exists() and not CFG["refit"]:
        print(f"[opt] reusing {hp_path.name}")
        with open(hp_path) as f:
            blob = json.load(f)
        opt_results = blob["results"]
        log_hp_init = blob["log_hp_init"]
    else:
        opt_results, log_hp_init = run_optimisation(data, lin_C)
        with open(hp_path, "w") as f:
            json.dump({
                "tag":          tag(),
                "cfg":          {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in CFG.items()},
                "log_hp_init":  log_hp_init,
                "results":      opt_results,
            }, f, indent=2)
        print(f"[opt] saved -> {hp_path.name}")

    # Diagnostic: warn for any hyperparameter that hit a bound.
    check_bounds(opt_results)

    # Build the three model ensembles (init, t2.1, opt) sharing centres
    init_model, t21_model, opt_model, t21_lambda = build_ensembles(
        data, opt_results, lin_C,
    )

    # Validation MSE summary (re-evaluated through the NumPy predict path)
    X_va, Y_va = data["X_va"], data["Y_va"]
    summary = {"init": {}, "t2.1": {}, "opt": {}}
    for label, ens in [("init", init_model),
                       ("t2.1", t21_model),
                       ("opt",  opt_model)]:
        if ens is None:
            summary[label] = None
            continue
        Y_pred = ens.predict(X_va)
        per_c = np.mean((Y_va - Y_pred) ** 2, axis=0)
        summary[label] = {
            "val_mse_total":     float(np.mean(per_c)),
            "val_mse_per_dim":   per_c.tolist(),
        }

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    print("[fig] optim traces ...")
    plot_optim_traces(opt_results, fig_path("optim_traces"))

    print("[fig] hyperparam table ...")
    plot_hyperparam_table(opt_results, np.asarray(log_hp_init),
                          t21_lambda, fig_path("hyperparam_table"))

    print("[fig] pred vs true ...")
    plot_pred_vs_true(Y_va, t21_model, opt_model, X_va,
                      fig_path("pred_vs_true_before_after"))

    print("[fig] 1d scans ...")
    plot_1d_scans(t21_model, opt_model, fig_path("1d_scans"))

    # Pick the output dim with the largest factor improvement for the 2D plot
    improvements = [(r["val_mse_init"] / max(r["val_mse_opt"], 1e-30), j)
                    for j, r in enumerate(opt_results)]
    improvements.sort(reverse=True)
    best_dim = improvements[0][1]
    print(f"[fig] 2d contours (dim {best_dim}: {DIM_NAMES[best_dim]}, "
          f"factor {improvements[0][0]:.1f}x) ...")
    plot_2d_contours(t21_model, opt_model,
                     fig_path("2d_contours"), out_dim=best_dim)

    rollout_ICs = [
        np.array([0.0, 0.0, np.pi,       0.0]),
        np.array([0.0, 0.0, np.pi - 0.5, 0.0]),
        np.array([0.0, 0.5, 0.5,         0.0]),
        np.array([0.0, 0.0, 0.1,         3.0]),
        np.array([1.0, 1.0, -1.5,        0.0]),
    ]
    IC_LABELS = [
        "IC1: pole down, still\nX=[0, 0, $\\pi$, 0]",
        "IC2: small swing from down\nX=[0, 0, $\\pi$-0.5, 0]",
        "IC3: mid-swing + cart drift\nX=[0, 0.5, 0.5, 0]",
        "IC4: near upright + rapid spin\nX=[0, 0, 0.1, 3.0]",
        "IC5: off-centre, downswing\nX=[1, 1, -1.5, 0]",
    ]

    print("[fig] rollouts ...")
    plot_rollouts(t21_model, opt_model,
                  rollout_ICs, IC_LABELS,
                  CFG["rollout_steps"], CFG["dt"],
                  fig_path("rollouts_before_after"))

    print("[fig] rollout horizon ...")
    horizon = plot_rollout_horizon(
        t21_model, opt_model,
        rollout_ICs, IC_LABELS, sigmas_state,
        CFG["rollout_steps"], CFG["dt"], CFG["horizon_threshold"],
        fig_path("rollout_horizon"),
    )

    # -----------------------------------------------------------------
    # results.json
    # -----------------------------------------------------------------
    out = {
        "tag":             tag(),
        "cfg":             {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in CFG.items()},
        "log_hp_init":     log_hp_init,
        "t21_lambda":      t21_lambda,
        "opt_results":     [{k: v for k, v in r.items() if k != "trace"}
                            for r in opt_results],   # drop bulky traces
        "val_mse_summary": summary,
        "rollout_horizon": horizon,
        "sigmas_state":    sigmas_state.tolist(),
    }
    with open(results_path(), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] -> {results_path().name}")

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------
    print("\n=== Validation MSE summary ===")
    for label in ("init", "t2.1", "opt"):
        s = summary[label]
        if s is None:
            print(f"  {label:5s}: n/a")
            continue
        per_c = "  ".join(f"{x:.2e}" for x in s["val_mse_per_dim"])
        print(f"  {label:5s}: total={s['val_mse_total']:.4e}   per-dim=[{per_c}]")

    print("\n=== Optimal sigma_j and lambda per output (original space) ===")
    for r in opt_results:
        sig = np.exp(r["log_hp_opt"][1:])
        lam = float(np.exp(r["log_hp_opt"][0]))
        print(f"  dim {r['dim']} {r['dim_name']:>8s}: "
              f"lam={lam:.2e}   sig=[{sig[0]:.3f}, {sig[1]:.3f}, "
              f"{sig[2]:.3f}, {sig[3]:.3f}]   "
              f"val MSE {r['val_mse_init']:.2e} -> {r['val_mse_opt']:.2e}  "
              f"({r['val_mse_init']/max(r['val_mse_opt'],1e-30):.1f}x)")

    print("\n=== Rollout horizon (threshold=0.5 normalised state-error units) ===")
    for label in ("t2.1", "opt"):
        if horizon.get(label) is None or len(horizon[label]) == 0:
            continue
        print(f"  [{label}]")
        for row in horizon[label]:
            tb = "n/a" if row["T_break_s"] is None else f"{row['T_break_s']:.2g} s"
            pe = "n/a" if row["period_s"]  is None else f"{row['period_s']:.2g} s"
            cy = "n/a" if row["cycles_at_break"] is None else f"{row['cycles_at_break']:.1f}"
            print(f"    {row['IC']:36s}  T_break={tb:>8s}   period={pe:>8s}   cycles={cy}")


if __name__ == "__main__":
    main()
