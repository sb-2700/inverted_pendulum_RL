"""
SF3 Machine Learning -- Week 2, Task 2.1
========================================

Single-script implementation of nonlinear modelling of the cartpole
one-step dynamics using kernel regression with a Gaussian basis and
a periodic angle.

Pipeline:
  1. Gather (X, Y) one-step pairs, split into train/val/test.
  2. Fit a linear baseline Y ~= C X (Week 1 model) for residual targets.
  3. Fit kernel models in a 2x2 matrix of choices:
       target  in {delta = Y, resid = Y - C X}
       form    in {NN (full Tikhonov), NM (sparse subset)}
  4. Choose lambda by validation MSE (log grid 1e-6 .. 1e-1).
  5. Produce a compact set of 11 figures covering all 4 (target, form)
     combos. Where natural, multiple tags are combined into one figure:
       - pred_vs_true            : 4 rows (tags) x 4 cols (components).
       - convergence             : 2x2 panels (vs N for NN, vs M for NM,
                                   each target on its own row).
       - 2d_slices_<tag>         : 4 files, each 3 slices x {true, model,
                                   model-true} contours.
       - rollouts_<tag>          : 4 files, each 4 labelled ICs x 4 state
                                   components over 20 s.
       - rollout_horizon         : 2x2 panels, one per tag, each showing
                                   the 4 ICs' error-vs-time with their
                                   breakdown time and cycles annotated.
  6. Save figures into <repo>/figures/ as task2_1_<name>.png and a
     comparison table into <repo>/figures/task2_1_results.json.

Maths summary
-------------
State X = [x, x_dot, theta, theta_dot], theta periodic on (-pi, pi].
Per-component model:
    f_k(X) = sum_{i=1..M} alpha_{k,i} K(X, X_i),   k = 0..3

Periodic Gaussian kernel:
    K(X^a, X^b) = exp( - sum_{j!=2} (X^a_j - X^b_j)^2 / (2 sigma_j^2)
                       - sin^2((theta^a - theta^b)/2) / (2 sigma_theta^2) )

Two regression forms:
    NN:  alpha = (K_NN + lam I)^{-1} Y
    NM:  alpha = (K_MN K_NM + lam K_MM)^{-1} K_MN Y
Both solved via np.linalg.lstsq (NEVER use np.linalg.inv).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no interactive backend required
import matplotlib.pyplot as plt

# Make ``cartpole.py`` importable regardless of the working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cartpole import CartPole, _remap_angle  # noqa: E402


# =========================================================================
# CONFIG -- edit here. Defaults run in a few minutes on a modern laptop.
# =========================================================================
CFG = {
    "seed":            0,
    "N_total":         3000,
    "frac":            (0.70, 0.15, 0.15),
    "headline_M":      400,
    "lambda_grid":     np.logspace(-6, -1, 11),
    "M_grid":          [10, 20, 40, 80, 160, 320, 640],
    "N_grid":          [100, 200, 400, 800, 1600],
    "rollout_steps":   200,
    "dt":              0.1,         # CartPole.delta_time
}

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Sampling ranges for initial states (per Hint 1 in the handout)
SAMPLE_RANGES = {
    "x":         (-5.0,    5.0),
    "x_dot":     (-10.0,  10.0),
    "theta":     (-np.pi,  np.pi),
    "theta_dot": (-15.0,  15.0),
}

COMPONENT_LABELS = [r"$x$", r"$\dot x$", r"$\theta$", r"$\dot\theta$"]
DELTA_LABELS = [r"$\Delta x$", r"$\Delta \dot x$",
                r"$\Delta \theta$", r"$\Delta \dot\theta$"]


# =========================================================================
# 1. DATA GATHERING
# =========================================================================
def sample_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample of n states from SAMPLE_RANGES. Returns (n, 4)."""
    keys = ("x", "x_dot", "theta", "theta_dot")
    lo = np.array([SAMPLE_RANGES[k][0] for k in keys])
    hi = np.array([SAMPLE_RANGES[k][1] for k in keys])
    return rng.uniform(lo, hi, size=(n, 4))


def gather_dataset(n: int, rng: np.random.Generator
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Gather n (X, Y) pairs with Y = X(T) - X(0) for one performAction call
    (no force). The output angle change is remapped to (-pi, pi] so that
    Y_theta is the actual short-time change and doesn't wrap.
    """
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


def split_dataset(X, Y, fractions, rng):
    """Random train/val/test split. Returns dict of (X, Y) tuples."""
    n = X.shape[0]
    idx = rng.permutation(n)
    n_tr = int(round(fractions[0] * n))
    n_va = int(round(fractions[1] * n))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    return {
        "train": (X[tr], Y[tr]),
        "val":   (X[va], Y[va]),
        "test":  (X[te], Y[te]),
    }


# =========================================================================
# 2. LINEAR BASELINE (Week 1)
# =========================================================================
def fit_linear(X, Y):
    """Solve Y = X C^T by least squares. Returns C of shape (4, 4).
    Uses lstsq because np.linalg.inv must never be used (handout Note 1)."""
    CT, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return CT.T


def predict_linear(C, X):
    return X @ C.T


# =========================================================================
# 3. KERNEL FUNCTION (periodic in theta)
# =========================================================================
def gram_matrix(Xa, Xb, sigmas):
    """
    K[i, j] = K(Xa[i], Xb[j]).  Xa: (Na, 4)  Xb: (Nb, 4)  sigmas: (4,)
    Returns (Na, Nb).
    The angle dimension (index 2) uses sin^2((da - db)/2) instead of
    (da - db)^2 so the kernel is 2*pi-periodic in theta.
    """
    diff = Xa[:, None, :] - Xb[None, :, :]         # (Na, Nb, 4)
    sq = diff ** 2
    sq[..., 2] = np.sin(diff[..., 2] / 2.0) ** 2   # periodic angle
    inv_2sig2 = 1.0 / (2.0 * sigmas ** 2)
    exponent = -np.sum(sq * inv_2sig2, axis=-1)
    return np.exp(exponent)


def default_sigmas_from_data(X_train):
    """sigma_j = std of component j (handout suggestion).
    For theta we halve it to crudely match the small-angle behaviour of the
    sin^2 substitution, which compresses angle distances into [0, 1]."""
    s = X_train.std(axis=0)
    s[2] = s[2] / 2.0
    return np.maximum(s, 1e-3)


# =========================================================================
# 4. NONLINEAR KERNEL MODEL
# =========================================================================
@dataclass
class KernelModel:
    centres: np.ndarray
    alpha:   np.ndarray
    sigmas:  np.ndarray
    lam:     float
    form:    str               # "NN" or "NM"
    C_baseline: np.ndarray | None = None

    def predict(self, X):
        K = gram_matrix(X, self.centres, self.sigmas)
        Y_hat = K @ self.alpha
        if self.C_baseline is not None:
            Y_hat = Y_hat + X @ self.C_baseline.T
        return Y_hat


def _lstsq_solve(A, b):
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


def fit_NN(X_train, Y_train, sigmas, lam, C_baseline=None):
    """Full N x N Tikhonov: (K_NN + lam I) alpha = Y (or Y - CX)."""
    N = X_train.shape[0]
    K_NN = gram_matrix(X_train, X_train, sigmas)
    A = K_NN + lam * np.eye(N)
    target = Y_train if C_baseline is None else Y_train - X_train @ C_baseline.T
    alpha = _lstsq_solve(A, target)
    return KernelModel(
        centres=X_train.copy(), alpha=alpha, sigmas=sigmas.copy(),
        lam=lam, form="NN", C_baseline=C_baseline,
    )


def fit_NM(X_train, Y_train, sigmas, lam, M, rng, C_baseline=None):
    """Sparse subset: (K_MN K_NM + lam K_MM) alpha = K_MN Y."""
    N = X_train.shape[0]
    if M > N:
        raise ValueError(f"M={M} cannot exceed N={N}")
    centre_idx = rng.choice(N, size=M, replace=False)
    centres = X_train[centre_idx]
    K_NM = gram_matrix(X_train, centres, sigmas)
    K_MM = gram_matrix(centres, centres, sigmas)
    K_MN = K_NM.T
    A = K_MN @ K_NM + lam * K_MM
    target = Y_train if C_baseline is None else Y_train - X_train @ C_baseline.T
    b = K_MN @ target
    alpha = _lstsq_solve(A, b)
    return KernelModel(
        centres=centres, alpha=alpha, sigmas=sigmas.copy(),
        lam=lam, form="NM", C_baseline=C_baseline,
    )


def mse(Y_true, Y_pred, per_component=False):
    err2 = (Y_true - Y_pred) ** 2
    return err2.mean(axis=0) if per_component else err2.mean()


def normalised_mse(Y_true, Y_pred):
    return mse(Y_true, Y_pred, per_component=True) / (Y_true.var(axis=0) + 1e-12)


# =========================================================================
# 5. ROLLOUTS
# =========================================================================
def true_rollout(X0, n_steps):
    sim = CartPole(visual=False)
    sim.setState(X0.tolist())
    traj = np.zeros((n_steps + 1, 4))
    traj[0] = sim.getState()
    for t in range(n_steps):
        sim.performAction(0.0)
        traj[t + 1] = sim.getState()
    for t in range(n_steps + 1):
        traj[t, 2] = _remap_angle(traj[t, 2])
    return traj


def model_rollout(model, X0, n_steps):
    """Iterate X_{n+1} = X_n + f(X_n), remapping theta into (-pi, pi]
    each step so the kernel doesn't see out-of-distribution angles."""
    traj = np.zeros((n_steps + 1, 4))
    X = X0.copy()
    traj[0] = X
    for t in range(n_steps):
        dX = model.predict(X[None, :])[0]
        X = X + dX
        X[2] = _remap_angle(X[2])
        traj[t + 1] = X
    return traj


# =========================================================================
# 6. PLOTS
# =========================================================================
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


def _angle_wrap(d):
    """Wrap an angle difference into (-pi, pi]."""
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def plot_pred_vs_true_all(Y_te, fits, savepath):
    """
    Handout: 'Verify by making plots that the nonlinear model does fit the data.'
    4 rows (one per tag) x 4 cols (state components). Points hug the identity
    if the model fits well on held-out test data.
    """
    tags = ["delta_NN", "delta_NM", "resid_NN", "resid_NM"]
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for r, tag in enumerate(tags):
        fit = fits[tag]
        Yp = fit["Y_pred_te"]
        log = fit["log"]
        for k in range(4):
            ax = axes[r, k]
            ax.scatter(Y_te[:, k], Yp[:, k], s=4, alpha=0.4)
            lo = min(Y_te[:, k].min(), Yp[:, k].min())
            hi = max(Y_te[:, k].max(), Yp[:, k].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            mse_k = float(np.mean((Y_te[:, k] - Yp[:, k]) ** 2))
            ax.set_title(f"{DELTA_LABELS[k]}  MSE={mse_k:.2e}", fontsize=9)
            if k == 0:
                ax.set_ylabel(
                    f"{tag}\nM={log['headline_M']},"
                    f" $\\lambda$={log['headline_lambda']:.0e}\n"
                    f"total MSE={log['headline_test_mse']:.2e}\npred",
                    fontsize=9,
                )
            if r == 3:
                ax.set_xlabel("true")
            ax.set_aspect("equal", "box")
    fig.suptitle("Pred-vs-true on test set | all 4 (target, form) combinations")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_convergence_2x2(conv_dict, savepath):
    """
    Handout: 'Study the convergence as a function of increasing both the amount
    of data and the number of basis functions.'
    2x2 panels:
        rows = {delta, resid}; cols = {vs N (NN form), vs M (NM form)}.
    NN form uses M = N by definition, so 'vs N' is the meaningful axis for it.
    NM form fixes N and doubles M from 10 -> 640, the handout-prescribed sweep.
    Each panel shows per-component lines + the total MSE in black.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    grid = [
        [("delta_NN", "N (training set size)", "delta_NN: convergence vs N"),
         ("delta_NM", "M (basis centres)",     "delta_NM: convergence vs M")],
        [("resid_NN", "N (training set size)", "resid_NN: convergence vs N"),
         ("resid_NM", "M (basis centres)",     "resid_NM: convergence vs M")],
    ]
    for r in range(2):
        for c in range(2):
            tag, xlab, title = grid[r][c]
            ax = axes[r, c]
            conv = conv_dict[tag]
            xs = sorted(conv.keys())
            total = [conv[x]["mse_total"] for x in xs]
            per_c = np.stack([conv[x]["mse_per_c"] for x in xs], axis=0)
            ax.loglog(xs, total, "ko-", lw=2, label="total")
            for k in range(4):
                ax.loglog(xs, per_c[:, k], "o-", label=DELTA_LABELS[k])
            ax.set_xlabel(xlab)
            ax.set_ylabel("test MSE")
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.4)
            ax.legend(fontsize=8)
    fig.suptitle("Test-MSE convergence: data size N and basis centres M, both targets")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_2d_slices_grid(model, base_state, slice_configs, n_grid, savepath, tag):
    """
    Handout: 'plot 2D slices of your target function and the fit'.
    N_slices rows x 3 cols (true | model | model-true).
    Each row is a different (vary_j, vary_k) pair; non-varied components fixed.
    """
    nslices = len(slice_configs)
    fig, axes = plt.subplots(nslices, 3, figsize=(14, 4.4 * nslices))
    if nslices == 1:
        axes = axes[None, :]
    for r, (vj, vk, rj, rk, oc) in enumerate(slice_configs):
        a = np.linspace(*rj, n_grid)
        b = np.linspace(*rk, n_grid)
        A, B = np.meshgrid(a, b, indexing="xy")
        flat = np.tile(base_state, (n_grid * n_grid, 1))
        flat[:, vj] = A.ravel()
        flat[:, vk] = B.ravel()
        Y_true = _true_dY_for_states(flat)[:, oc].reshape(n_grid, n_grid)
        Y_pred = model.predict(flat)[:, oc].reshape(n_grid, n_grid)
        D = Y_pred - Y_true
        vmin = min(Y_true.min(), Y_pred.min())
        vmax = max(Y_true.max(), Y_pred.max())
        levels = np.linspace(vmin, vmax, 21)
        absmax = max(np.max(np.abs(D)), 1e-12)
        diff_levels = np.linspace(-absmax, absmax, 21)
        panel_data = [
            (Y_true, "true",         levels,      "viridis"),
            (Y_pred, "model",        levels,      "viridis"),
            (D,      "model - true", diff_levels, "RdBu_r"),
        ]
        for c, (Z, name, lv, cmap) in enumerate(panel_data):
            ax = axes[r, c]
            cs = ax.contourf(A, B, Z, levels=lv, cmap=cmap)
            fig.colorbar(cs, ax=ax)
            ax.set_xlabel(COMPONENT_LABELS[vj])
            ax.set_ylabel(COMPONENT_LABELS[vk])
            ax.set_title(f"{name}: {DELTA_LABELS[oc]}  (others = 0)")
    fig.suptitle(f"2D slices of target $\\Delta Y$ and kernel fit ({tag} model)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def plot_rollouts_labeled(model, ICs, ic_labels, n_steps, dt, savepath, tag):
    """
    Handout: 'use rollouts to illustrate how closely the iterated model matches
    the real dynamics for a wide range of sensible initial conditions.'
    4 rows (labelled ICs) x 4 cols (state components). Each cell overlays
    the iterated kernel-model trajectory (orange dashed) on the true simulator
    trajectory (black solid) over CFG['rollout_steps'] * dt seconds.
    """
    n_ic = len(ICs)
    fig, axes = plt.subplots(n_ic, 4, figsize=(16, 3.2 * n_ic), sharex=True)
    if n_ic == 1:
        axes = axes[None, :]
    t = np.arange(n_steps + 1) * dt
    for r, X0 in enumerate(ICs):
        traj_t = true_rollout(X0, n_steps)
        traj_m = model_rollout(model, X0, n_steps)
        for k in range(4):
            ax = axes[r, k]
            ax.plot(t, traj_t[:, k], "k-", lw=1.8, label="true")
            ax.plot(t, traj_m[:, k], "C1--", lw=1.6, label="model")
            if r == 0:
                ax.set_title(COMPONENT_LABELS[k])
            if k == 0:
                ax.set_ylabel(ic_labels[r], fontsize=9)
            if r == n_ic - 1:
                ax.set_xlabel("time / s")
            if r == 0 and k == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Iterated rollouts ({tag} model): true (black) vs model (orange)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def _estimate_period(traj_t, dt):
    """
    Dominant period (s) of the true angle dynamics.
    - Rotation regime (|mean(theta_dot)| >> std): T = 2*pi / |mean(theta_dot)|.
    - Oscillation regime: detrend unwrapped theta and pick the dominant FFT bin.
    - No motion -> inf.
    """
    theta_dot = traj_t[:, 3]
    om_mean = float(np.mean(theta_dot))
    om_std  = float(np.std(theta_dot))
    if abs(om_mean) > 0.5 and om_std < 0.7 * abs(om_mean):
        return 2.0 * np.pi / abs(om_mean)
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
                    return 1.0 / f_dom
    return float("inf")


def compute_rollout_horizon(model, ICs, ic_labels, sigmas_state, n_steps, dt,
                            threshold=0.5):
    """
    Compute per-IC normalised state-error curves and breakdown summary.
    Returns (per_ic_data, summary_rows):
      per_ic_data[i] = {"t", "err", "t_break", "period", "cycles", "ic_short"}
      summary_rows[i] = dict suitable for results.json
    """
    t = np.arange(n_steps + 1) * dt
    per_ic = []
    summary = []
    for i, X0 in enumerate(ICs):
        traj_t = true_rollout(X0, n_steps)
        traj_m = model_rollout(model, X0, n_steps)
        diff = traj_m - traj_t
        diff[:, 2] = _angle_wrap(diff[:, 2])
        err = np.sqrt(np.sum((diff / sigmas_state) ** 2, axis=1))
        above = err > threshold
        t_break = float(t[int(np.argmax(above))]) if above.any() else float("inf")
        period = _estimate_period(traj_t, dt)
        if np.isfinite(t_break) and np.isfinite(period) and period > 0:
            n_cycles = t_break / period
        else:
            n_cycles = None
        ic_short = ic_labels[i].splitlines()[0]
        per_ic.append({
            "t": t, "err": err, "t_break": t_break,
            "period": period, "cycles": n_cycles, "ic_short": ic_short,
        })
        summary.append({
            "IC": ic_short,
            "T_break_s": t_break if np.isfinite(t_break) else None,
            "period_s": period if np.isfinite(period) else None,
            "cycles_at_break": n_cycles,
        })
    return per_ic, summary


def plot_rollout_horizon_2x2(data_by_tag, savepath, threshold=0.5):
    """
    Handout: 'How long does your model show reasonable agreement with the real
    dynamics? Quantify this both in units of time and in terms of the number of
    oscillation cycles.'

    2x2 panels (one per tag), each showing the 4 ICs as coloured lines with
    breakdown markers (dotted vertical at T_break).

    Per-step normalised state error  e(t) = || (X_m(t) - X_t(t)) / sigma_state ||_2
    so the four heterogeneous components contribute on the same scale.
    T_break is the first time e(t) > threshold; cycles = T_break / dominant
    period of the true trajectory.
    """
    tags = ["delta_NN", "delta_NM", "resid_NN", "resid_NM"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    for ax, tag in zip(axes.ravel(), tags):
        for data in data_by_tag[tag]:
            tarr = data["t"]
            t_break = data["t_break"]
            cycles = data["cycles"]
            tb_str = f"T_br={t_break:.2g}s" if np.isfinite(t_break) else f"T_br>{tarr[-1]:.0f}s"
            cyc_str = "n/a" if cycles is None else f"{cycles:.1f}"
            label = f"{data['ic_short']} | {tb_str}, cyc~{cyc_str}"
            line, = ax.semilogy(tarr, np.maximum(data["err"], 1e-6), lw=1.4, label=label)
            if np.isfinite(t_break):
                ax.axvline(t_break, color=line.get_color(), ls=":", lw=1, alpha=0.7)
        ax.axhline(threshold, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_title(tag)
        ax.set_xlabel("time / s")
        ax.set_ylabel(r"$\|(X_m - X_t)/\sigma_{\mathrm{state}}\|_2$")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, which="both", alpha=0.4)
    fig.suptitle(f"Rollout horizon by (target, form)  -  breakdown threshold = {threshold}")
    fig.tight_layout()
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


# =========================================================================
# 7. SWEEPS
# =========================================================================
def _fit(form, X_tr, Y_tr, sigmas, lam, M, rng, C_baseline):
    if form == "NN":
        return fit_NN(X_tr, Y_tr, sigmas, lam, C_baseline=C_baseline)
    if form == "NM":
        return fit_NM(X_tr, Y_tr, sigmas, lam, M, rng, C_baseline=C_baseline)
    raise ValueError(form)


def sweep_lambda(form, X_tr, Y_tr, X_va, Y_va, sigmas, lambdas, M, rng, C_b):
    table = []
    for lam in lambdas:
        m = _fit(form, X_tr, Y_tr, sigmas, lam, M, rng, C_b)
        table.append((float(lam), float(mse(Y_va, m.predict(X_va)))))
    lam_best = min(table, key=lambda r: r[1])[0]
    return lam_best, table


def sweep_M(form, X_tr, Y_tr, X_te, Y_te, sigmas, lam, M_grid, rng, C_b):
    out = {}
    for M in M_grid:
        if M > X_tr.shape[0]:
            continue
        m = _fit(form, X_tr, Y_tr, sigmas, lam, M, rng, C_b)
        Y_pred = m.predict(X_te)
        out[M] = {
            "mse_total": float(mse(Y_te, Y_pred)),
            "mse_per_c": mse(Y_te, Y_pred, per_component=True).tolist(),
        }
    return out


def sweep_N(form, X_tr_full, Y_tr_full, X_te, Y_te, sigmas, lam,
            N_grid, M_fixed, rng, C_b):
    out = {}
    for N in N_grid:
        if N > X_tr_full.shape[0]:
            continue
        X_tr, Y_tr = X_tr_full[:N], Y_tr_full[:N]
        M = min(M_fixed, N) if form == "NM" else N
        m = _fit(form, X_tr, Y_tr, sigmas, lam, M, rng, C_b)
        Y_pred = m.predict(X_te)
        out[N] = {
            "mse_total": float(mse(Y_te, Y_pred)),
            "mse_per_c": mse(Y_te, Y_pred, per_component=True).tolist(),
        }
    return out


# =========================================================================
# 8. MAIN
# =========================================================================
def main():
    rng = np.random.default_rng(CFG["seed"])

    print(f"[1/6] Gathering {CFG['N_total']} samples...")
    X_all, Y_all = gather_dataset(CFG["N_total"], rng)
    split = split_dataset(X_all, Y_all, CFG["frac"], np.random.default_rng(1))
    X_tr, Y_tr = split["train"]
    X_va, Y_va = split["val"]
    X_te, Y_te = split["test"]
    print(f"      train={X_tr.shape[0]}  val={X_va.shape[0]}  test={X_te.shape[0]}")

    # Per-component scale of the actual state distribution: used to normalise
    # rollout errors so the four heterogeneous components compare fairly.
    sigmas_state = X_all.std(axis=0)

    print("[2/6] Linear baseline...")
    C = fit_linear(X_tr, Y_tr)
    lin_mse = float(mse(Y_te, predict_linear(C, X_te)))
    print(f"      linear test MSE: {lin_mse:.6f}")

    sigmas = default_sigmas_from_data(X_tr)
    print(f"      sigmas (default): {sigmas}")

    all_results = {
        "config": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in CFG.items()},
        "linear_test_mse": lin_mse,
        "sigmas": sigmas.tolist(),
        "sigmas_state": sigmas_state.tolist(),
        "C_linear": C.tolist(),
        "runs": {},
    }

    rollout_ICs = [
        np.array([0.0, 0.0, np.pi,       0.0]),
        np.array([0.0, 0.0, np.pi - 0.5, 0.0]),
        np.array([0.0, 0.5, 0.5,         0.0]),
        np.array([0.0, 0.0, 0.1,         3.0]),
    ]
    IC_LABELS = [
        "IC1: pole down, still\nX=[0, 0, $\\pi$, 0]",
        "IC2: small swing from down\nX=[0, 0, $\\pi$-0.5, 0]",
        "IC3: mid-swing + cart drift\nX=[0, 0.5, 0.5, 0]",
        "IC4: near upright + rapid spin\nX=[0, 0, 0.1, 3.0]",
    ]
    scan_base = np.array([0.0, 0.0, 0.0, 0.0])
    slice_configs = [
        (2, 3, (-np.pi, np.pi), (-15.0, 15.0), 3),   # vary theta, theta_dot -> Delta theta_dot
        (2, 3, (-np.pi, np.pi), (-15.0, 15.0), 1),   # vary theta, theta_dot -> Delta x_dot
        (1, 3, (-10.0,  10.0),  (-15.0, 15.0), 1),   # vary x_dot, theta_dot -> Delta x_dot
    ]

    # ------------------------------------------------------------------
    # Fit all 4 (target, form) combos -- only their NUMBERS go into the
    # comparison table; only the delta target gets PLOTTED, per the
    # handout's 'you could choose either' phrasing (delta wins here).
    # ------------------------------------------------------------------
    fits = {}
    print("[3/6] Fitting all 4 (target, form) combos for the comparison table...")
    for target in ("delta", "resid"):
        C_b = C if target == "resid" else None
        for form in ("NN", "NM"):
            tag = f"{target}_{form}"
            print(f"  -- {tag}")
            log = {}

            lam_best, lam_table = sweep_lambda(
                form, X_tr, Y_tr, X_va, Y_va, sigmas,
                CFG["lambda_grid"], CFG["headline_M"], rng, C_b,
            )
            log["lambda_sweep"] = lam_table
            log["lambda_best"]  = float(lam_best)
            print(f"     best lambda = {lam_best:.2e}")

            M_use = CFG["headline_M"] if form == "NM" else X_tr.shape[0]
            model = _fit(form, X_tr, Y_tr, sigmas, lam_best, M_use, rng, C_b)
            Y_pred_te = model.predict(X_te)
            log["headline_M"]          = M_use
            log["headline_lambda"]     = float(lam_best)
            log["headline_test_mse"]   = float(mse(Y_te, Y_pred_te))
            log["headline_test_per_c"] = mse(Y_te, Y_pred_te,
                                             per_component=True).tolist()
            log["headline_norm_mse"]   = normalised_mse(Y_te, Y_pred_te).tolist()
            print(f"     test MSE: {log['headline_test_mse']:.6f}  "
                  f"(linear: {lin_mse:.6f})")

            fits[tag] = {"model": model, "Y_pred_te": Y_pred_te, "log": log}
            all_results["runs"][tag] = log

    # ---- Convergence sweeps for ALL 4 tags ----
    # NN form: vary N (M = N implicitly). NM form: vary M (N fixed).
    print("[4/6] Convergence sweeps for all 4 tags...")
    conv_dict = {}
    for tag, fit in fits.items():
        target, form = tag.split("_")
        C_b = C if target == "resid" else None
        lam_best = fit["log"]["lambda_best"]
        if form == "NN":
            print(f"      sweep N for {tag}...")
            conv_dict[tag] = sweep_N("NN", X_tr, Y_tr, X_te, Y_te, sigmas,
                                     lam_best, CFG["N_grid"],
                                     CFG["headline_M"], rng, C_b)
            all_results["runs"][tag]["convergence_N"] = conv_dict[tag]
        else:  # NM
            print(f"      sweep M for {tag}...")
            conv_dict[tag] = sweep_M("NM", X_tr, Y_tr, X_te, Y_te, sigmas,
                                     lam_best, CFG["M_grid"], rng, C_b)
            all_results["runs"][tag]["convergence_M"] = conv_dict[tag]

    # ---- Figures ----
    print("[5/6] Drawing figures...")
    # 1 combined pred-vs-true (all 4 tags)
    plot_pred_vs_true_all(
        Y_te, fits, savepath=FIG_DIR / "task2_1_pred_vs_true.png",
    )
    # 1 combined convergence (2x2 of all 4 tags)
    plot_convergence_2x2(conv_dict, savepath=FIG_DIR / "task2_1_convergence.png")

    # 4 per-tag 2D slice files and 4 per-tag rollout files
    horizon_data = {}
    horizon_summary = {}
    for tag, fit in fits.items():
        plot_2d_slices_grid(
            fit["model"], scan_base, slice_configs, 40,
            savepath=FIG_DIR / f"task2_1_2d_slices_{tag}.png",
            tag=tag,
        )
        plot_rollouts_labeled(
            fit["model"], rollout_ICs, IC_LABELS,
            CFG["rollout_steps"], CFG["dt"],
            savepath=FIG_DIR / f"task2_1_rollouts_{tag}.png",
            tag=tag,
        )
        per_ic, summary = compute_rollout_horizon(
            fit["model"], rollout_ICs, IC_LABELS, sigmas_state,
            CFG["rollout_steps"], CFG["dt"],
        )
        horizon_data[tag] = per_ic
        horizon_summary[tag] = summary

    # 1 combined rollout-horizon (2x2 of all 4 tags)
    plot_rollout_horizon_2x2(
        horizon_data, savepath=FIG_DIR / "task2_1_rollout_horizon.png",
    )
    all_results["rollout_horizon"] = horizon_summary

    # ---- results.json ----
    results_path = FIG_DIR / "task2_1_results.json"
    print(f"[6/6] Saving {results_path}...")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray)
                  else str(o))

    print("\n=== Summary: test MSE (lower = better) ===")
    print(f"  linear baseline          {lin_mse:.6f}")
    for tag, log in all_results["runs"].items():
        print(f"  {tag:14s}  MSE={log['headline_test_mse']:.6f}  "
              f"lambda={log['headline_lambda']:.0e}  M={log['headline_M']}")

    print("\n=== Rollout horizon (threshold = 0.5 normalised state-error units) ===")
    for tag, rows in horizon_summary.items():
        print(f"  [{tag}]")
        for row in rows:
            tb = "n/a" if row["T_break_s"] is None else f"{row['T_break_s']:.2g} s"
            pe = "n/a" if row["period_s"]  is None else f"{row['period_s']:.2g} s"
            cy = "n/a" if row["cycles_at_break"] is None else f"{row['cycles_at_break']:.1f}"
            print(f"    {row['IC']:32s}  T_break={tb:>8s}   period={pe:>8s}   cycles={cy}")


if __name__ == "__main__":
    main()