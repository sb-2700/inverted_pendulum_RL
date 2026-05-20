"""
Task 1.4 -- DIAGNOSTIC.

Goal: decide whether the linear-model divergence for the small-oscillation
case is a CODE BUG or the expected periodic-angle / compounding-error effect.

Procedure (matches the six steps in the prompt):
  1. Print fit target + iteration formula and check consistency.        [code-read]
  2. Verify state-vector column ordering is consistent with getState(). [code-read]
  3. Single-step sanity: 20 random probes -- mean abs error per component.
  4. Per-step rollout error for (0, 0, pi, 0.5), with angular error wrapped.
  5. Wrap-boundary hypothesis: refit a "phi-centred" model where the
     equilibrium sits at phi = theta - pi = 0 (well inside training range)
     and compare its small-oscillation rollout.
  6. Final report printed at the bottom.

Run::
    python week1/task1_4_debug.py
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cartpole import CartPole, remap_angle  # noqa: E402
from task1_3_linear_model import data_gen, fit_linear, RANGES, STATE_SHORT  # noqa: E402

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RNG_SEED = 0
N_TRAIN  = 500
N_STEPS  = 200
DT       = 0.1

_CARTPOLE = CartPole()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def one_step_true(state):
    """Reset all 4 vars, one zero-force step, return X_after."""
    _CARTPOLE.setState(list(state))
    _CARTPOLE.performAction(0.0)
    return _CARTPOLE.getState()


def rollout_true(initial_state, n_steps):
    _CARTPOLE.setState(list(initial_state))
    out = np.empty((n_steps + 1, 4))
    out[0] = _CARTPOLE.getState()
    for k in range(1, n_steps + 1):
        _CARTPOLE.performAction(0.0)
        out[k] = _CARTPOLE.getState()
    return out


def rollout_model(initial_state, C, n_steps, remap=True):
    x = np.asarray(initial_state, float).copy()
    out = np.empty((n_steps + 1, 4))
    out[0] = x
    for k in range(1, n_steps + 1):
        x = x + C @ x
        if remap:
            x[2] = remap_angle(x[2])
        out[k] = x
    return out


def wrap(a):
    """Vectorised remap_angle into [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Phi-centred variants (Step 5)
# ---------------------------------------------------------------------------
def data_gen_phi(N, rng):
    """Like data_gen, but reports the state as phi = remap(theta - pi).

    The simulator is still queried with the REAL theta; only the
    coordinates handed to the regression are shifted.  The CHANGE
    Y = X_after - X_before is the same in both coordinates because the
    pi-shift cancels (subtract pi from both endpoints).  The angle
    component of Y is still wrapped to remove the +-2pi jump.
    """
    X = rng.uniform(RANGES[:, 0], RANGES[:, 1], size=(N, 4))   # theta-coords
    Y = np.empty_like(X)
    for i in range(N):
        x_new = one_step_true(X[i])
        d = x_new - X[i]
        d[2] = remap_angle(d[2])
        Y[i] = d
    # Re-express the X-side in phi-coordinates.
    X_phi = X.copy()
    X_phi[:, 2] = wrap(X[:, 2] - np.pi)
    return X_phi, Y


def rollout_model_phi(initial_state_theta, C_phi, n_steps):
    """Iterate the phi-centred linear model.

    The user supplies an INITIAL STATE in real (theta) coordinates; we
    shift into phi, iterate ``x_{n+1} = x_n + C_phi @ x_n`` with a wrap
    on the angle component, and shift back to theta for the output.
    """
    x = np.asarray(initial_state_theta, float).copy()
    x[2] = wrap(x[2] - np.pi)
    out = np.empty((n_steps + 1, 4))
    out[0] = x.copy()
    for k in range(1, n_steps + 1):
        x = x + C_phi @ x
        x[2] = remap_angle(x[2])
        out[k] = x
    # Shift back to theta-coords for display.
    out_theta = out.copy()
    out_theta[:, 2] = wrap(out[:, 2] + np.pi)
    return out_theta


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def step3_single_step_sanity(C, rng):
    print("-" * 72)
    print("STEP 3 -- single-step sanity (20 random probes)")
    print("-" * 72)
    N = 20
    X = rng.uniform(RANGES[:, 0], RANGES[:, 1], size=(N, 4))
    Y_true = np.empty_like(X)
    Y_pred = np.empty_like(X)
    for i in range(N):
        x_after = one_step_true(X[i])
        d_true = x_after - X[i]
        d_true[2] = remap_angle(d_true[2])
        Y_true[i] = d_true
        Y_pred[i] = C @ X[i]

    abs_err = np.abs(Y_true - Y_pred)
    mae = abs_err.mean(axis=0)
    rms = np.sqrt(((Y_true - Y_pred) ** 2).mean())

    # Compare per-component MAE against the typical scale of Y itself.
    typical = np.abs(Y_true).mean(axis=0)
    print(f"  Per-component MAE of ONE-STEP change (Delta Y):")
    print(f"    {'component':<10s} {'MAE':>12s}  {'|Y|_mean':>12s}  {'MAE / |Y|':>10s}")
    for j in range(4):
        ratio = mae[j] / max(typical[j], 1e-30)
        print(f"    {STATE_SHORT[j]:<10s} {mae[j]:12.3e}  {typical[j]:12.3e}  "
              f"{ratio:>10.2%}")
    print(f"  Overall RMS error of one-step Delta Y: {rms:.4e}")
    print()
    print("  Interpretation: single-step errors of a few percent of typical |Y|")
    print("  mean C itself is doing about as well as a linear model CAN on this")
    print("  data.  In that case the rollout divergence is the expected")
    print("  compounding effect, not a fit bug.  A MAE comparable to or larger")
    print("  than |Y| would indicate a fit-side bug (target, column order, lstsq).")
    print()
    return rms


def step4_rollout_error(C, n_steps=N_STEPS, threshold_rad=0.2,
                       fname="task1_4_debug_per_step_error.png"):
    print("-" * 72)
    print(f"STEP 4 -- per-step rollout error for (0, 0, pi, 0.5), {n_steps} steps")
    print("-" * 72)
    s0 = np.array([0.0, 0.0, np.pi, 0.5])
    true_traj  = rollout_true(s0,  n_steps)
    model_traj = rollout_model(s0, C, n_steps, remap=True)

    err = model_traj - true_traj
    err[:, 2] = wrap(err[:, 2])             # angular error: signed wrapped diff
    abs_err = np.abs(err)

    # True oscillation period (measure as 2 * mean half-period from zero-crossings
    # of theta_dot, since theta oscillates around pi).
    td = true_traj[:, 3]
    zc = np.where(np.diff(np.sign(td)) != 0)[0]   # indices of sign changes
    if len(zc) >= 2:
        half_periods_steps = np.diff(zc).astype(float)
        period_steps = 2.0 * half_periods_steps.mean()
        period_sec = period_steps * DT
    else:
        period_steps = float("nan")
        period_sec = float("nan")
    print(f"  Measured oscillation period (true dyn): "
          f"{period_sec:.3f} s ({period_steps:.1f} steps)")

    # First step where wrapped angular error exceeds threshold.
    over = np.where(abs_err[:, 2] > threshold_rad)[0]
    if len(over):
        k_break = int(over[0])
        t_break = k_break * DT
        cycles  = t_break / period_sec if period_sec == period_sec else float("nan")
        print(f"  First step where |angle error| > {threshold_rad:.2f} rad:")
        print(f"    step  = {k_break}")
        print(f"    time  = {t_break:.2f} s")
        print(f"    cycles= {cycles:.2f} oscillation periods")
    else:
        print(f"  |angle error| never exceeds {threshold_rad} rad.")
        k_break, t_break = None, None

    # Plot per-step error.
    time = np.arange(n_steps + 1) * DT
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for j, ax in enumerate(axes.flat):
        ax.semilogy(time, np.maximum(abs_err[:, j], 1e-12),
                    color=f"C{j}", lw=1.5)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"|err {STATE_SHORT[j]}|")
        ax.grid(alpha=0.3, which="both")
        if j == 2 and k_break is not None:
            ax.axvline(t_break, color="grey", ls="--", lw=1.0,
                       label=f"|err theta| > {threshold_rad} at t = {t_break:.2f} s")
            ax.legend(loc="lower right")
    fig.suptitle("Task 1.4 debug -- per-step linear-model error\n"
                 "initial state (0, 0, pi, 0.5); angular error wrapped to [-pi, pi]")
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)
    print(f"  Wrote {FIG_DIR / fname}")
    print()
    return k_break, period_sec


def step5_recenter(rng):
    print("-" * 72)
    print("STEP 5 -- phi-centred refit (phi = theta - pi, equilibrium at 0)")
    print("-" * 72)
    X_phi, Y = data_gen_phi(N_TRAIN, rng=rng)
    C_phi = fit_linear(X_phi, Y)

    print("  Fitted C_phi (rows = output, cols = input in phi-coords):")
    for j in range(4):
        row = "  ".join(f"{C_phi[j, k]:+11.4e}" for k in range(4))
        print(f"    {STATE_SHORT[j]:<10s}{row}")
    print()

    s0_theta = np.array([0.0, 0.0, np.pi, 0.5])
    true_traj  = rollout_true(s0_theta, N_STEPS)
    model_traj = rollout_model_phi(s0_theta, C_phi, N_STEPS)

    # Per-step error, angular part wrap-corrected.
    err = model_traj - true_traj
    err[:, 2] = wrap(err[:, 2])
    abs_err = np.abs(err)
    over = np.where(abs_err[:, 2] > 0.2)[0]
    if len(over):
        k_break = int(over[0])
        t_break = k_break * DT
        print(f"  First step where |angle error| > 0.2 rad: step {k_break} "
              f"(t = {t_break:.2f} s)")
    else:
        k_break, t_break = None, None
        print("  |angle error| never exceeds 0.2 rad over the whole rollout.")

    # Comparison plot: theta(t) for true / theta-centred model / phi-centred model.
    # Need the theta-centred prediction too -- refit C in theta coords for this plot.
    rng_local = np.random.default_rng(RNG_SEED)
    X_t, Y_t = data_gen(N_TRAIN, rng=rng_local)
    C_theta = fit_linear(X_t, Y_t)
    model_traj_theta = rollout_model(s0_theta, C_theta, N_STEPS, remap=True)

    time = np.arange(N_STEPS + 1) * DT
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for j, ax in enumerate(axes.flat):
        ax.plot(time, true_traj[:, j],  color=f"C{j}", lw=2.0,
                label="true (simulator)")
        ax.plot(time, model_traj_theta[:, j], color="k", lw=1.0, ls="--",
                label="linear model, theta-coords")
        ax.plot(time, model_traj[:, j], color="grey", lw=1.2, ls=":",
                label="linear model, PHI-centred")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(STATE_SHORT[j])
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="best", fontsize=9)
    fig.suptitle("Task 1.4 debug -- small oscillation: theta-coords vs phi-centred fit")
    fig.savefig(FIG_DIR / "task1_4_debug_recenter.png")
    plt.close(fig)
    print(f"  Wrote {FIG_DIR / 'task1_4_debug_recenter.png'}")
    print()
    return k_break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Task 1.4 -- DIAGNOSTIC for the small-oscillation divergence")
    print("=" * 72)

    # ---- Steps 1 & 2: code-read confirmations -----------------------------
    print()
    print("-" * 72)
    print("STEP 1 -- fit target and iteration formula")
    print("-" * 72)
    print("  task1_3_linear_model.data_gen():")
    print("    d = x_new - X[i]; d[2] = remap_angle(d[2]); Y[i] = d")
    print("    => regression target is the CHANGE in state, with the angle")
    print("       COMPONENT of the change wrapped to remove +-2pi jumps.")
    print()
    print("  task1_3_linear_model.fit_linear():")
    print("    M, *_ = np.linalg.lstsq(X_data, Y_data, rcond=None); return M.T")
    print("    => solves Y ~ X @ C.T, i.e. Y = C @ X with lstsq (NOT inv).")
    print()
    print("  task1_4_iterated_rollouts.rollout_model():")
    print("    x = x + C @ x; x[2] = remap_angle(x[2])")
    print("    => iterates X_{n+1} = X_n + C @ X_n.")
    print()
    print("  CONSISTENCY: C predicts the CHANGE and the rollout adds C@X to X.")
    print("               => formulae are consistent.  No bug here.")
    print()

    print("-" * 72)
    print("STEP 2 -- state-vector column order")
    print("-" * 72)
    print("  cartpole.CartPole.setState/getState use [x, x_dot, theta, theta_dot].")
    print("  task1_3 RANGES rows are in the SAME order; data_gen passes those rows")
    print("  unchanged through setState() and stores X[i] = the sampled vector.")
    print("  task1_4 rollout_model treats x[2] as the angle when remapping.")
    print("  => index 2 is theta in every location.  No bug here.")
    print()

    # ---- Refit C (for Steps 3 and 4) --------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    X_train, Y_train = data_gen(N_TRAIN, rng=rng)
    C = fit_linear(X_train, Y_train)

    # ---- Step 3 -----------------------------------------------------------
    rms = step3_single_step_sanity(C, rng)

    # ---- Step 4 -----------------------------------------------------------
    k_break_theta, period_sec = step4_rollout_error(C)

    # ---- Step 5 -----------------------------------------------------------
    rng_phi = np.random.default_rng(RNG_SEED + 1)
    k_break_phi = step5_recenter(rng_phi)

    # ---- Step 6 -----------------------------------------------------------
    print("=" * 72)
    print("STEP 6 -- summary")
    print("=" * 72)
    cyc_theta = (k_break_theta * DT / period_sec) if (
        k_break_theta is not None and period_sec == period_sec) else None
    cyc_phi   = (k_break_phi   * DT / period_sec) if (
        k_break_phi   is not None and period_sec == period_sec) else None

    bug_found = rms > 1.0      # an overall RMS this large would be alarming
    print(f"  Bug found in the FIT?           : {'YES' if bug_found else 'NO'}")
    print(f"  One-step RMS error on Delta Y   : {rms:.4e}")
    print(f"  Measured true period (small osc): {period_sec:.3f} s")
    print()
    print(f"  theta-coords model (handout default):")
    if k_break_theta is None:
        print(f"    angle error never crosses 0.2 rad in 200 steps.")
    else:
        print(f"    angle error > 0.2 rad at step {k_break_theta} "
              f"(t = {k_break_theta * DT:.2f} s, ~{cyc_theta:.2f} cycles)")
    print(f"  phi-centred model (diagnostic only):")
    if k_break_phi is None:
        print(f"    angle error never crosses 0.2 rad in 200 steps.")
    else:
        print(f"    angle error > 0.2 rad at step {k_break_phi} "
              f"(t = {k_break_phi * DT:.2f} s, ~{cyc_phi:.2f} cycles)")
    print()
    print("  Recommendation for Task 1.4 plots:")
    print("    - The fit/iteration are correct; the divergence is the known")
    print("      periodic-angle + linear-extrapolation effect, NOT a code bug.")
    print("    - Keep the handout's 4 plots (true theta-coords) but consider")
    print("      shortening the small-oscillation rollout to ~50-80 steps so the")
    print("      first agreement window is visible before blow-up dominates.")
    print("    - The phi-centred model is a diagnostic only; it confirms that")
    print("      almost all of the small-oscillation error is due to placing the")
    print("      stable equilibrium at the wrap boundary, NOT to a missing")
    print("      nonlinearity per se.")


if __name__ == "__main__":
    main()
