"""
Task 1.4 -- Iterated rollouts of the LINEAR one-step model.

Background
----------
Task 1.3 fitted a 4x4 matrix C so that the one-step CHANGE in state is
modelled as

    Y = X_{n+1} - X_n = C @ X_n.

Task 1.4 iterates this rule:

    X_{n+1} = X_n + C @ X_n,

starting from the same initial state as the true simulator, and compares
the model trajectory against the simulator over many steps.

Two implementation details matter:

1. The pole angle component of X_{n+1} must be remapped into [-pi, pi]
   AFTER every update, before being fed into the next iteration.  Without
   this, theta accumulates as if it were unbounded (the linear model does
   not know that theta and theta + 2pi describe the same physical state),
   and the prediction diverges as soon as the pole makes a half-turn.

2. The true simulator's ``performAction`` uses theta only through sin and
   cos, which are 2pi-periodic; the real physics needs no remap.

Run from the repository root::

    python week1/task1_4_iterated_rollouts.py

Figures and the C matrix are written to ``<repo>/figures/``.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make ``cartpole.py`` importable regardless of working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cartpole import CartPole, remap_angle  # noqa: E402

# Reuse the Task 1.3 fitter so we know we are running the same model.
from task1_3_linear_model import data_gen, fit_linear, RANGES, STATE_LABELS, STATE_SHORT  # noqa: E402

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Plot styling -- matches the rest of the week-1 scripts.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":       120,
    "savefig.dpi":      200,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "figure.titlesize": 14,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "lines.linewidth":  1.8,
    "legend.fontsize":  10,
})

RNG_SEED = 0
N_TRAIN  = 500
N_STEPS  = 200            # 200 * 0.1 s = 20 s simulated per rollout
DT       = 0.1            # CartPole.delta_time -- used only for the time axis


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------
# One reusable simulator instance.  setState() fully overwrites the four
# state variables, so each rollout is independent.
_CARTPOLE = CartPole()


def rollout_true(initial_state, n_steps):
    """Run the real CartPole for ``n_steps`` zero-force steps.

    Returns
    -------
    states : (n_steps + 1, 4) array
        Row 0 is ``initial_state``; row k is the state AFTER k calls to
        ``performAction(0.0)``.

    Notes
    -----
    No remapping is applied to the returned trajectory: the simulator's
    physics uses sin/cos of theta internally, so theta can drift outside
    [-pi, pi] without any loss of correctness.
    """
    _CARTPOLE.setState(list(initial_state))           # overwrite all 4 vars
    states = np.empty((n_steps + 1, 4))
    states[0] = _CARTPOLE.getState()
    for k in range(1, n_steps + 1):
        _CARTPOLE.performAction(0.0)
        states[k] = _CARTPOLE.getState()
    return states


def rollout_model(initial_state, C, n_steps, remap=True):
    """Iterate the linear model X_{n+1} = X_n + C @ X_n.

    Parameters
    ----------
    initial_state : (4,) array_like
    C             : (4, 4) array     fitted in Task 1.3
    n_steps       : int
    remap         : bool, default True
        If True (the correct behaviour for a meaningful comparison),
        wrap the pole-angle COMPONENT of X_{n+1} into [-pi, pi] before
        feeding it into the next iteration.  If False (diagnostic), let
        theta drift freely -- this is the divergent rollout requested
        in the handout.

    Returns
    -------
    states : (n_steps + 1, 4) array
    """
    x = np.asarray(initial_state, dtype=float).copy()
    states = np.empty((n_steps + 1, 4))
    states[0] = x
    for k in range(1, n_steps + 1):
        x = x + C @ x
        if remap:
            x[2] = remap_angle(x[2])
        states[k] = x
    return states


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_compare_2x2(time, true_traj, model_traj, title, fname,
                     extra_traj=None, extra_label=None, yscale="linear"):
    """2x2 grid: one state component per panel, true vs model vs time.

    ``true_traj`` and ``model_traj`` are (T, 4) arrays.  An optional
    ``extra_traj`` is overlaid as a third curve (used to show the
    un-remapped model on the full-rotation rollout).

    ``yscale`` is either ``"linear"`` (default) or ``"symlog"``.  Symlog is
    used only for the no-remap diagnostic, where the un-remapped trajectory
    explodes by tens of orders of magnitude and would otherwise crush the
    well-behaved curves to a flat line.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for j, ax in enumerate(axes.flat):
        col = f"C{j}"
        ax.plot(time, true_traj[:, j],  color=col, lw=2.0, label="true (simulator)")
        ax.plot(time, model_traj[:, j], color=col, lw=1.4, ls="--",
                label="linear model (remapped)")
        if extra_traj is not None:
            ax.plot(time, extra_traj[:, j], color="k", lw=1.0, ls=":",
                    label=extra_label)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(STATE_LABELS[j])
        ax.grid(alpha=0.3)
        if yscale == "symlog":
            # linthresh=1.0 keeps the [-1, 1] band linear so the small,
            # well-behaved oscillations of the true / remapped trajectories
            # remain readable on the same axes as the blow-up curve.
            ax.set_yscale("symlog", linthresh=1.0)
        if j == 0:
            ax.legend(loc="best")
    fig.suptitle(title)
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


def _fmt_state(s):
    """Compact human-readable initial-state string for figure titles."""
    return (f"x={s[0]:+.2f}, xdot={s[1]:+.2f}, "
            f"theta={s[2]:+.2f}, thetadot={s[3]:+.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Task 1.4 -- Iterated rollouts of the linear model")
    print("=" * 72)

    # Report-question answers, printed up-front as required.
    print()
    print("Report question (i) -- why does the UN-remapped linear model diverge?")
    print("  The linear model C was fitted on states X with theta sampled in")
    print("  [-pi, pi].  When iterated without remap, theta accumulates")
    print("  monotonically once the pole has angular momentum, so the input")
    print("  state moves far outside the training range -- this is extreme")
    print("  extrapolation.  Worse, the linear model treats theta and")
    print("  theta + 2pi as physically DIFFERENT, even though they describe")
    print("  the same configuration; with each turn the C @ X term picks up")
    print("  larger and larger contributions, and the trajectory blows up.")
    print()
    print("Report question (ii) -- why does the TRUE dynamics need no remap?")
    print("  The simulator's equations of motion enter theta only through")
    print("  sin(theta) and cos(theta).  Both are 2pi-periodic, so the real")
    print("  physics is automatically invariant under theta -> theta + 2pi;")
    print("  the absolute value of theta is irrelevant to its time evolution.")
    print()

    # ---- Step 1: refit C exactly as in Task 1.3 --------------------------
    print(f"[1/3] Fitting linear model C on N={N_TRAIN} random one-step samples...")
    rng = np.random.default_rng(RNG_SEED)
    X_train, Y_train = data_gen(N_TRAIN, rng=rng)
    C = fit_linear(X_train, Y_train)

    print()
    print("Fitted C (rows = output, cols = input):")
    header = "          " + "".join(f"{s:>12s}" for s in STATE_SHORT)
    print(header)
    for j in range(4):
        row = "  ".join(f"{C[j, k]:+11.4e}" for k in range(4))
        print(f"  {STATE_SHORT[j]:<8s}{row}")
    print()

    # Stability of the iterated linear map X_{n+1} = (I + C) X_n is set by
    # the spectrum of (I + C): |lambda| > 1 in any direction means the
    # un-remapped rollout grows without bound along that mode.
    eigvals = np.linalg.eigvals(np.eye(4) + C)
    print("Eigenvalues of (I + C):")
    for e in eigvals:
        print(f"  {e}  |lambda| = {abs(e):.4f}")
    print()

    # Persist the fitted C so later tasks can load the same model.
    c_path = FIG_DIR / "task1_4_C.npy"
    np.save(c_path, C)
    print(f"  Saved C matrix to {c_path}")
    print()

    # ---- Step 2: comparison rollouts -------------------------------------
    # Initial conditions.  Recall theta=pi is the STABLE (pole-down)
    # equilibrium; theta=0 is the UPRIGHT unstable equilibrium.
    initial_states = {
        "small_oscillation": {
            # Slight kick about the down equilibrium -- the linear model
            # should track this well because we never leave the training
            # region and the dynamics are nearly harmonic here.
            "state": np.array([0.0, 0.0, np.pi, 0.5]),
            "title": "(a) Small oscillation about pole-down equilibrium",
            "fname": "task1_4_rollout_a_small.png",
        },
        "large_swing": {
            # A bigger swing through the bottom but still NOT going over
            # the top.  This exercises the strongly nonlinear part of the
            # potential, where the linear model is known (from Task 1.3)
            # to fit Delta theta_dot poorly.
            "state": np.array([0.0, 0.0, np.pi, 5.0]),
            "title": "(b) Larger-amplitude swing",
            "fname": "task1_4_rollout_b_large.png",
        },
        "full_rotation": {
            # Enough angular velocity to send the pole over the top --
            # this is the case that exposes the periodicity issue.
            # Task 1.1 placed the separatrix at |theta_dot| ~ 13 rad/s, so
            # 15 rad/s is comfortably past it and gives clean rotation.
            "state": np.array([0.0, 0.0, np.pi, 15.0]),
            "title": "(c) Full 360-degree rotation (theta_dot_0 = 15 rad/s)",
            "fname": "task1_4_rollout_c_rotation.png",
        },
    }

    time = np.arange(N_STEPS + 1) * DT

    print(f"[2/3] Running {len(initial_states)} comparison rollouts "
          f"(N_STEPS = {N_STEPS}, total time = {N_STEPS * DT:.1f} s)...")
    for key, cfg in initial_states.items():
        s0 = cfg["state"]
        print(f"  - {key:<18s}  init = [{_fmt_state(s0)}]")
        true_traj  = rollout_true(s0,  N_STEPS)
        model_traj = rollout_model(s0, C, N_STEPS, remap=True)
        if key == "full_rotation":
            # Sanity-check that the chosen theta_dot is actually past the
            # separatrix.  The simulator does not remap theta internally,
            # so the un-wrapped net change in theta directly counts turns.
            net_dtheta = true_traj[-1, 2] - true_traj[0, 2]
            n_rotations = abs(net_dtheta) / (2 * np.pi)
            print(f"    net delta theta over rollout = {net_dtheta:+.2f} rad "
                  f"({n_rotations:.1f} rotations)")
        plot_compare_2x2(
            time, true_traj, model_traj,
            title=f"Task 1.4  --  {cfg['title']}\ninitial state: {_fmt_state(s0)}",
            fname=cfg["fname"],
        )

    # ---- Step 3: diagnostic -- un-remapped full rotation -----------------
    print()
    print("[3/3] Diagnostic: full-rotation case with remap DISABLED...")
    s0 = initial_states["full_rotation"]["state"]
    true_traj         = rollout_true(s0,  N_STEPS)
    model_remapped    = rollout_model(s0, C, N_STEPS, remap=True)
    model_no_remap    = rollout_model(s0, C, N_STEPS, remap=False)

    plot_compare_2x2(
        time, true_traj, model_remapped,
        title=("Task 1.4  --  full-rotation rollout: with vs without angle remap\n"
               f"initial state: {_fmt_state(s0)}  (symlog y-axis)"),
        fname="task1_4_rollout_c_no_remap_diagnostic.png",
        extra_traj=model_no_remap,
        extra_label="linear model (NO remap)",
        yscale="symlog",
    )

    # Print summary numbers for the report.
    print()
    print("  Final theta after 200 steps (un-remapped vs remapped vs true):")
    print(f"    un-remapped model:   theta = {model_no_remap[-1, 2]:+10.3f} rad")
    print(f"    remapped model:      theta = {model_remapped[-1, 2]:+10.3f} rad")
    print(f"    true simulator:      theta = {true_traj[-1, 2]:+10.3f} rad")
    print()

    print(f"Figures written to: {FIG_DIR}/")
    for cfg in initial_states.values():
        print(f"  {cfg['fname']}")
    print("  task1_4_rollout_c_no_remap_diagnostic.png")
    print()
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
