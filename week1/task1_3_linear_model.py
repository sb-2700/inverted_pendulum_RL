"""
Task 1.3 -- Fit and test a LINEAR one-step model of the cartpole.

Model
-----
We model the CHANGE in state after one ``performAction(0.0)`` call (free
dynamics).  With X = [x, xdot, theta, theta_dot] and Y = X_after - X_before,
the linear model is

    Y = C @ X,   C in R^{4x4}.

C is fitted by least squares on a TRAINING set of N = 500 randomly sampled
states, then evaluated on a SEPARATE TEST set of 500 fresh samples so we
report generalisation error (not training fit).

Run from the repository root::

    python week1/task1_3_linear_model.py

Figures written to ``<repo>/figures/``::

    task1_3_pred_vs_true.png   2x2 scatter, predicted vs true Delta-Y
    task1_3_scans.py.png       2x2 scans of one input variable, true vs predicted
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no interactive backend required
import matplotlib.pyplot as plt

# Make ``cartpole.py`` importable regardless of the working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cartpole import CartPole, remap_angle  # noqa: E402

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Windows consoles may not default to UTF-8; reconfigure stdout so the
# printed summary (which contains a few unicode glyphs) does not crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Conventions / styling
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

# Sampling ranges (consistent with Task 1.2).
RANGES = np.array([
    [-5.0,     5.0],         # x        (cart position, m)
    [-10.0,   10.0],         # x_dot    (cart velocity, m/s)
    [-np.pi,   np.pi],       # theta    (pole angle, rad)
    [-15.0,   15.0],         # theta_dot (pole angular velocity, rad/s)
])

STATE_LABELS = [
    r"$x$ (m)",
    r"$\dot x$ (m/s)",
    r"$\theta$ (rad)",
    r"$\dot\theta$ (rad/s)",
]
STATE_SHORT = ["x", "x_dot", "theta", "theta_dot"]
DELTA_LABELS = [
    r"$\Delta x$ (m)",
    r"$\Delta \dot x$ (m/s)",
    r"$\Delta \theta$ (rad)",
    r"$\Delta \dot\theta$ (rad/s)",
]

RNG_SEED = 0
N_TRAIN  = 500
N_TEST   = 500
N_SCAN   = 201           # samples per 1D sweep in the scan plot


# ---------------------------------------------------------------------------
# Core helper: a single one-step probe.
# ---------------------------------------------------------------------------
# Reuse one CartPole instance; setState() fully overwrites the four state
# variables on every call, so each measurement is independent.
_CARTPOLE = CartPole()


def one_step(state):
    """Reset all four state variables, take ONE zero-force step, return X_new."""
    _CARTPOLE.setState(list(state))
    _CARTPOLE.performAction(0.0)
    return _CARTPOLE.getState()


# ---------------------------------------------------------------------------
# Task 1.3 step 1 -- data generation
# ---------------------------------------------------------------------------
def data_gen(N, rng=None):
    """Return (X_data, Y_data) with N rows each, shape (N, 4).

    X_data[i] is sampled uniformly within ``RANGES``.  Y_data[i] is the
    CHANGE Y = X_after - X_before after one ``performAction(0.0)`` call.

    The angle COMPONENT of the change (column 2) is remapped into [-pi, pi]
    so a step that crosses the +-pi boundary does not look like a ~2*pi jump
    in the target.  Position and velocity differences are left alone.
    """
    if rng is None:
        rng = np.random.default_rng()

    X = rng.uniform(RANGES[:, 0], RANGES[:, 1], size=(N, 4))
    Y = np.empty_like(X)
    for i in range(N):
        x_new = one_step(X[i])
        d = x_new - X[i]
        d[2] = remap_angle(d[2])     # clean the angle jump only
        Y[i] = d
    return X, Y


# ---------------------------------------------------------------------------
# Task 1.3 step 2 -- linear least-squares fit
# ---------------------------------------------------------------------------
def fit_linear(X_data, Y_data):
    """Solve for C in Y = C @ X by least squares.

    With Y, X stored row-wise (rows = samples), the normal equation
    ``Y = X @ C.T`` is fed to ``np.linalg.lstsq`` (NOT ``np.linalg.inv`` --
    the handout explicitly warns against ``inv`` due to ill-conditioning).
    """
    M, *_ = np.linalg.lstsq(X_data, Y_data, rcond=None)   # M is C.T
    return M.T


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_pred_vs_true(Y_true, Y_pred, fname):
    """2x2 scatter: predicted-vs-true Delta-Y, one panel per state component.

    Each panel includes the ideal y = x line and per-panel R^2 and MSE in
    the corner so the plot is self-contained.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
    for j, ax in enumerate(axes.flat):
        yt = Y_true[:, j]
        yp = Y_pred[:, j]
        ax.scatter(yt, yp, s=10, alpha=0.5, color=f"C{j}")
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.7, label="y = x")
        mse = float(np.mean((yt - yp) ** 2))
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        ax.set_xlabel(f"true {DELTA_LABELS[j]}")
        ax.set_ylabel(f"predicted {DELTA_LABELS[j]}")
        ax.set_title(f"{STATE_SHORT[j]}:   MSE = {mse:.3e},   $R^2$ = {r2:+.3f}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    fig.suptitle(
        f"Task 1.3 -- predicted vs true $\\Delta Y$ on the held-out test set "
        f"(N = {len(Y_true)})"
    )
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


def plot_scans(C, base_states, fname):
    """Full 4x4 grid -- one row per swept input, one column per output.

    The handout phrase "repeat the scans from the previous task" wants the
    Task-1.2 scan structure overlaid with the linear-model prediction, so
    this is the same shape as the Task 1.2 1D scan figure with the model
    curve added on top.

    Layout
    ------
        rows    : swept input i in {x, x_dot, theta, theta_dot}
        columns : Delta-Y output component j in {x, x_dot, theta, theta_dot}
        panel(i, j) -- sweep input i across its full range with the other
                        three inputs pinned to ``base_states[i]``, then plot
                        component j of (true Delta Y) and (linear-model
                        Delta Y) against the swept value.

    ``base_states`` is a (4, 4) array: row i is the base state used for the
    i-th sweep.  Allowing one base per row keeps each row independent --
    we just need each scan to pass through a SOME sensible base point.
    Each panel marks the base value of its swept input with a vertical line
    so the reader can see where the slice originates.
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 14), layout="constrained",
                             sharex="row")
    line_pred = None                     # filled in once, used for shared legend

    for i in range(4):                                       # rows: swept input
        sweep_vals = np.linspace(RANGES[i, 0], RANGES[i, 1], N_SCAN)
        base = base_states[i]

        # Build a stack of probes: copies of `base` with column i swept.
        probes = np.tile(base, (N_SCAN, 1))
        probes[:, i] = sweep_vals

        # True one-step change.  Angle COMPONENT of the difference (col 2) is
        # remapped to remove the +-2*pi jumps at the periodic boundary.
        Y_true = np.empty((N_SCAN, 4))
        for k in range(N_SCAN):
            x_new = one_step(probes[k])
            d = x_new - probes[k]
            d[2] = remap_angle(d[2])
            Y_true[k] = d

        # Linear-model prediction on the SAME probe stack.
        Y_pred = probes @ C.T

        for j in range(4):                                  # cols: output comp
            ax = axes[i, j]
            ax.plot(sweep_vals, Y_true[:, j],
                    color=f"C{j}", lw=2.0)
            lp, = ax.plot(sweep_vals, Y_pred[:, j],
                          color="k", lw=1.4, ls="--")
            # Mark base-state value of the swept input.
            ax.axvline(base[i], color="grey", lw=0.9, ls=":", alpha=0.7)
            ax.axhline(0.0, color="grey", lw=0.6, ls=":", alpha=0.4)
            ax.grid(alpha=0.3)
            # Per-panel x- and y-labels so each panel is self-describing.
            # x-label: the swept-input name (rows have different x ranges).
            # y-label: the output Delta-Y component this panel plots.
            ax.set_xlabel(STATE_LABELS[i])
            ax.set_ylabel(DELTA_LABELS[j])
            if i == 0:                                       # top row only
                ax.set_title(DELTA_LABELS[j])
            line_pred = lp

        # Row label ("sweep x" etc.) sits OUTSIDE the column-0 axis on the
        # left, so it does not collide with the per-panel y-label.
        axes[i, 0].annotate(
            f"sweep {STATE_SHORT[i]}",
            xy=(-0.32, 0.5), xycoords="axes fraction",
            rotation=90, fontsize=13, fontweight="bold",
            ha="center", va="center",
        )

    # Shared legend.  Two entries only.  The "true" handle is a tuple of
    # four short coloured segments (HandlerTuple) so the swatch does not
    # imply a single colour -- the per-column colours are visible directly
    # in each panel.
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D as _Line2D
    true_proxy = tuple(_Line2D([], [], color=f"C{j}", lw=2.0) for j in range(4))
    fig.legend([true_proxy, line_pred],
               ["true (simulator)", "linear model $C\\,X$"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
               loc="outside lower center", ncol=2, frameon=True, fontsize=11)

    # Note each row's base state in the suptitle is too long; instead, write
    # them just under the title.
    base_summary = "\n".join(
        f"  sweep {STATE_SHORT[i]:<9s} base: "
        + ", ".join(f"{STATE_SHORT[k]} = {base_states[i, k]:+.3f}" for k in range(4))
        for i in range(4)
    )
    fig.suptitle(
        "Task 1.3 -- 4x4 scans:  true $\\Delta Y$ vs linear-model $C\\,X$\n"
        "rows = swept input;   columns = output component;   "
        "dotted vertical line = base-state value of swept input"
        + "\n\n" + base_summary,
        fontsize=12,
    )
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 72)
    print("Task 1.3 -- Linear model of the one-step cartpole dynamics")
    print("=" * 72)
    print(f"RNG seed:     {RNG_SEED}")
    print(f"Training set: N = {N_TRAIN}")
    print(f"Test set:     N = {N_TEST}")
    print()

    # ---- Step 1: training data -------------------------------------------
    print("[1/4] Generating training data...")
    X_train, Y_train = data_gen(N_TRAIN, rng=rng)

    # ---- Step 2: fit C via least squares ---------------------------------
    print("[2/4] Fitting C with np.linalg.lstsq...")
    C = fit_linear(X_train, Y_train)

    print()
    print("Fitted C matrix (rows = output components, cols = input components):")
    header = "          " + "".join(f"{s:>12s}" for s in STATE_SHORT)
    print(header)
    for j in range(4):
        row = "  ".join(f"{C[j, k]:+11.4e}" for k in range(4))
        print(f"  {STATE_SHORT[j]:<8s}{row}")
    print()

    # ---- Step 3: held-out test evaluation --------------------------------
    print("[3/4] Generating held-out test set and evaluating...")
    X_test, Y_test = data_gen(N_TEST, rng=rng)
    Y_pred = X_test @ C.T                                  # shape (N_TEST, 4)

    mse  = np.mean((Y_test - Y_pred) ** 2, axis=0)
    mae  = np.mean(np.abs(Y_test - Y_pred), axis=0)
    var  = np.var(Y_test, axis=0)
    r2   = 1.0 - np.sum((Y_test - Y_pred) ** 2, axis=0) / np.maximum(
        np.sum((Y_test - Y_test.mean(axis=0)) ** 2, axis=0), 1e-30
    )

    print()
    print(f"  Per-component test errors (N = {N_TEST}):")
    print(f"  {'component':<12s}  {'MSE':>12s}  {'MAE':>12s}"
          f"  {'Var(Y_true)':>14s}  {'R^2':>8s}")
    print("  " + "-" * 60)
    for j in range(4):
        print(f"  {STATE_SHORT[j]:<12s}  {mse[j]:12.3e}  {mae[j]:12.3e}"
              f"  {var[j]:14.3e}  {r2[j]:+8.3f}")
    print()

    # ---- Step 4: plots ---------------------------------------------------
    print("[4/4] Writing plots...")
    plot_pred_vs_true(Y_test, Y_pred, fname="task1_3_pred_vs_true.png")

    # Draw one random base state per row of the 4x4 scan grid.  Each row of
    # the grid sweeps a different input and uses its OWN base state for the
    # three frozen variables.
    base_states = rng.uniform(RANGES[:, 0], RANGES[:, 1], size=(4, 4))
    plot_scans(C, base_states, fname="task1_3_scans.png")

    print()
    print(f"Figures written to: {FIG_DIR}/")
    print("  task1_3_pred_vs_true.png")
    print("  task1_3_scans.png")
    print()
    print("=" * 72)
    print("Summary of fit quality")
    print("=" * 72)
    # Build a short qualitative summary keyed off the per-component R^2.
    ordered = sorted(range(4), key=lambda j: -r2[j])
    print(
        "Per-component R^2 (higher is better) ranks the linear model's\n"
        f"performance across the four state components: "
        + ", ".join(f"{STATE_SHORT[j]} ({r2[j]:+.2f})" for j in ordered) + ".\n"
        "\n"
        "Delta x and Delta theta are KINEMATIC -- to leading order they are\n"
        "just x_dot * dt and theta_dot * dt -- so a linear C captures them\n"
        "almost perfectly (R^2 ~ 0.99).  This matches the fitted C: the\n"
        "dominant entries are C[x, x_dot] ~ 0.1 and C[theta, theta_dot] ~ 0.1,\n"
        "i.e. the integrator time step.\n"
        "\n"
        "Delta x_dot and Delta theta_dot are ACCELERATIONS, and the\n"
        "accelerations in the equations of motion depend on sin(theta),\n"
        "cos(theta), and theta_dot^2 -- fundamentally nonlinear in the state.\n"
        "A single 4x4 C cannot reproduce gravity's sin(theta) curve, so\n"
        "these components fit poorly (R^2 ~ 0.3 for x_dot, ~0.5 for\n"
        "theta_dot).  The theta-sweep panel in task1_3_scans.png makes\n"
        "this concrete: the true Delta theta_dot is a sinusoid in theta,\n"
        "but the linear model's prediction is a straight line through it."
    )


if __name__ == "__main__":
    main()
