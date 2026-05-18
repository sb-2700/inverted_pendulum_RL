"""
Task 1.2 — Scanning the X -> Y functional relationship.

Treats one call to ``performAction(0.0)`` as a black-box function X -> Y and
probes it POINTWISE: pick a base state, then sweep one (or two) state
variables across a sensible range while holding the rest fixed, recording
the next state Y (or the change DeltaY = Y - X) for each sample.

Critical contrast with rollouts (Task 1.1):
    EVERY measurement RESETS all four state variables before the simulator
    step.  Each sample is an independent one-step probe of the X -> Y map,
    NOT a continuous trajectory.

Produces figures in ``<repo>/figures/``::

    task1_2_scan_Y.png            1D scans, target Y  (next state)
    task1_2_scan_dY.png           1D scans, target DY (change in state)
    contour_x_xdot.png            2D DY scans, one figure per input pair
    contour_x_theta.png             (1x4 panels: DeltaY components)
    contour_x_thetadot.png
    contour_xdot_theta.png
    contour_xdot_thetadot.png
    contour_theta_thetadot.png

Run from the repo root::

    python week1/task1_2_scans.py
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no interactive backend required
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

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
# Global plot styling
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


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------
RNG_SEED = 0
N_1D     = 201          # samples per 1D scan
N_2D     = 41           # samples per axis in each 2D scan (41 * 41 = 1681 pts)

# Sensible scan ranges from the handout (Task 1.1 hints).  Cart position is
# given a modest range; the equations of motion do not contain x at all, so
# scanning over it should produce a perfectly flat DeltaY regardless.
RANGES = [
    (-5.0,     5.0),         # x       (cart position, m)
    (-10.0,   10.0),         # x_dot   (cart velocity, m/s)
    (-np.pi,   np.pi),       # theta   (pole angle, rad)
    (-15.0,   15.0),         # theta_dot (pole angular velocity, rad/s)
]

INPUT_LABELS = [
    r"Cart position $x$ (m)",
    r"Cart velocity $\dot x$ (m/s)",
    r"Pole angle $\theta$ (rad)",
    r"Pole angular velocity $\dot\theta$ (rad/s)",
]
INPUT_SHORT  = ["x", "x_dot", "theta", "theta_dot"]

OUTPUT_LABELS_Y = [
    r"$x'$ (m)",
    r"$\dot x'$ (m/s)",
    r"$\theta'$ (rad, continuous)",
    r"$\dot\theta'$ (rad/s)",
]
OUTPUT_LABELS_DY = [
    r"$\Delta x$ (m)",
    r"$\Delta \dot x$ (m/s)",
    r"$\Delta \theta$ (rad)",
    r"$\Delta \dot\theta$ (rad/s)",
]
LINESTYLES = ["-", "--", "-.", ":"]


# ---------------------------------------------------------------------------
# Core helper -- one independent one-step probe.
# ---------------------------------------------------------------------------
#
# A single CartPole instance is reused; setState() fully overwrites the four
# state variables on every call, so there is no leakage between
# measurements.  This is the entire reason the function is named
# "one_step" rather than "step" -- every call is independent.
#
_CARTPOLE = CartPole()


def one_step(state):
    """Reset all four state variables, take ONE zero-force step, return Y.

    Each call is an independent measurement of X -> Y.  No force is
    applied.  The simulator does not remap the pole angle internally,
    so the returned Y[2] is the (possibly out-of-range) continuous angle
    -- callers that care should apply ``remap_angle`` themselves.
    """
    _CARTPOLE.setState(list(state))
    _CARTPOLE.performAction(0.0)
    return _CARTPOLE.getState()


# ---------------------------------------------------------------------------
# 1D scans
# ---------------------------------------------------------------------------
def scan_1d_raw(var_index, values, base_state):
    """Scan ``var_index`` across ``values`` (others pinned to ``base_state``);
    return the raw Y matrix of shape ``(len(values), 4)``.

    No angle remapping is applied here -- callers decide whether to remap
    Y[:, 2] (for the next-state plot) or remap Y[:, 2] - X[:, 2] (for the
    change plot).
    """
    Y = np.empty((len(values), 4))
    for i, v in enumerate(values):
        s = base_state.copy()
        s[var_index] = v
        Y[i] = one_step(s)
    return Y


def to_Y_plot(Y_raw):
    """Return Y as-is for plotting.

    The simulator advances theta continuously by at most ~theta_dot * dt
    ~ 1.5 rad per step, so Y[2] sits in roughly [-pi - 1.5, pi + 1.5].
    REMAPPING Y[2] into [-pi, pi] would CREATE artificial jumps wherever
    the natural value left that band -- the opposite of what we want.
    The trend reads more cleanly with Y[2] left continuous.
    """
    return Y_raw.copy()


def to_dY_plot(Y_raw, values, var_index, base_state):
    """Build DeltaY = Y - X for a 1D scan, with the angle component remapped
    so plotted Delta-theta has no spurious +-2*pi jumps."""
    dY = np.empty_like(Y_raw)
    for i, v in enumerate(values):
        s = base_state.copy()
        s[var_index] = v
        dY[i] = Y_raw[i] - s
    dY[:, 2] = np.array([remap_angle(t) for t in dY[:, 2]])
    return dY


def plot_scans_1d(scan_values, scans,
                  output_labels, ylabel, suptitle, fname):
    """2x2 grid: each panel scans one input variable and overlays all 4
    output components.

    Output components are drawn with both distinct colours AND distinct
    linestyles so the plot stays readable in greyscale.  Every panel has a
    dashed horizontal reference at y = 0.  A single shared legend at the
    bottom of the figure identifies the four output components.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), layout="constrained")
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        for j in range(4):
            ax.plot(scan_values[i], scans[i][:, j],
                    color=f"C{j}", lw=2.0, ls=LINESTYLES[j])
        ax.axhline(0.0, color="black", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel(INPUT_LABELS[i])
        ax.set_ylabel(ylabel)
        ax.set_title(f"Scan over {INPUT_LABELS[i]}")
        ax.grid(alpha=0.3)

    # Shared figure-level legend for the four output components.
    legend_handles = [Line2D([0], [0], color=f"C{j}", lw=2.0, ls=LINESTYLES[j])
                      for j in range(4)]
    fig.legend(legend_handles, output_labels,
               loc="outside lower center", ncol=4, frameon=True)
    fig.suptitle(suptitle)
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2D contour scans
# ---------------------------------------------------------------------------
def scan_2d(var_a, var_b, base_state, n=N_2D):
    """Scan variables ``var_a`` and ``var_b`` over an ``n x n`` grid, with
    the other two pinned to ``base_state``.  Returns flat arrays
    ``(a, b, dY)`` suitable for ``tricontourf``; ``dY`` has shape ``(N, 4)``
    and is the change vector with its theta component remapped.
    """
    grid_a = np.linspace(*RANGES[var_a], n)
    grid_b = np.linspace(*RANGES[var_b], n)
    A, B = np.meshgrid(grid_a, grid_b)
    flat_a = A.ravel()
    flat_b = B.ravel()
    N = flat_a.size

    dY = np.empty((N, 4))
    for k in range(N):
        s = base_state.copy()
        s[var_a] = flat_a[k]
        s[var_b] = flat_b[k]
        y = one_step(s)
        d = y - s
        d[2] = remap_angle(d[2])
        dY[k] = d
    return flat_a, flat_b, dY


def _contour_levels_and_cmap(z, n_levels=20):
    """Choose levels and a colormap suitable for ``z``.

    - Signed data (crosses zero): diverging ``coolwarm`` with SYMMETRIC
      levels around zero, so the zero contour reads as the colour-neutral
      band and positive/negative regions are visually distinguishable.
    - Single-sign data: perceptually-uniform ``viridis``.
    - Near-constant data (e.g. when an inert input has no effect): widen
      the levels by a tiny pad so ``tricontourf`` still draws something
      instead of erroring on degenerate levels.
    """
    vmin, vmax = float(np.min(z)), float(np.max(z))
    if vmax - vmin < 1e-12:
        pad = max(1e-10, 1e-6 * (abs(vmin) + abs(vmax) + 1.0))
        return np.linspace(vmin - pad, vmax + pad, n_levels + 1), "viridis"
    if vmin < 0.0 < vmax:
        abs_max = max(abs(vmin), abs(vmax))
        return np.linspace(-abs_max, abs_max, n_levels + 1), "coolwarm"
    return np.linspace(vmin, vmax, n_levels + 1), "viridis"


def plot_contour_pair(va, vb, a, b, dY, base_state, fname):
    """Render a single 1x4 figure for one input-variable pair.

    Each of the four panels shows one DeltaY output component over the
    scanned (var_a, var_b) plane.  The other two state variables sit at the
    base state, marked on every panel by a black/white star.  Every panel
    gets its own colorbar -- DeltaY components have very different
    magnitudes, so a shared scale would hide structure.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.0),
                             layout="constrained", squeeze=False)
    axes_row = axes[0]

    for col, ax in enumerate(axes_row):
        z = dY[:, col]
        levels, cmap = _contour_levels_and_cmap(z)
        if levels[0] < 0.0 < levels[-1]:
            norm = TwoSlopeNorm(vmin=levels[0], vcenter=0.0,
                                vmax=levels[-1])
        else:
            norm = None

        tcf = ax.tricontourf(a, b, z, levels=levels,
                             cmap=cmap, norm=norm, extend="both")
        # Star at the base-state slice point: tells the reader where the
        # other (frozen) state variables sit.
        ax.plot(base_state[va], base_state[vb], "*",
                ms=14, color="black",
                markeredgecolor="white", markeredgewidth=1.2,
                zorder=5)
        cbar = fig.colorbar(tcf, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(OUTPUT_LABELS_DY[col])

        ax.set_xlabel(INPUT_LABELS[va])
        ax.set_ylabel(INPUT_LABELS[vb])
        ax.set_title(OUTPUT_LABELS_DY[col])
        ax.grid(False)

    # The two frozen variables for this pair -- annotate the suptitle so
    # readers can compare figures without paging back to the console.
    frozen = [k for k in range(4) if k not in (va, vb)]
    frozen_desc = ", ".join(
        f"{INPUT_SHORT[k]} = {base_state[k]:+.3f}" for k in frozen
    )
    fig.suptitle(
        rf"Task 1.2 -- $\Delta Y$ over ({INPUT_SHORT[va]}, "
        rf"{INPUT_SHORT[vb]});  frozen at {frozen_desc};  "
        r"$\bigstar$ marks base-state slice."
    )
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RNG_SEED)
    base_state = np.array([rng.uniform(*r) for r in RANGES])

    print("=" * 72)
    print("Task 1.2 -- Scanning the X -> Y functional relationship")
    print("=" * 72)
    print(f"RNG seed:    {RNG_SEED}")
    print("Base state:  X = ["
          f"x={base_state[0]:+.3f}, "
          f"x_dot={base_state[1]:+.3f}, "
          f"theta={base_state[2]:+.3f} rad, "
          f"theta_dot={base_state[3]:+.3f} rad/s]")
    print()

    # ------------------------------------------------------------------
    # 1D scans -- record raw Y once, then derive both Y-plot and dY-plot
    # data from it, halving the simulator work.
    # ------------------------------------------------------------------
    print("[1/3] 1D scans (Y and DeltaY)...")
    scan_values = [np.linspace(*RANGES[i], N_1D) for i in range(4)]
    Y_raw       = [scan_1d_raw(i, scan_values[i], base_state) for i in range(4)]
    scans_Y     = [to_Y_plot(Y_raw[i])                                    for i in range(4)]
    scans_dY    = [to_dY_plot(Y_raw[i], scan_values[i], i, base_state)    for i in range(4)]

    plot_scans_1d(
        scan_values, scans_Y,
        output_labels=OUTPUT_LABELS_Y,
        ylabel="next-state component",
        suptitle=("Task 1.2 -- 1D scans, target $Y$ (next state). "
                  "Each panel scans one input; the other three are pinned "
                  "to the base state."),
        fname="task1_2_scan_Y.png",
    )
    plot_scans_1d(
        scan_values, scans_dY,
        output_labels=OUTPUT_LABELS_DY,
        ylabel=r"$\Delta Y$ component",
        suptitle=(r"Task 1.2 -- 1D scans, target $\Delta Y = Y - X$. "
                  r"Note the $x$ panel: all four $\Delta Y$ components are "
                  r"flat (cart position is inert)."),
        fname="task1_2_scan_dY.png",
    )

    # ------------------------------------------------------------------
    # 2D contour scans -- all six pairs of the four state variables.
    # Each pair becomes its own 1x4 figure with one panel per DeltaY
    # component.  The base state is fixed (seeded RNG) so every figure
    # uses the same frozen values for its two non-swept variables.
    # ------------------------------------------------------------------
    print("[2/3] 2D contour scans (DeltaY): 6 pairs...")
    PAIRS = [
        (0, 1, "contour_x_xdot.png"),
        (0, 2, "contour_x_theta.png"),
        (0, 3, "contour_x_thetadot.png"),
        (1, 2, "contour_xdot_theta.png"),
        (1, 3, "contour_xdot_thetadot.png"),
        (2, 3, "contour_theta_thetadot.png"),
    ]
    contour_filenames = []
    for va, vb, fn in PAIRS:
        a, b, dY = scan_2d(va, vb, base_state)
        plot_contour_pair(va, vb, a, b, dY, base_state, fname=fn)
        contour_filenames.append(fn)
        print(f"  wrote {fn}  "
              f"(pair: {INPUT_SHORT[va]}, {INPUT_SHORT[vb]})")

    # ------------------------------------------------------------------
    # Inert-variable analysis: numerical confirmation that the 1D scan
    # over x produces a perfectly flat DeltaY in every component.
    # ------------------------------------------------------------------
    print("[3/3] Inert-variable check...")
    print()
    print(f"  Max variation (peak-to-peak range) of each Delta Y component"
          f" across each 1D scan:")
    header = (f"  {'scanned input':<14s}"
              f"  {'dY[0]':>12s}  {'dY[1]':>12s}"
              f"  {'dY[2]':>12s}  {'dY[3]':>12s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    ptp_rows = []
    for i in range(4):
        row = np.ptp(scans_dY[i], axis=0)
        ptp_rows.append(row)
        print(f"  {INPUT_SHORT[i]:<14s}"
              f"  {row[0]:12.3e}  {row[1]:12.3e}"
              f"  {row[2]:12.3e}  {row[3]:12.3e}")
    print()

    x_max  = float(np.max(ptp_rows[0]))
    th_max = float(np.max(ptp_rows[2]))
    print(f"  Across the x scan, max range of any Delta Y component "
          f"= {x_max:.3e}")
    print(f"  For comparison, the theta scan reaches "
          f"{th_max:.3e} -- ~{th_max / max(x_max, 1e-30):.1e}x bigger.")
    print()

    print("=" * 72)
    print("INERT VARIABLE: cart position x")
    print("=" * 72)
    print(
        "Cart position x does NOT appear on the right-hand side of either\n"
        "equation of motion (only x_dot, theta, theta_dot and the applied\n"
        "force enter the pole and cart accelerations).  Therefore the\n"
        "one-step change Delta Y = X(T) - X(0) is independent of x.\n"
        "\n"
        "Evidence:\n"
        "  (a) the 1D scan over x: max peak-to-peak variation of any\n"
        f"      Delta Y component is {x_max:.2e} (machine precision);\n"
        "  (b) the three contour figures that include x as a swept\n"
        "      variable -- contour_x_xdot.png, contour_x_theta.png and\n"
        "      contour_x_thetadot.png -- show contours that are perfectly\n"
        "      horizontal stripes: Delta Y depends only on the OTHER\n"
        "      swept variable, never on x.\n"
        "\n"
        "Note: Y[0] = x' = x + Delta x DOES depend on x trivially -- the\n"
        "next-state cart position carries the initial position forward.\n"
        "It is the CHANGE Delta Y, not Y itself, that is independent of x.\n"
        "This is exactly why switching the regression target from Y to\n"
        "Delta Y is a good idea: it strips out a trivial dependence and\n"
        "leaves only the dynamics worth fitting."
    )
    print()
    print(f"Figures written to: {FIG_DIR}/")
    for fn in ("task1_2_scan_Y.png", "task1_2_scan_dY.png", *contour_filenames):
        print(f"  {fn}")


if __name__ == "__main__":
    main()
