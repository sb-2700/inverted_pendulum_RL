"""
Task 1.1 — CartPole rollouts and visualisation.

Simulates the FREE dynamics (no applied force) of the cartpole starting from a
range of initial conditions, and produces time-evolution and phase-portrait
plots that show the qualitative behaviours of the system.

Run from the repository root:
    python week1/task1_1_rollouts.py

Figures are written to <repo>/figures/.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless: do not require an interactive backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# Make cartpole.py importable when running from the repo root or week1/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cartpole import CartPole, remap_angle  # noqa: E402

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def rollout(initial_state, n_steps):
    """Run the free dynamics for n_steps starting from initial_state.

    State vector is [x, x_dot, theta, theta_dot].  No force is applied.
    The simulator does NOT remap the pole angle internally, so theta in the
    returned trajectory is the continuous (unwrapped) angle.

    Returns a (n_steps + 1, 4) NumPy array; row 0 is initial_state.
    """
    cp = CartPole()
    cp.setState(list(initial_state))
    traj = np.empty((n_steps + 1, 4))
    traj[0] = np.asarray(initial_state, dtype=float)
    for i in range(1, n_steps + 1):
        cp.performAction(0.0)        # 0 force = free dynamics
        traj[i] = cp.getState()
    return traj


def remap_trajectory(traj):
    """Return a copy of traj with theta (column 2) wrapped into [-pi, pi]."""
    out = traj.copy()
    out[:, 2] = np.array([remap_angle(t) for t in traj[:, 2]])
    return out


def _wrap_with_breaks(theta, theta_dot):
    """Wrap theta into [-pi, pi] and insert NaN at wrap jumps so phase
    portraits of rotating trajectories do not draw a horizontal line across
    the figure at each ±pi crossing."""
    wrapped = np.array([remap_angle(t) for t in theta])
    th_out, td_out = [wrapped[0]], [theta_dot[0]]
    for i in range(1, len(wrapped)):
        if abs(wrapped[i] - wrapped[i - 1]) > np.pi:
            th_out.append(np.nan)
            td_out.append(np.nan)
        th_out.append(wrapped[i])
        td_out.append(theta_dot[i])
    return np.array(th_out), np.array(td_out)


def _wrap_with_breaks_t(theta, theta_dot, t):
    """As `_wrap_with_breaks`, but also carries a third array `t` aligned
    with the wrapped output; needed when colouring the phase trajectory by
    time."""
    wrapped = np.array([remap_angle(x) for x in theta])
    th_out, td_out, t_out = [wrapped[0]], [theta_dot[0]], [t[0]]
    for i in range(1, len(wrapped)):
        if abs(wrapped[i] - wrapped[i - 1]) > np.pi:
            th_out.append(np.nan)
            td_out.append(np.nan)
            t_out.append(np.nan)
        th_out.append(wrapped[i])
        td_out.append(theta_dot[i])
        t_out.append(t[i])
    return np.array(th_out), np.array(td_out), np.array(t_out)


# ---------------------------------------------------------------------------
# Shared plotting constants and helpers
# ---------------------------------------------------------------------------
TITLES = ["Cart position",
          "Cart velocity",
          "Pole angle (unwrapped)",
          "Pole angular velocity"]
YLABELS = [r"$x$ (m)",
           r"$\dot x$ (m/s)",
           r"$\theta$ (rad)",
           r"$\dot\theta$ (rad/s)"]

DT = CartPole().delta_time          # seconds per outer step (0.1 s)

# Reusable legend handle for the start-of-rollout dot.
_START_HANDLE = Line2D([0], [0], marker="o", ls="", color="0.35", ms=6,
                       label=r"start ($t=0$)")


def classify_pole(traj):
    """Loose classification of pole motion from the unwrapped angle trace."""
    th = traj[:, 2]
    span = th.max() - th.min()
    if abs(th[-1] - th[0]) > 2 * np.pi:
        return f"rotation (total dtheta = {th[-1] - th[0]:+.2f} rad over the run)"
    if span < 1.0:
        return f"small oscillation (peak deviation {span / 2:.2f} rad)"
    return f"large oscillation (peak deviation {span / 2:.2f} rad)"


def _classify_compact(traj):
    """Compact two-line classification, formatted to fit inside a rotated
    row label.  Same logic as classify_pole()."""
    th = traj[:, 2]
    span = th.max() - th.min()
    if abs(th[-1] - th[0]) > 2 * np.pi:
        return f"rotation\n$\\Delta\\theta={th[-1] - th[0]:+.1f}$ rad"
    if span < 1.0:
        return f"small osc.\npeak {span / 2:.3f} rad"
    return f"large osc.\npeak {span / 2:.2f} rad"


def _is_rotation(traj):
    return abs(traj[-1, 2] - traj[0, 2]) > 2 * np.pi


def _has_meaningful_pole_motion(traj, threshold=0.05):
    """True iff the pole moves at least `threshold` rad over the rollout.

    Used to filter out cases where the pole is effectively stationary (e.g.
    ẋ-only initial conditions, where the pole stays at the stable equilibrium
    to within ~10⁻³ rad).  Drawing a per-case phase portrait for such a case
    is uninformative — the trajectory is just a dot — so we skip the panel.
    """
    th = traj[:, 2]
    return (th.max() - th.min()) > threshold


# ---------------------------------------------------------------------------
# Time-evolution: one row per initial condition, very wide panels
# ---------------------------------------------------------------------------
def plot_time_grid(trajectories, title, fname,
                   theta_ylim=None, theta_dot_ylim=None,
                   annotate_theta_dev=True):
    """Plot state-vs-time as an (n_cases x 4) grid: one row per initial
    condition, one column per state variable.  Every panel autoscales
    independently so no trace is hidden behind another at an incompatible
    scale.  Panels are made deliberately wide so individual oscillation
    cycles are resolvable over a full 20 s run.

    The UNWRAPPED θ trajectory is used so that oscillations and full
    rotations read clearly.  `theta_ylim`, if given, pins the θ-column
    y-limits — useful when the pole barely moves and autoscale would
    exaggerate a non-effect.  When `annotate_theta_dev` is true, each θ panel
    is annotated with the peak |θ − π| (for libration) or the net Δθ (for
    rotation).
    """
    n = len(trajectories)
    fig, axes = plt.subplots(n, 4, figsize=(22.0, 2.4 * n + 0.9),
                             squeeze=False)
    for i, (label, traj) in enumerate(trajectories):
        t = np.arange(traj.shape[0]) * DT
        color = f"C{i}"
        for j in range(4):
            ax = axes[i, j]
            ax.plot(t, traj[:, j], lw=1.3, color=color)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(TITLES[j], fontsize=11)
            if i == n - 1:
                ax.set_xlabel("time (s)")
            ax.set_ylabel(YLABELS[j], fontsize=9)
            if j == 2:
                ax.axhline(np.pi, color="grey", ls="--", lw=0.7)
                if theta_ylim is not None:
                    ax.set_ylim(*theta_ylim)
                if annotate_theta_dev:
                    th = traj[:, 2]
                    if _is_rotation(traj):
                        ann = rf"net $\Delta\theta$ = {th[-1] - th[0]:+.1f} rad"
                    else:
                        peak = float(np.max(np.abs(th - np.pi)))
                        ann = rf"peak $|\theta - \pi|$ = {peak:.4f} rad"
                    ax.text(0.02, 0.95, ann,
                            transform=ax.transAxes, va="top", ha="left",
                            fontsize=8,
                            bbox=dict(facecolor="white", alpha=0.75,
                                      edgecolor="none", pad=2))
            if j == 3 and theta_dot_ylim is not None:
                ax.set_ylim(*theta_dot_ylim)

    fig.suptitle(title, fontsize=12, y=0.995)
    # Reserve ~5% on the left for the vertical row labels.
    fig.tight_layout(rect=[0.05, 0, 1, 0.96])

    # Place row labels in the reserved left margin, after layout has settled.
    # Use a compact, multi-line classification so the rotated text fits
    # comfortably inside a single row's vertical extent.
    for i, (label, traj) in enumerate(trajectories):
        pos = axes[i, 0].get_position()
        y_mid = 0.5 * (pos.y0 + pos.y1)
        row_label = f"{label}\n{_classify_compact(traj)}"
        fig.text(0.015, y_mid, row_label,
                 rotation=90, va="center", ha="center",
                 fontsize=8.5, fontweight="bold")

    fig.savefig(FIG_DIR / fname, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase-portrait helpers (single-axis primitives)
# ---------------------------------------------------------------------------
def _draw_pole_phase_panel(ax, cases, panel_title,
                           show_start_in_legend=True):
    """Draw a pole phase portrait into a single axis.  The initial state of
    each rollout is marked with a dot; if `show_start_in_legend` is true an
    extra dot-only handle labelled "start (t=0)" is appended to the legend."""
    for label, traj in cases:
        th_w, td_w = _wrap_with_breaks(traj[:, 2], traj[:, 3])
        line, = ax.plot(th_w, td_w, lw=1.0, label=label, alpha=0.9)
        ax.plot(remap_angle(traj[0, 2]), traj[0, 3], "o",
                ms=6, color=line.get_color())
    ax.set_title(panel_title)
    ax.set_xlabel(r"$\theta$ (rad, remapped to $[-\pi,\pi]$)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.axvline( np.pi, color="grey", ls="--", lw=0.8)
    ax.axvline(-np.pi, color="grey", ls="--", lw=0.8)
    ax.axvline(0.0,    color="red",  ls=":",  lw=0.8,
               label=r"upright ($\theta=0$)")
    ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if show_start_in_legend:
        handles.append(_START_HANDLE)
        labels.append(_START_HANDLE.get_label())
    ax.legend(handles=handles, labels=labels, fontsize=8, loc="best")


# ---------------------------------------------------------------------------
# (3a) GLOBAL phase portrait — many short rollouts on one axis
# ---------------------------------------------------------------------------
SHORT_N_STEPS = 35     # ~3.5 s — ~3 small-oscillation periods, a few rotations.

# Initial θ̇ values sampled for the global phase portrait.  Coverage spans the
# small libration regime up through clear rotation, with both signs.
GLOBAL_LIB_THETADOTS = [-10.0, -7.0, -4.0, -1.0, 1.0, 4.0, 7.0, 10.0]
GLOBAL_ROT_THETADOTS = [-16.0, -14.0, 14.0, 16.0]


def plot_phase_global(fname,
                      title=("Global pole phase portrait: many short rollouts "
                             r"from $X=[0,0,\pi,\dot\theta_0]$ over a range of "
                             r"initial $\dot\theta$")):
    """Combined (θ, θ̇) phase portrait built from many short rollouts.

    Each rollout starts at the stable equilibrium with a different initial
    pole angular velocity and runs for SHORT_N_STEPS steps — long enough to
    trace a clean arc/loop, short enough that the friction-driven inward
    spiral has not yet collapsed the curve.  Together the rollouts reveal
    the libration region around (±π, 0), the rotation region (high |θ̇|),
    and the separatrix between them.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    lib_cmap = plt.get_cmap("Blues")
    rot_cmap = plt.get_cmap("Reds")

    def _plot_one(td0, color):
        traj = rollout([0.0, 0.0, np.pi, td0], SHORT_N_STEPS)
        th_w, td_w = _wrap_with_breaks(traj[:, 2], traj[:, 3])
        ax.plot(th_w, td_w, color=color, lw=1.2, alpha=0.9)
        ax.plot(remap_angle(traj[0, 2]), traj[0, 3], "o",
                color=color, ms=5)

    for k, td0 in enumerate(GLOBAL_LIB_THETADOTS):
        c = lib_cmap(0.35 + 0.55 * k / max(1, len(GLOBAL_LIB_THETADOTS) - 1))
        _plot_one(td0, c)
    for k, td0 in enumerate(GLOBAL_ROT_THETADOTS):
        c = rot_cmap(0.40 + 0.50 * k / max(1, len(GLOBAL_ROT_THETADOTS) - 1))
        _plot_one(td0, c)

    ax.axvline( np.pi, color="grey",  ls="--", lw=0.9)
    ax.axvline(-np.pi, color="grey",  ls="--", lw=0.9)
    ax.axvline( 0.0,   color="black", ls=":",  lw=0.9)
    ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
    ax.set_xlabel(r"$\theta$ (rad, remapped to $[-\pi,\pi]$)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=11)

    td_lib_label = ", ".join(f"{v:g}" for v in GLOBAL_LIB_THETADOTS)
    td_rot_label = ", ".join(f"{v:g}" for v in GLOBAL_ROT_THETADOTS)
    handles = [
        Line2D([0], [0], color=lib_cmap(0.75), lw=1.6,
               label=rf"libration ($\dot\theta_0\in\{{{td_lib_label}\}}$)"),
        Line2D([0], [0], color=rot_cmap(0.75), lw=1.6,
               label=rf"rotation ($\dot\theta_0\in\{{{td_rot_label}\}}$)"),
        _START_HANDLE,
        Line2D([0], [0], color="grey",  ls="--", lw=0.9,
               label=r"hanging ($\theta=\pm\pi$)"),
        Line2D([0], [0], color="black", ls=":",  lw=0.9,
               label=r"upright ($\theta=0$)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (3b) PER-CASE phase portraits — one trajectory per axes, full 20 s,
#      coloured by time so the friction-driven inward spiral is legible.
#      Cart (x, ẋ) phase portraits removed entirely — the cart drifts
#      monotonically, so an (x, ẋ) plot has no orbit structure; the cart
#      time-evolution panels carry the same information.
# ---------------------------------------------------------------------------
def plot_phase_per_case_pole_time_coloured(trajectories, title, fname,
                                           theta_dot_ylim=None,
                                           cmap_name="viridis"):
    """Per-case pole phase portrait grid with each trajectory coloured by
    time (start → dark, end → bright).  Energy is dissipated to the cart
    through friction, so the trajectory spirals inward in the (θ, θ̇) plane;
    a time colormap makes that decay legible where a single-colour line
    would not.

    Cases with negligible pole motion (peak |Δθ| < ~0.05 rad over the run)
    should be filtered out by the caller — drawing them produces an empty
    panel.  `theta_dot_ylim` pins the θ̇ axis range when given (used for the
    coupled case to make the surviving panels share a comparable scale).
    """
    n = len(trajectories)
    if n == 0:
        return     # nothing to plot — caller should print a note

    if n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), squeeze=False)
        axes_list = list(axes.flatten())
    else:
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.0), squeeze=False)
        axes_list = list(axes[0])

    cmap = plt.get_cmap(cmap_name)
    t_max = N_STEPS * DT
    norm = plt.Normalize(vmin=0.0, vmax=t_max)
    last_lc = None

    for ax, (label, traj) in zip(axes_list, trajectories):
        t = np.arange(traj.shape[0]) * DT
        th_w, td_w, t_w = _wrap_with_breaks_t(traj[:, 2], traj[:, 3], t)

        # Build per-segment LineCollection coloured by mid-segment time.
        # Segments containing a NaN endpoint (i.e. wrap breaks) are skipped
        # by LineCollection automatically.
        points = np.column_stack([th_w, td_w]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_t = 0.5 * (t_w[:-1] + t_w[1:])
        lc = LineCollection(segments, cmap=cmap, norm=norm,
                            linewidths=1.4, alpha=0.95)
        lc.set_array(seg_t)
        ax.add_collection(lc)
        last_lc = lc

        # Start (dark) and end (bright) markers — also keyed off the cmap.
        ax.plot(remap_angle(traj[0, 2]), traj[0, 3], "o",
                ms=8, color=cmap(0.0), mec="black", mew=0.8, zorder=4)
        ax.plot(remap_angle(traj[-1, 2]), traj[-1, 3], "s",
                ms=8, color=cmap(1.0), mec="black", mew=0.8, zorder=4)

        ax.set_title(f"{label}\n[{classify_pole(traj)}]", fontsize=10)
        ax.set_xlabel(r"$\theta$ (rad, remapped to $[-\pi,\pi]$)")
        ax.set_ylabel(r"$\dot\theta$ (rad/s)")
        ax.axvline( np.pi, color="grey", ls="--", lw=0.8)
        ax.axvline(-np.pi, color="grey", ls="--", lw=0.8)
        ax.axvline(0.0,    color="red",  ls=":",  lw=0.8)
        ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
        ax.grid(alpha=0.3)
        if theta_dot_ylim is not None:
            ax.set_ylim(*theta_dot_ylim)
        else:
            valid = ~np.isnan(td_w)
            ymin, ymax = float(np.min(td_w[valid])), float(np.max(td_w[valid]))
            pad = 0.1 * (ymax - ymin) + 0.05
            ax.set_ylim(ymin - pad, ymax + pad)

        legend_handles = [
            Line2D([0], [0], marker="o", ls="", color=cmap(0.0),
                   mec="black", mew=0.8, ms=7, label=r"start ($t=0$)"),
            Line2D([0], [0], marker="s", ls="", color=cmap(1.0),
                   mec="black", mew=0.8, ms=7,
                   label=rf"end ($t={t_max:.0f}$ s)"),
            Line2D([0], [0], color="red",  ls=":",  lw=0.9,
                   label=r"upright ($\theta=0$)"),
            Line2D([0], [0], color="grey", ls="--", lw=0.9,
                   label=r"hanging ($\theta=\pm\pi$)"),
        ]
        ax.legend(handles=legend_handles, fontsize=7.5, loc="best")

    # Hide any unused subplot slot (e.g. n=3 case ends up with axes[3] unused
    # only if we used a 2×2 grid — currently n=3 uses 1×3 so this is a no-op,
    # but kept for safety if the layout choices ever change).
    for ax in axes_list[len(trajectories):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12)
    # Leave room on the right for a shared colorbar.
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    cbar_ax = fig.add_axes([0.935, 0.12, 0.014, 0.76])
    cbar = fig.colorbar(last_lc, cax=cbar_ax)
    cbar.set_label("time (s)")

    fig.savefig(FIG_DIR / fname, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
N_STEPS = 200          # 200 * 0.1 s = 20 s — several oscillation periods


def experiment_theta_dot():
    """Vary the INITIAL pole angular velocity across separate rollouts,
    starting from the stable down equilibrium X = [0, 0, π, θ̇₀].

    With the cart free to move, the cart absorbs energy from the pole,
    raising the rotation threshold above the cart-fixed estimate
    θ̇ ≈ √(6g/L) ≈ 10.8 rad/s.  Empirically θ̇₀ ≈ 13 is the boundary; 10
    sits well below it (still libration), 15 well above it (clear rotation).
    """
    cases = [
        (r"$\dot\theta_0=1.0$",  [0.0, 0.0, np.pi,  1.0]),
        (r"$\dot\theta_0=5.0$",  [0.0, 0.0, np.pi,  5.0]),
        (r"$\dot\theta_0=10.0$", [0.0, 0.0, np.pi, 10.0]),
        (r"$\dot\theta_0=15.0$", [0.0, 0.0, np.pi, 15.0]),
    ]
    trajs = [(lbl, rollout(s, N_STEPS)) for lbl, s in cases]

    plot_time_grid(
        trajs,
        (r"Varying initial $\dot\theta$ across rollouts "
         r"(starting state $X=[0,0,\pi,\dot\theta_0]$)"),
        "task1_1_time_thetadot.png",
    )
    # Per-case pole phase portraits with trajectories coloured by time so
    # the friction-driven inward spiral over the full 20 s is visible.
    # Cart (x, ẋ) phase portraits are not produced — the cart drifts
    # monotonically and the cart time-evolution panels already cover it.
    plot_phase_per_case_pole_time_coloured(
        trajs,
        (r"Per-case pole phase portraits, time-coloured (full 20 s) — "
         r"varying initial $\dot\theta$"),
        "task1_1_phase_per_case_thetadot.png",
    )
    return trajs


def experiment_x_dot():
    """Vary the INITIAL cart velocity across separate rollouts, with the pole
    at rest at the stable down equilibrium: X = [0, ẋ₀, π, 0].

    With the pole hanging straight down the cart mostly coasts (decelerated
    only by the small friction μ_c = 0.001); cart–pole coupling means the
    finite cart velocity nonetheless perturbs the pole at the 10⁻³ rad level.
    """
    cases = [
        (r"$\dot x_0=1.0$",  [0.0,  1.0, np.pi, 0.0]),
        (r"$\dot x_0=3.0$",  [0.0,  3.0, np.pi, 0.0]),
        (r"$\dot x_0=7.0$",  [0.0,  7.0, np.pi, 0.0]),
        (r"$\dot x_0=10.0$", [0.0, 10.0, np.pi, 0.0]),
    ]
    trajs = [(lbl, rollout(s, N_STEPS)) for lbl, s in cases]

    # Time grid — pin θ y-limits to π ± 0.1 so the negligible pole wiggle
    # (peak |θ − π| < 0.002 rad) reads as flat and is not autoscaled into a
    # fake-dramatic oscillation.
    plot_time_grid(
        trajs,
        (r"Varying initial $\dot x$ across rollouts "
         r"(starting state $X=[0,\dot x_0,\pi,0]$)"),
        "task1_1_time_xdot.png",
        theta_ylim=(np.pi - 0.1, np.pi + 0.1),
        theta_dot_ylim=(-0.1, 0.1),
    )
    # Per-case pole phase portraits intentionally NOT produced for this
    # experiment: all 4 cases have negligible pole motion (peak |Δθ| < 0.05
    # rad over the full run), so every panel would be empty.  The cart
    # time-evolution panels already show what the cart is doing; the cart
    # phase portrait is dropped per request.
    return trajs


def experiment_combined():
    """Coupled case: both ẋ and θ̇ non-zero initially, plus the single-
    variable references for comparison."""
    cases = [
        (r"only $\dot\theta_0=4$",            [0.0, 0.0, np.pi, 4.0]),
        (r"only $\dot x_0=5$",                [0.0, 5.0, np.pi, 0.0]),
        (r"$\dot x_0=5,\ \dot\theta_0=4$",    [0.0, 5.0, np.pi, 4.0]),
    ]
    trajs = [(lbl, rollout(s, N_STEPS)) for lbl, s in cases]

    plot_time_grid(
        trajs,
        "Coupled cart + pole motion — varying initial $\\dot x$ and $\\dot\\theta$",
        "task1_1_time_combined.png",
    )
    # Per-case pole phase portraits, time-coloured, full 20 s.  Filter out
    # cases with negligible pole motion (the only-ẋ case here) — drawing
    # them produces an empty panel.  Cart phase portraits are not produced.
    phase_trajs = [(lbl, tr) for (lbl, tr) in trajs
                   if _has_meaningful_pole_motion(tr)]
    skipped = [lbl for (lbl, tr) in trajs
               if not _has_meaningful_pole_motion(tr)]
    if skipped:
        print(f"  (pole phase panel omitted for: {', '.join(skipped)} -- "
              f"pole effectively stationary)")
    plot_phase_per_case_pole_time_coloured(
        phase_trajs,
        "Per-case pole phase portraits, time-coloured (full 20 s) — coupled case",
        "task1_1_phase_per_case_combined.png",
    )
    return trajs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"dt = {DT} s, n_steps = {N_STEPS} -> simulated {N_STEPS * DT:.1f} s\n")

    print("== Global phase portrait (many short rollouts) ==")
    plot_phase_global("task1_1_phase_global.png")
    print(f"  wrote task1_1_phase_global.png "
          f"({len(GLOBAL_LIB_THETADOTS)} librating + "
          f"{len(GLOBAL_ROT_THETADOTS)} rotating, {SHORT_N_STEPS} steps each)")

    print("\n== Experiment 1: varying initial theta_dot ==")
    trajs1 = experiment_theta_dot()
    for lbl, tr in trajs1:
        print(f"  {lbl:40s} -> {classify_pole(tr)}")

    print("\n== Experiment 2: varying initial x_dot ==")
    trajs2 = experiment_x_dot()
    for lbl, tr in trajs2:
        peak_th = float(np.max(np.abs(tr[:, 2] - np.pi)))
        drift   = float(tr[-1, 0] - tr[0, 0])
        print(f"  {lbl:40s} -> pole peak deviation from pi = {peak_th:.3f} rad,"
              f" cart drift = {drift:+.2f} m")
    print("  (per-case pole phase portrait omitted: pole effectively stationary "
          "in all x_dot-only cases)")

    print("\n== Experiment 3: coupled ==")
    trajs3 = experiment_combined()
    for lbl, tr in trajs3:
        print(f"  {lbl:40s} -> {classify_pole(tr)}")

    print(f"\nFigures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
