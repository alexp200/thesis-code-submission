"""Individual models of the conservative sPHNN stay on closed orbits (Appendix B, Figure B.2; supports the statement made next to Figure 4.5).

State-space plot of the retained members of the sPHNN ensemble over the 15 simulated periods, for lower half and left half: no vector-field uncertainty and no predictive mean.
The orbits are the stored member rollouts (long_horizon/results/cache_sPHNN_{dataset}/la_k*.npz; members kept by the training-RMSE filter of methods_sPHNN_{dataset}.json), nothing is recomputed;
the true orbit is drawn for one period (black dotted) and the circle marks x0. p is on the horizontal and q on the vertical axis (swapped with respect to Figure 4.5), so that the two panels are
landscape at equal aspect. 15 periods are 15 periods of the TRUE orbit; the members have their own periods and complete 13 to 16 loops.
Output: the export folder (imgs/appendix/a12_member_orbits_phase_space), drawn at the inclusion width of figure_widths.json. Usage: python make_member_orbits.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import export_paths as EP
import make_figures as mf                              # RESULT_DIR, T_TRUE, DOTTED (true orbit), true_trajectory; importing it writes nothing
import plot_lib as pl                                 # thesis plot style (pl.tps, pl.style)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

tps, style = pl.tps, pl.style
KEY = "a12_member_orbits_phase_space"
EXPORT = EP.IMG_APP / KEY
GEN = "02_harmonic_oscillator/figures/make_member_orbits.py"
MODEL = "sPHNN"
DATASETS = (("lower_half", "lower half"), ("left_half", "left half"))
MEMBER = dict(color=style.MEMBER["color"], lw=0.5, alpha=0.9)          # thesis style of single members, solid: dotted lines fill in over many loops
LIM_P, LIM_Q = (-5.5, 5.5), (-4.0, 4.0)                                # horizontal / vertical window
TICKS_P, TICKS_Q = [-5, 0, 5], [-4, -2, 0, 2, 4]
LEFT_MM, RIGHT_MM, GAP_MM, BOTTOM_MM, TOP_MM = 10.2, 1.5, 11.0, 10.5, 5.0     # margins around the axes (mm)


def load_members(dataset):
    """(members (n, n_time, 2), x0, t): the retained sPHNN members over the 15 simulated periods, as stored."""
    meta = json.loads((mf.RESULT_DIR / f"methods_{MODEL}_{dataset}.json").read_text())
    t = np.load(mf.RESULT_DIR / f"methods_{MODEL}_{dataset}.npz")["t"]
    assert abs(t[-1] - 15 * mf.T_TRUE) < 1e-6 * t[-1], "the stored horizon is not 15 periods of the true orbit"
    members = np.stack([np.load(mf.RESULT_DIR / f"cache_{MODEL}_{dataset}" / f"la_k{k}.npz")["mu"] for k in meta["filter"]["kept"]])
    return members, np.asarray(meta["x0"]), t


def draw():
    tps.apply_style(KEY)
    mm = 1 / 25.4
    W = tps.sizes.width_in(KEY)
    n = len(DATASETS)
    pw = W - (LEFT_MM + RIGHT_MM + (n - 1) * GAP_MM) * mm
    pw /= n
    ph = pw * (LIM_Q[1] - LIM_Q[0]) / (LIM_P[1] - LIM_P[0])                 # equal aspect
    H = (BOTTOM_MM + TOP_MM) * mm + ph
    fig = plt.figure(figsize=(W, H))
    for c, (dataset, heading) in enumerate(DATASETS):
        ax = fig.add_axes([(LEFT_MM + c * (pw / mm + GAP_MM)) * mm / W, BOTTOM_MM * mm / H, pw / W, ph / H])
        members, x0, t = load_members(dataset)
        for traj in members:
            ax.plot(traj[:, 1], traj[:, 0], zorder=3, **MEMBER)
        truth = mf.true_trajectory(x0, t[t <= mf.T_TRUE])
        ax.plot(truth[:, 1], truth[:, 0], zorder=4, **mf.DOTTED)
        ax.plot([x0[1]], [x0[0]], "o", mfc="white", mec="k", mew=0.6, ms=3, zorder=5)
        ax.set_xlim(*LIM_P)
        ax.set_ylim(*LIM_Q)
        ax.set_xticks(TICKS_P)
        ax.set_yticks(TICKS_Q)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_xlabel(r"$p$")
        ax.set_ylabel(r"$q$")
        tps.col_heading(ax, heading)
    handles = [Line2D([], [], label="individual models (15 periods)", **MEMBER), Line2D([], [], label="true orbit (one period)", **mf.DOTTED),
               Line2D([], [], ls="", marker="o", mfc="white", mec="k", mew=0.6, ms=4, label=r"$x_0$")]
    tps.legend_above(fig, handles, len(handles), columnspacing=1.6)
    return fig


if __name__ == "__main__":
    tps.save_figure(draw(), EXPORT, generator=GEN)
