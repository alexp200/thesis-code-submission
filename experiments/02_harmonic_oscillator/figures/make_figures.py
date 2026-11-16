"""Harmonic oscillator: long-rollout phase-space figure (Figure 4.5) and short-rollout figures of NODE and sPHNN (Figures 4.3, 4.4, B.4, B.5); paths from common/export_paths.py.

  fig04_long_phase_space_lower_half_v2
      rows NODE (top) / sPHNN (bottom), columns LA, MoLA, DE, SWAG, MultiSWAG; background = vector-field uncertainty (own colour scale and colour bar per panel),
      orange predictive mean over 15 periods, circle at x0, true orbit = black dotted line for one period.
  fig01/fig02 (lower half) and a01/a02 (left half) _short_{model}_{dataset}_wide_v2
      function-space panel on top, vector-field panel below; the fields on q in [-10, 10], p in [-8, 8] come from short_horizon/compute_short_rollout.py.
Typography at the final document size (thesis_plot_style), method and architecture names as headings outside the axes, widths from
thesis_plot_style/figure_widths.json.
Usage: python -u make_figures.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "common"))
import export_paths as EP
import plot_lib as pl

import numpy as np
from matplotlib.lines import Line2D

MODELS = ("NODE", "sPHNN")
RESULT_DIR = ROOT / "long_horizon" / "results"
COMP_DIR = ROOT / "swag_dependence" / "comparison"
DATA_DIR = ROOT / "short_horizon" / "data"
K_SPRING, MASS = 2.0, 1.0
T_TRUE = 2 * np.pi / np.sqrt(2.0)
DOTTED = dict(color="k", ls=(0, (1, 2.2)), lw=1.1)       # true orbit, one period, black dotted
GEN = "02_harmonic_oscillator/figures/make_figures.py"


# ---------------------------------------------------------------- stored results
def true_trajectory(x0, t):
    """Analytic solution of the oscillator (q, p) for the initial state x0."""
    w = np.sqrt(K_SPRING / MASS)
    q = x0[0] * np.cos(w * t) + x0[1] / (MASS * w) * np.sin(w * t)
    p = x0[1] * np.cos(w * t) - MASS * w * x0[0] * np.sin(w * t)
    return np.stack([q, p], axis=-1)


def load_long(model, dataset):
    """Predictive means over 15 periods, the true orbit and the vector-field uncertainty on q in [-10, 10]."""
    z = dict(np.load(RESULT_DIR / f"methods_{model}_{dataset}.npz"))
    meta = json.loads((RESULT_DIR / f"methods_{model}_{dataset}.json").read_text())
    swag = np.load(COMP_DIR / f"{model}_{dataset}.npz")
    for m in ("SWAG", "MultiSWAG"):
        z[f"{m}__mean"] = swag[f"{m}__mean"]
    z["x0"] = np.asarray(meta["x0"])
    z["truth"] = true_trajectory(z["x0"], z["t"])
    short = np.load(DATA_DIR / f"{model}_{dataset}.npz")
    z["grid_q"], z["grid_p"] = short["grid_q"], short["grid_p"]
    for m in pl.METHODS:
        z[f"{m}__field_unc"] = short[f"{m}__field_unc"]
    return z


def left_drawer(d):
    """Function-space panel (means and +-2 std bands of one method) from the stored short-rollout data."""
    obs = (d["obs_t"], d["obs_y"])
    return lambda ax, m: pl.draw_function_space(ax, d["t"], d["truth"], d[f"{m}__mean"], d[f"{m}__std"], train_t_end=float(d["train_t_end"]), obs=obs)


# ---------------------------------------------------------------- figures
def merged_phase_space(ds, key):
    """Long-rollout phase-space figure of both architectures on one data set."""
    zs = {m: load_long(m, ds) for m in MODELS}
    spec = pl.color_spec("own")

    def draw(ax, r, m):
        z = zs[MODELS[r]]
        one_period = z["t"] <= T_TRUE
        return pl.draw_phase_space(ax, z["grid_q"], z["grid_p"], z[f"{m}__field_unc"], spec, z[f"{m}__mean"], z["truth"][one_period], z["x0"], truth_kw=DOTTED, wide=True)

    handles = [Line2D([], [], color=pl.MEAN_COLOR, lw=1.4, label="predictive mean (15 periods)"),
               Line2D([], [], label="true orbit (one period)", **DOTTED),
               Line2D([], [], ls="", marker="o", mfc="white", mec="k", mew=0.6, ms=4, label=r"$x_0$")]
    return pl.phase_grid_figure_thesis(draw, spec, list(MODELS), handles, key, wide=True)


def short_ott(model, ds, key):
    """Short-rollout figure: function-space panel above, vector-field panel below on q in [-10, 10]."""
    d = dict(np.load(DATA_DIR / f"{model}_{ds}.npz"))
    spec = pl.color_spec("own_iso")
    obs_xy = d["obs_y"].reshape(-1, 2)

    def draw_bottom(ax, m):
        return pl.draw_vector_field(ax, d["grid_q"], d["grid_p"], d[f"{m}__field_unc"], d[f"{m}__field_mean"], spec, obs_xy=obs_xy, wide=True)

    return pl.ott_figure_thesis(left_drawer(d), draw_bottom, spec, pl.function_space_handles(), key, wide=True, ph_t_factor=0.8)


if __name__ == "__main__":
    pl.set_style()
    EP.ensure()
    name = "fig04_long_phase_space_lower_half_v2"
    pl.tps.save_figure(merged_phase_space("lower_half", name), EP.IMG_MAIN / name, generator=GEN)
    for ds, dest, nums in (("lower_half", EP.IMG_MAIN, {"NODE": "fig01", "sPHNN": "fig02"}), ("left_half", EP.IMG_APP, {"NODE": "a01", "sPHNN": "a02"})):
        for model in MODELS:
            name = f"{nums[model]}_short_{model}_{ds}_wide_v2"
            pl.tps.save_figure(short_ott(model, ds, name), dest / name, generator=GEN)
