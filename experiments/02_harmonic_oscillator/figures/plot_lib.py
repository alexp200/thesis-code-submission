"""Plotting helpers of the harmonic-oscillator figures: function-space and vector-field panels, the Ott-style and phase-plane layouts at the final document size.

Figures never carry a title; method names are written as headings outside the axes, the caption belongs to LaTeX.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))   # experiments/shared/ -> thesis_plot_style
import thesis_plot_style as tps  # (shared plotting standard: colours, line styles, markers, labels, typography, export)
from thesis_plot_style import style

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

METHODS = style.METHODS
LABEL = dict(style.METHOD_LABEL)
# method colours of the thesis (see thesis_plot_style)
COLOR = style.METHOD_COLOR
MARKER = style.METHOD_MARKER
Q_COLOR, P_COLOR = style.Q_COLOR, style.P_COLOR

FIELD_XLIM, FIELD_YLIM = (-6.0, 6.0), (-8.0, 8.0)
FIELD_XLIM_WIDE = (-10.0, 10.0)   # short-rollout thesis figures: landscape vector-field panels, fields re-evaluated on q in [-10, 10] (compute_extended_field.py)
FIELD_LABEL = r"$\|\sqrt{\mathbb{V}[f_\theta(q,p)]}\|$"
FUNCTION_YLIM = (-12.0, 12.0)  # same display window as the Ott-style verification figure

# colour scale of the vector-field uncertainty: "own" = every panel has its own linear 0..max scale and its own colorbar, "own_iso" adds contour lines at fixed fractions of
# the panel maximum (marked on the colorbar), "capN" = one linear scale 0..N for all panels (values above N saturate)
ISO_FRACTIONS = (0.05, 0.1, 0.25, 0.5)   # contour levels of "own_iso", as fractions of the panel maximum
ISO_COLOR = style.ISO_COLOR


def set_style():
    tps.apply_style()


# --- colour scale -------------------------------------------------------------


def color_spec(variant: str) -> dict:
    """Norm / colorbar extension / ticks for one colour-scale variant ("own", "own_iso" or "capN")."""
    if variant in ("own", "own_iso"):
        return {"name": variant, "per_panel": True, "isolines": variant == "own_iso", "norm": None}
    cap = float(variant[3:])
    ticks = [0.0, cap / 2.0, cap]
    return {"norm": Normalize(0.0, cap), "extend": "max", "ticks": ticks,
            "ticklabels": [f"{t:g}" if t != cap else rf"$\geq {cap:g}$" for t in ticks], "name": variant}


def colorbar(fig, spec, cax=None, ax=None, orientation="vertical", labelsize=7):
    """labelsize=None: tick label size of the style; 7 reproduces the older exports."""
    sm = ScalarMappable(norm=spec["norm"], cmap=style.CMAP_UNCERTAINTY)
    cb = fig.colorbar(sm, cax=cax, ax=ax, extend=spec["extend"], ticks=spec["ticks"], orientation=orientation)
    if spec["ticklabels"] is not None:
        (cb.ax.set_yticklabels if orientation == "vertical" else cb.ax.set_xticklabels)(spec["ticklabels"])
    cb.set_label(FIELD_LABEL)
    if labelsize is not None:
        cb.ax.tick_params(labelsize=labelsize)
    return cb


def nice_ceil(x: float) -> float:
    """Round up to two significant digits (11.39 -> 12, 0.6876 -> 0.7, 1.26e-5 -> 1.3e-5)."""
    e = np.floor(np.log10(x))
    return float(np.ceil(x / 10.0**e * 10.0 - 1e-9) / 10.0 * 10.0**e)


def panel_scale(spec, unc):
    """Norm of one vector-field panel: its own linear 0..max scale for the per-panel variants, else the shared one."""
    return Normalize(0.0, nice_ceil(float(unc.max()))) if spec.get("per_panel") else spec["norm"]


def _isolines(ax, grid_q, grid_p, unc, norm):
    levels = [f * norm.vmax for f in ISO_FRACTIONS if unc.min() < f * norm.vmax < unc.max()]
    if not levels:
        return None
    return ax.contour(grid_q, grid_p, unc, levels=levels, colors=ISO_COLOR, linewidths=0.5, alpha=0.95)


# --- panels -------------------------------------------------------------------


def draw_function_space(ax, t, truth, mean, std, ylim=FUNCTION_YLIM, train_t_end=None, obs=None, yscale=None):
    """E[q], E[p] and the +-2 std bands of one method; dashed = true trajectory."""
    if train_t_end is not None:
        ax.axvspan(0.0, train_t_end, color="#9fb6cd", alpha=0.16, lw=0, zorder=0)
    if obs is not None:
        obs_t, obs_y = obs
        ax.scatter(np.tile(obs_t, obs_y.shape[0]), obs_y[..., 0].reshape(-1), s=1.0, c="0.3", alpha=0.35, lw=0, rasterized=True, zorder=1)
        ax.scatter(np.tile(obs_t, obs_y.shape[0]), obs_y[..., 1].reshape(-1), s=1.0, c="0.3", alpha=0.35, lw=0, rasterized=True, zorder=1)
    for i, c in enumerate((Q_COLOR, P_COLOR)):
        ax.plot(t, truth[:, i], zorder=3, **style.TRUTH)
        ax.fill_between(t, mean[:, i] - 2.0 * std[:, i], mean[:, i] + 2.0 * std[:, i], color=c, alpha=0.25, lw=0, zorder=2)
        ax.plot(t, mean[:, i], color=c, lw=1.1, zorder=4)
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=5.0)
    ax.set_xlim(float(t[0]), float(t[-1]))
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yscale == "symlog":
        top = max(abs(v) for v in ax.get_ylim())
        decades = [k for k in range(1, 10) if 10.0**k <= top]
        ticks = [-(10.0**k) for k in reversed(decades)] + [0.0] + [10.0**k for k in decades]
        ax.set_yticks(ticks)
        ax.set_yticklabels([("0" if v == 0 else (r"$-10^{%d}$" if v < 0 else r"$10^{%d}$") % round(np.log10(abs(v)))) for v in ticks])
        ax.minorticks_off()


def function_space_handles(with_obs=True, with_truth=True):
    """Legend entries; E and V are typeset as \\mathbb{E}, \\mathbb{V}. Order: (E[q], band q), (E[p], band p), then
    truth / observations, so that a column-major legend keeps each mean with its band."""
    h = [Line2D([], [], color=Q_COLOR, lw=1.1, label=r"$\mathbb{E}[q]$"),
         Patch(facecolor=Q_COLOR, alpha=0.25, label=r"$2\sqrt{\mathbb{V}[q]}$"),
         Line2D([], [], color=P_COLOR, lw=1.1, label=r"$\mathbb{E}[p]$"),
         Patch(facecolor=P_COLOR, alpha=0.25, label=r"$2\sqrt{\mathbb{V}[p]}$")]
    if with_truth:
        h.append(Line2D([], [], label=r"true $q,\,p$", **style.TRUTH))
    if with_obs:
        h.append(Line2D([], [], ls="", marker="o", ms=2.0, color="0.3", alpha=0.6, label="train obs"))
    return h


def _field_view(wide=False):
    """Display axes of the (q, p) plane: (xlim, ylim, xticks, yticks, xlabel, ylabel); wide=True: q in [-10, 10]."""
    if wide:
        return FIELD_XLIM_WIDE, FIELD_YLIM, [-10, 0, 10], [-8, 0, 8], r"$q$", r"$p$"
    return FIELD_XLIM, FIELD_YLIM, [-6, 0, 6], [-8, 0, 8], r"$q$", r"$p$"


def draw_vector_field(ax, grid_q, grid_p, unc, mean_field, spec, obs_xy=None, stride=3, wide=False):
    """||std f_theta|| as colour, mean field as arrows, training points as dots. Returns the colour info ({norm, contours}) that the caller needs for a per-panel colorbar."""
    norm = panel_scale(spec, unc)
    ax.pcolormesh(grid_q, grid_p, unc, shading="auto", cmap=style.CMAP_UNCERTAINTY, norm=norm, rasterized=True)
    cs = _isolines(ax, grid_q, grid_p, unc, norm) if spec.get("isolines") else None
    if obs_xy is not None:
        ax.scatter(obs_xy[:, 0], obs_xy[:, 1], s=1.0, c="k", alpha=0.3, lw=0, rasterized=True)
    mf = np.asarray(mean_field).reshape(grid_q.shape + (2,))
    ax.quiver(grid_q[::stride, ::stride], grid_p[::stride, ::stride], mf[::stride, ::stride, 0], mf[::stride, ::stride, 1], color="0.15", alpha=0.8, width=0.004, pivot="mid")
    xlim, ylim, xt, yt, _, _ = _field_view(wide)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.grid(False)
    return {"norm": norm, "contours": cs}


MEAN_COLOR = style.MEAN_COLOR


def draw_phase_space(ax, grid_q, grid_p, unc, spec, mean_traj, true_traj, x0, truth_kw=None, wide=False):
    """Phase-plane panel: the field uncertainty as background, the true orbit (line style of the true trajectory in the function-space panels unless truth_kw is given),
    the predictive mean in red and the initial state."""
    norm = panel_scale(spec, unc)
    ax.pcolormesh(grid_q, grid_p, unc, shading="auto", cmap=style.CMAP_UNCERTAINTY, norm=norm, rasterized=True)
    cs = _isolines(ax, grid_q, grid_p, unc, norm) if spec.get("isolines") else None
    ax.plot(true_traj[:, 0], true_traj[:, 1], zorder=4, **(truth_kw or style.TRUTH))
    ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=MEAN_COLOR, lw=0.6, alpha=0.95, zorder=3)
    ax.plot([x0[0]], [x0[1]], "o", mfc="white", mec="k", mew=0.6, ms=3, zorder=5)
    xlim, ylim, xt, yt, _, _ = _field_view(wide)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.grid(False)
    return {"norm": norm, "contours": cs}


# --- thesis layouts at the final document size: headings outside the axes, normal weight, sizes from thesis_plot_style.TARGET_PT -----------------
def _mm(v):
    return v / 25.4


def panel_colorbar_thesis(fig, info, cax, nticks=3):
    """Horizontal colour bar of ONE panel (own range, contour levels marked on it); tick label size from the style (no explicit sizes)."""
    sm = ScalarMappable(norm=info["norm"], cmap=style.CMAP_UNCERTAINTY)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    if info.get("contours") is not None:
        cb.add_lines(info["contours"])
    cb.locator = matplotlib.ticker.MaxNLocator(nticks)
    cb.formatter = matplotlib.ticker.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.2g}".replace("e-0", "e-").replace("e+0", "e+"))
    cb.update_ticks()
    return cb


def _colorbar_label(fig, cbs, x_centre, gap_pt=3.0):
    r = fig.canvas.get_renderer()
    y0 = min(cb.ax.get_tightbbox(r).y0 for cb in cbs) / fig.bbox.height
    return fig.text(x_centre, y0 - tps.pad_pt(gap_pt) / 72 / fig.get_figheight(), FIELD_LABEL + "  (own colour scale per panel)", ha="center", va="top", fontsize=tps.raw_pt("label"))


def ott_figure_thesis(draw_top, draw_bottom, spec, handles, key, top_ylabel=r"$q,\ p$", ph_t_factor=0.95, wide=False):
    """Layout of Ott et al. (Fig. 2) at the final document size: one COLUMN per method (column heading above the function-space panel), top row = function space ($t$ axis),
    bottom row = equal-aspect vector-field panel with its own horizontal colour bar, one common colour-bar label, one frameless legend above the headings."""
    tps.apply_style(key)
    n = len(METHODS)
    W = tps.sizes.width_in(key)
    ml, mr, g = _mm(15.5), _mm(1.5), _mm(5.5)
    pw = (W - ml - mr - (n - 1) * g) / n
    xl_ = FIELD_XLIM_WIDE if wide else FIELD_XLIM
    fx, fy = xl_[1] - xl_[0], FIELD_YLIM[1] - FIELD_YLIM[0]
    ph_v = pw * fy / fx
    ph_t = ph_t_factor * pw
    gap_tv, cb_off, cb_h, top, bot = _mm(11.5), _mm(12.5), _mm(2.6), _mm(17.0), _mm(24.0)
    H = top + ph_t + gap_tv + ph_v + bot
    fig = plt.figure(figsize=(W, H))
    tops, cbs = [], []
    y_t = (bot + ph_v + gap_tv) / H
    for i, m in enumerate(METHODS):
        x0 = (ml + i * (pw + g)) / W
        at = fig.add_axes([x0, y_t, pw / W, ph_t / H], sharey=tops[0] if tops else None)
        draw_top(at, m)
        at.set_xlabel(r"$t$")
        at.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
        if i:
            plt.setp(at.get_yticklabels(), visible=False)
        else:
            at.set_ylabel(top_ylabel)
        tops.append(at)
        ab = fig.add_axes([x0, bot / H, pw / W, ph_v / H])
        info = draw_bottom(ab, m)
        _, _, _, _, xl, yl = _field_view(wide)
        ab.set_xlabel(xl)
        if i:
            ab.set_yticklabels([])
        else:
            ab.set_ylabel(yl)
        cbs.append(panel_colorbar_thesis(fig, info, ab.inset_axes([0.05, -cb_off / ph_v, 0.9, cb_h / ph_v])))
        tps.col_heading(at, LABEL[m])
    _colorbar_label(fig, cbs, (ml + 0.5 * (n * pw + (n - 1) * g)) / W)
    tps.legend_above(fig, handles, len(handles))
    return fig


def phase_grid_figure_thesis(draw, spec, row_labels, handles, key, wide=False):
    """len(row_labels) x 5 grid of equal-aspect phase-plane panels (columns = methods, rows = architectures) at the final document size: column headings above the first row, row headings
    (horizontal, left margin, one vertical line), own horizontal colour bar under every panel, one common colour-bar label and one legend above the headings. No time axis."""
    tps.apply_style(key)
    nrow, n = len(row_labels), len(METHODS)
    W = tps.sizes.width_in(key)
    ml, mr, g = _mm(24.0), _mm(1.5), _mm(5.5)
    pw = (W - ml - mr - (n - 1) * g) / n
    xl_ = FIELD_XLIM_WIDE if wide else FIELD_XLIM
    fx, fy = xl_[1] - xl_[0], FIELD_YLIM[1] - FIELD_YLIM[0]
    ph = pw * fy / fx
    cb_off, cb_h, block, top, bot = _mm(12.5), _mm(2.6), _mm(21.0), _mm(17.0), _mm(8.0)     # block: tick labels, axis label, colour bar and its tick labels below one axes
    H = top + nrow * (ph + block) + bot
    fig = plt.figure(figsize=(W, H))
    axes, cbs = [], []
    for r in range(nrow):
        y0 = (bot + (nrow - 1 - r) * (ph + block) + block) / H
        row = []
        for c, m in enumerate(METHODS):
            ax = fig.add_axes([(ml + c * (pw + g)) / W, y0, pw / W, ph / H])
            info = draw(ax, r, m)
            _, _, _, _, xl, yl = _field_view(wide)
            ax.set_xlabel(xl)
            if c:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(yl)
            cbs.append(panel_colorbar_thesis(fig, info, ax.inset_axes([0.05, -cb_off / ph, 0.9, cb_h / ph])))
            if r == 0:
                tps.col_heading(ax, LABEL[m])
            row.append(ax)
        axes.append(row)
    tps.row_headings([(axes[r], row_labels[r]) for r in range(nrow)])
    _colorbar_label(fig, cbs, (ml + 0.5 * (n * pw + (n - 1) * g)) / W)
    tps.legend_above(fig, handles, len(handles), columnspacing=1.6)
    return fig
