"""General schematic of an AL experiment as a graph (Figure B.5; identical for the thermal food processing and the DED experiment, the table gives the values of the symbols).
Symbols in the figure: S start trajectories (s_1 ... s_|S|, a start node is the data set {s_i}); n_D layers 1 ... n_max; every node (= data set) carries K trained models m_{n,1} ... m_{n,K} (K training seeds, all strategies; stacked circles);
edge = acquisition step, solid = AL streams u_1 ... u_U (UQ method, criterion; equal picks merge into one node), dashed = realised Random draws r_1 ... r_{R_n} (R_n = number of realised Random children per node at the transition n -> n+1),
light grey = possible but not realised child (arrows to the existing nodes of the next layer); A_1, A_2, ... architectures (one graph per architecture, stacked sheets). No explanatory text: definitions go into the LaTeX text.
Output: the export folder (figA0_experiment_graph_schematic). Usage: python schematic_general.py"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "shared")]
import thesis_plot_style as tps

KEY = "figA0_experiment_graph_schematic"
EXPORT = Path(os.environ.get("CHICKEN_EXPORT_ROOT", HERE.parents[1] / "thesis_figures/final")) / "00_Masterthesis/Knowledgebase/imgs/appendix/exp03" / KEY
tps.apply_style(KEY)
FA, FH, FL = tps.raw_pt("annot"), tps.raw_pt("heading"), tps.raw_pt("legend")                  # labels 8 pt, layer headings 9 pt, legend 8 pt (final document size)
_w = tps.sizes.width_in(KEY)
fig = plt.figure(figsize=(_w, _w * 9.6 / 10.6)); ax = fig.add_axes([0, 0, 1, 1])                                           # one data unit = the same length in x and y (aspect equal)
EDGE, SOFT, GHOST = "0.15", "0.5", "0.78"
X = {1: 1.9, 2: 4.2, 3: 5.95, "n": 8.1}; R = 0.27


def node(x, y, label=None, fc="white", ec="k", fs=FA, ring=True, lw=0.9, r=R, zorder=3):
    """Data-set node; two shifted circles behind it stand for the K models trained on it."""
    if ring:                                                   # K models trained on every data set: two shifted circles behind
        for d in (0.11, 0.055):
            ax.add_patch(Circle((x + d, y - d), r, fc="white", ec="0.55", lw=0.6, zorder=zorder - 1))
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=zorder))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=zorder + 1)


def curve(p, q, rad):
    """Quadratic Bezier curve from p to q (rad = bend)."""
    dx, dy = q[0] - p[0], q[1] - p[1]; c = ((p[0] + q[0]) / 2 + rad * dy, (p[1] + q[1]) / 2 - rad * dx); t = np.linspace(0, 1, 80)[:, None]
    return (1 - t) ** 2 * np.array(p) + 2 * (1 - t) * t * np.array(c) + t ** 2 * np.array(q)


def pill(x, y, label, fc="0.93", ec="0.55", w=1.5, h=0.62, fs=FA, lw=0.9, z=3):
    """rounded node with a wide label (symbols with long subscripts do not fit into a circle)"""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.24", fc=fc, ec=ec, lw=lw, zorder=z)); ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=z + 1)


def edge(p, q, rad=0.0, dashed=False, lw=1.0, col=EDGE, sa=R + 0.03, sb=R + 0.06, head=0.17, hw=0.065, z=2):
    """edge as an explicit polyline (optionally dashed) plus a solid triangular head, so that dashed styles do not break the arrow head"""
    pts = curve(p, q, rad); seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    keep = (seg >= sa) & (seg <= seg[-1] - sb); pts = pts[keep]; tip = pts[-1]; u = pts[-1] - pts[-4]; u = u / np.hypot(*u); n = np.array([-u[1], u[0]])
    shaft = pts[np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))] <= np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))][-1] - head * 0.8]
    ax.plot(shaft[:, 0], shaft[:, 1], color=col, lw=lw, ls=(0, (3.2, 2.2)) if dashed else "-", solid_capstyle="butt", dash_capstyle="butt", zorder=z)
    ax.add_patch(Polygon([tip, tip - u * head + n * hw, tip - u * head - n * hw], closed=True, fc=col, ec="none", zorder=z))


def lab(p, q, rad, text, t=0.62, col="0.15", fs=FA):
    """Label on an edge (white background)."""
    x, y = curve(p, q, rad)[int(t * 79)]; ax.text(x, y, text, fontsize=fs, ha="center", va="center", color=col, zorder=6, bbox=dict(fc="white", ec="none", pad=0.6))


def lab_out(p, q, rad, text, t, off, col="0.15", fs=FA):
    """label next to an edge: curve point at t pushed along the normal (off > 0: left of the direction of travel)"""
    pts = curve(p, q, rad); i = int(t * 79); x, y = pts[i]; d = pts[min(i + 2, 79)] - pts[max(i - 2, 0)]; d = d / np.hypot(*d)
    ax.text(x - off * d[1], y + off * d[0], text, fontsize=fs, ha="center", va="center", color=col, zorder=6)


def dots(x, y, fs=9, col="0.35"):
    """Vertical ellipsis."""
    ax.text(x, y, r"$\vdots$", ha="center", va="center", fontsize=fs, color=col)


# --- one sheet per architecture (one graph per architecture)
ax.add_patch(FancyBboxPatch((0.35 + 0.45, -4.25 - 0.45), 9.4, 8.7, boxstyle="round,pad=0.0,rounding_size=0.15", fc="0.96", ec="0.6", lw=0.7, zorder=0))
ax.add_patch(FancyBboxPatch((0.35, -4.25), 9.4, 8.7, boxstyle="round,pad=0.0,rounding_size=0.15", fc="white", ec="0.6", lw=0.7, zorder=1))
ax.text(9.45, 0.0, r"$A_1$", ha="center", va="center", fontsize=FH, color="0.25", rotation=90, zorder=5); ax.text(9.95, -0.45, r"$A_2, \ldots$", ha="center", va="center", fontsize=FH, color="0.25", rotation=90, zorder=5)

# --- layer headers and transition labels
for k, l in ((1, "1"), (2, "2"), (3, "3"), ("n", r"$n_{D,\max}$")):
    ax.text(X[k], 3.65, rf"$n_D$ = {l}" if k != "n" else r"$n_D=n_{D,\max}$", ha="center", fontsize=FH, color="0.2")
ax.text((X[3] + X["n"]) / 2, 3.65, r"$\ldots$", ha="center", fontsize=FH + 1)
ax.text((X[1] + X[2]) / 2 + 0.1, 3.1, r"$n_{\mathrm{branch}}(1)$", ha="center", fontsize=FA, color="0.35"); ax.text((X[2] + X[3]) / 2 + 0.1, 3.1, r"$n_{\mathrm{branch}}(2)$", ha="center", fontsize=FA, color="0.35")

# --- start column: the set S of start nodes (data set {s_i}); one start expanded to show its K models
for y, l in ((2.9, r"$s_1$"), (-2.9, r"$s_{n_{\mathrm{start}}}$")):
    pill(X[1], y, l)
dots(X[1], 2.1); dots(X[1], -2.1)
ax.add_patch(FancyBboxPatch((X[1] - 0.92, -1.08), 1.84, 2.16, boxstyle="round,pad=0.0,rounding_size=0.12", fc="0.97", ec="0.45", lw=0.8, zorder=2))
pill(X[1], 0.64, r"$m_{1,1}$", fc="white", ec="k", w=1.5, h=0.66); pill(X[1], -0.64, r"$m_{1,n_{\mathrm{ens}}}$", fc="white", ec="k", w=1.5, h=0.66); dots(X[1], 0.0, fs=8)
ax.text(X[1], 1.4, r"$s_i$", ha="center", va="center", fontsize=FA)
ax.plot([0.72, 0.72], [-3.3, 3.3], color="0.35", lw=0.9); ax.plot([0.72, 0.82], [3.3, 3.3], color="0.35", lw=0.9); ax.plot([0.72, 0.82], [-3.3, -3.3], color="0.35", lw=0.9)
ax.text(0.55, 0, r"$\mathcal{S}$", ha="center", va="center", fontsize=FH, color="0.2")

# --- layer 2: AL streams u (equal picks merge into one node), realised Random draws r (all lead to different nodes)
y2 = {"a": 2.4, "b": 1.2, "c": -1.0, "d": -1.9, "e": -2.9}
for y in y2.values():
    node(X[2], y)
p0 = (X[1] + 0.92, 0.0)
edge(p0, (X[2], y2["a"]), rad=0.26, sa=0); edge(p0, (X[2], y2["a"]), rad=-0.1, sa=0); edge(p0, (X[2], y2["b"]), sa=0)
lab_out(p0, (X[2], y2["a"]), 0.26, r"$u_2$", 0.45, 0.25); lab_out(p0, (X[2], y2["a"]), -0.1, r"$u_1$", 0.62, -0.25); lab_out(p0, (X[2], y2["b"]), 0.0, r"$u_{n_{\mathrm{strat}}}$", 0.72, -0.5)
for k, l, tt, oo in (("c", r"$r_1$", 0.55, 0.0), ("d", r"$r_2$", 0.55, 0.0), ("e", r"$r_{n_{\mathrm{branch}}(1)}$", 0.62, -0.5)):
    edge(p0, (X[2], y2[k]), dashed=True, lw=0.9, col=SOFT, sa=0)
    (lab(p0, (X[2], y2[k]), 0.0, l, t=tt, col="0.3") if oo == 0 else lab_out(p0, (X[2], y2[k]), 0.0, l, tt, oo, col="0.3"))
dots(X[2], 0.1, fs=8); dots(X[2], -2.4, fs=8, col="0.5"); dots(X[2], 3.25); dots(X[2], -3.5)

# --- layer 3: AL continues deterministically; every Random node could be followed by any Random node of the next layer, only R_2 edge is realised
y3 = {"a": 2.4, "b": 1.4, "c": -1.0, "d": -1.9, "e": -2.9}
for y in y3.values():
    node(X[3], y)
for k in ("c", "d", "e"):                                  # possible, not realised (drawn first, below the realised edges)
    for k2 in ("c", "d", "e"):
        if k2 != k:
            edge((X[2], y2[k]), (X[3], y3[k2]), col=GHOST, lw=0.6, head=0.13, hw=0.045, z=1)
edge((X[2], y2["a"]), (X[3], y3["a"])); edge((X[2], y2["b"]), (X[3], y3["a"])); edge((X[2], y2["b"]), (X[3], y3["b"]))
for k in ("c", "d", "e"):
    edge((X[2], y2[k]), (X[3], y3[k]), dashed=True, lw=0.9, col=SOFT)
dots(X[3], 3.25); dots(X[3], -3.5)

# --- last layer
yn = [2.0, 1.0, -1.0, -1.9, -2.9]
for y in yn:
    node(X["n"], y)
for y3_, yn_ in zip([2.4, 1.4, -1.0, -1.9, -2.9], yn):
    ax.plot([X[3] + 0.5, X["n"] - 0.55], [y3_, yn_], color="0.55", lw=0.8, ls=(0, (1, 2.2)), zorder=1)
dots(X["n"], 3.25); dots(X["n"], -3.5)

h = [Line2D([], [], color=EDGE, lw=1.1, label=r"AL, stream $u$ (UQ method, criterion)"), Line2D([], [], color=SOFT, lw=1.0, ls=(0, (3.2, 2.2)), label=r"Random, draw $r$"), Line2D([], [], color=GHOST, lw=0.8, label="possible, not realised")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False, fontsize=FL, handlelength=2.2, columnspacing=1.2)
ax.set_xlim(0.0, 10.6); ax.set_ylim(-4.9, 4.7); ax.axis("off"); ax.set_aspect("equal")
tps.save_figure(fig, EXPORT, generator="03_thermal_food_processing/figures/experiment_graph/schematic_general.py")
