"""General schematic of an AL experiment as a graph (identical for the thermal food processing and the DED experiment; landscape layout, aspect about 2:1).
Symbols: S start trajectories (s_1 ... s_{n_start}, a start node is the data set {s_i}); n_D layers 1 ... n_{D,max}; every node (= data set) carries n_ens trained models m_{n,1} ... m_{n,n_ens} (stacked circles);
edge = acquisition step, solid = AL streams u_1 ... u_{n_strat} (UQ method, criterion; equal picks merge into one node), dashed = realised Random draws r_1 ... r_{n_branch(n_D)} (n_branch(n_D) = number of
realised Random children per node at the transition n_D -> n_D + 1), light grey = possible but not realised child; A_1, A_2, ... architectures (one graph per architecture, stacked sheets).
No explanatory text: definitions go into the LaTeX text.
Output: the export folder (figA0_experiment_graph_schematic), drawn at the inclusion width of figure_widths.json (144 mm = 0.9 linewidth). Usage: python schematic_general.py"""
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
OUT = Path(os.environ.get("CHICKEN_EXPORT_ROOT", HERE.parents[1] / "thesis_figures/final")) / "00_Masterthesis/Knowledgebase/imgs/appendix/exp03" / KEY
tps.apply_style(KEY)
FA, FH, FL = tps.raw_pt("annot"), tps.raw_pt("heading"), tps.raw_pt("legend")                  # labels 8 pt, layer headings 9 pt, legend 8 pt (final document size)

XMAX, YMIN, YMAX = 11.2, -2.98, 2.62                      # data limits: one data unit = the same length in x and y (aspect equal)
_w = tps.sizes.width_in(KEY)
fig = plt.figure(figsize=(_w, _w * (YMAX - YMIN) / XMAX)); ax = fig.add_axes([0, 0, 1, 1])
EDGE, SOFT, GHOST = "0.15", "0.5", "0.78"
X = {1: 2.0, 2: 4.7, 3: 7.0, "n": 9.2}; R = 0.24; RING = (0.075, 0.037)                       # layer x positions, node radius, offsets of the two stacked circles behind a node


def node(x, y, label=None, fc="white", ec="k", fs=FA, ring=True, lw=0.9, r=R, zorder=3):
    """Data-set node; two shifted circles behind it stand for the n_ens models trained on it."""
    if ring:
        for d in RING:
            ax.add_patch(Circle((x + d, y - d), r, fc="white", ec="0.55", lw=0.6, zorder=zorder - 1))
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=zorder))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=zorder + 1)


def curve(p, q, rad):
    """Quadratic Bezier curve from p to q (rad = bend)."""
    dx, dy = q[0] - p[0], q[1] - p[1]; c = ((p[0] + q[0]) / 2 + rad * dy, (p[1] + q[1]) / 2 - rad * dx); t = np.linspace(0, 1, 80)[:, None]
    return (1 - t) ** 2 * np.array(p) + 2 * (1 - t) * t * np.array(c) + t ** 2 * np.array(q)


def pill(x, y, label, fc="0.93", ec="0.55", w=1.4, h=0.44, fs=FA, lw=0.9, z=3):
    """rounded node with a wide label (symbols with long subscripts do not fit into a circle)"""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.18", fc=fc, ec=ec, lw=lw, zorder=z)); ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=z + 1)


def edge(p, q, rad=0.0, dashed=False, lw=1.0, col=EDGE, sa=R + 0.03, sb=R + 0.06, head=0.16, hw=0.06, z=2):
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
SX0, SX1, SY0, SY1, OFF = 0.3, 10.15, -2.22, 2.55, 0.26
ax.add_patch(FancyBboxPatch((SX0 + OFF, SY0 - OFF), SX1 - SX0, SY1 - SY0, boxstyle="round,pad=0.0,rounding_size=0.15", fc="0.96", ec="0.6", lw=0.7, zorder=0))
ax.add_patch(FancyBboxPatch((SX0, SY0), SX1 - SX0, SY1 - SY0, boxstyle="round,pad=0.0,rounding_size=0.15", fc="white", ec="0.6", lw=0.7, zorder=1))
ax.text(SX1 - 0.27, 0.2, r"$A_1$", ha="center", va="center", fontsize=FH, color="0.25", rotation=90, zorder=5); ax.text(SX1 + OFF - 0.12, -0.15, r"$A_2, \ldots$", ha="center", va="center", fontsize=FH, color="0.25", rotation=90, zorder=5)

# --- layer headers and transition labels
YH, YB = 2.24, 1.9
for k, l in ((1, "1"), (2, "2"), (3, "3"), ("n", r"$n_{D,\max}$")):
    ax.text(X[k], YH, rf"$n_D$ = {l}" if k != "n" else r"$n_D=n_{D,\max}$", ha="center", va="center", fontsize=FH, color="0.2")
ax.text((X[3] + X["n"]) / 2, YH, r"$\ldots$", ha="center", va="center", fontsize=FH + 1)
ax.text((X[1] + X[2]) / 2 + 0.1, YB, r"$n_{\mathrm{branch}}(1)$", ha="center", va="center", fontsize=FA, color="0.35"); ax.text((X[2] + X[3]) / 2 + 0.1, YB, r"$n_{\mathrm{branch}}(2)$", ha="center", va="center", fontsize=FA, color="0.35")

# --- start column: the set S of start nodes (data set {s_i}); one start expanded to show its n_ens models
Y1, Y2 = 1.45, -1.75                                              # s_1 and s_{n_start}
pill(X[1], Y1, r"$s_1$"); pill(X[1], Y2, r"$s_{n_{\mathrm{start}}}$")
BX0, BX1, BY0, BY1 = X[1] - 0.9, X[1] + 0.9, -0.72, 0.9            # expanded start s_i
ax.add_patch(FancyBboxPatch((BX0, BY0), BX1 - BX0, BY1 - BY0, boxstyle="round,pad=0.0,rounding_size=0.12", fc="0.97", ec="0.45", lw=0.8, zorder=2))
ym = (BY0 + BY1) / 2 - 0.1
pill(X[1], ym + 0.33, r"$m_{1,1}$", fc="white", ec="k", w=1.4, h=0.42); pill(X[1], ym - 0.33, r"$m_{1,n_{\mathrm{ens}}}$", fc="white", ec="k", w=1.4, h=0.42); dots(X[1], ym, fs=8)
ax.text(X[1], BY1 - 0.17, r"$s_i$", ha="center", va="center", fontsize=FA)
dots(X[1], (Y1 - 0.22 + BY1) / 2, fs=8); dots(X[1], (Y2 + 0.22 + BY0) / 2, fs=8)
ax.plot([0.75, 0.75], [Y2 - 0.1, Y1 + 0.1], color="0.35", lw=0.9); ax.plot([0.75, 0.85], [Y1 + 0.1, Y1 + 0.1], color="0.35", lw=0.9); ax.plot([0.75, 0.85], [Y2 - 0.1, Y2 - 0.1], color="0.35", lw=0.9)
ax.text(0.55, (Y1 + Y2) / 2, r"$\mathcal{S}$", ha="center", va="center", fontsize=FH, color="0.2")

# --- layer 2: AL streams u (equal picks merge into one node), realised Random draws r (all lead to different nodes)
y2 = {"a": 1.5, "b": 0.72, "c": -0.25, "d": -0.93, "e": -1.75}
for y in y2.values():
    node(X[2], y)
p0 = (BX1, ym)
edge(p0, (X[2], y2["a"]), rad=-0.28, sa=0); edge(p0, (X[2], y2["a"]), sa=0); edge(p0, (X[2], y2["b"]), sa=0)
lab_out(p0, (X[2], y2["a"]), -0.28, r"$u_2$", 0.45, 0.2); lab(p0, (X[2], y2["a"]), 0.0, r"$u_1$", t=0.62); lab_out(p0, (X[2], y2["b"]), 0.0, r"$u_{n_{\mathrm{strat}}}$", 0.55, -0.24)
for k, l, tt, oo in (("c", r"$r_1$", 0.55, 0.0), ("d", r"$r_2$", 0.55, 0.0), ("e", r"$r_{n_{\mathrm{branch}}(1)}$", 0.5, 0.0)):
    edge(p0, (X[2], y2[k]), dashed=True, lw=0.9, col=SOFT, sa=0)
    (lab(p0, (X[2], y2["e" if k == "e" else k]), 0.0, l, t=tt, col="0.3") if oo == 0 else lab_out(p0, (X[2], y2[k]), 0.0, l, tt, oo, col="0.3"))
dots(X[2], (y2["b"] + y2["c"]) / 2 + 0.06, fs=8); dots(X[2], (y2["d"] + y2["e"]) / 2 + 0.04, fs=8, col="0.5")

# --- layer 3: AL continues deterministically; every Random node could be followed by any Random node of the next layer, only R_2 edge is realised
y3 = dict(y2)
for y in y3.values():
    node(X[3], y)
for k in ("c", "d", "e"):                                  # possible, not realised (drawn first, below the realised edges)
    for k2 in ("c", "d", "e"):
        if k2 != k:
            edge((X[2], y2[k]), (X[3], y3[k2]), col=GHOST, lw=0.6, head=0.12, hw=0.04, z=1)
edge((X[2], y2["a"]), (X[3], y3["a"])); edge((X[2], y2["b"]), (X[3], y3["a"])); edge((X[2], y2["b"]), (X[3], y3["b"]))
for k in ("c", "d", "e"):
    edge((X[2], y2[k]), (X[3], y3[k]), dashed=True, lw=0.9, col=SOFT)
dots(X[3], (y3["b"] + y3["c"]) / 2 + 0.06, fs=8); dots(X[3], (y3["d"] + y3["e"]) / 2 + 0.04, fs=8, col="0.5")

# --- last layer
yn = [y2[k] for k in "abcde"]
for y in yn:
    node(X["n"], y)
for y3_, yn_ in zip([y3["a"], y3["b"], y3["c"], y3["d"], y3["e"]], yn):
    ax.plot([X[3] + 0.45, X["n"] - 0.5], [y3_, yn_], color="0.55", lw=0.8, ls=(0, (1, 2.2)), zorder=1)
dots(X["n"], (yn[1] + yn[2]) / 2 + 0.06, fs=8); dots(X["n"], (yn[3] + yn[4]) / 2 + 0.04, fs=8, col="0.5")

h = [Line2D([], [], color=EDGE, lw=1.1, label=r"AL, stream $u$ (UQ method, criterion)"), Line2D([], [], color=SOFT, lw=1.0, ls=(0, (3.2, 2.2)), label=r"Random, draw $r$"), Line2D([], [], color=GHOST, lw=0.8, label="possible, not realised")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.085), ncol=3, frameon=False, fontsize=FL, handlelength=2.2, columnspacing=1.6)
ax.set_xlim(0.0, XMAX); ax.set_ylim(YMIN, YMAX); ax.axis("off"); ax.set_aspect("equal")
tps.save_figure(fig, OUT, generator="03_thermal_food_processing/figures/experiment_graph/schematic_general.py")
w_mm, h_mm = tps.sizes.pdf_box_mm(Path(str(OUT) + ".pdf"))
print(f"box {w_mm:.1f} x {h_mm:.1f} mm, aspect {w_mm / h_mm:.2f}, inclusion width 0.9\\linewidth ({tps.sizes.included_mm(KEY):.0f} mm), scale {tps.sizes.included_mm(KEY) / w_mm:.3f}")
