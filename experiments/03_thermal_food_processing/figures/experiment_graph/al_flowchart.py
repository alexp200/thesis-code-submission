"""Flowchart of the trajectory-pool AL loop (methodology chapter, Figure 3.1; shared by the thermal food processing and the DED experiment).
Accessible overview, not a computational graph: one pale-blue process style (training, evaluation, scoring, selection, update), one neutral style (start, stop, decision), dark arrows,
text 9 pt at the final document size, normal weight.  The optional validation-error stopping condition is set in subdued text (the reported experiments use the fixed budget n_D = n_{D,max}).
Definitions (E_val, E_tol, removal of the acquired candidate from the pool, diagonal of Sigma_e only) belong into the caption.
One data unit = 1 mm of the printed figure (axes fill the figure), box sizes follow from the measured text sizes.
Output: the export folder (imgs/main/method/al_flowchart).  Usage: python al_flowchart.py"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "shared")]
import thesis_plot_style as tps

KEY = "al_flowchart"
EXPORT = Path(os.environ.get("CHICKEN_EXPORT_ROOT", HERE.parents[1] / "thesis_figures/final")) / "00_Masterthesis/Knowledgebase/imgs/main/method" / KEY
STYLE = {"process": ("#DAE8FC", "#6C8EBF"), "neutral": ("#F2F2F2", "#7A7A7A")}
INK, SUB = "0.12", "0.55"                                           # arrows / text; subdued text (optional condition)
PADX, PADY, PITCH_GAP, GAP_V = 2.6, 2.2, 1.0, 8.0                      # mm


def build():
    """Measure the text, place the nodes on a grid and draw boxes and arrows."""
    tps.apply_style(KEY)
    W, H = tps.sizes.width_in(KEY) * 25.4, 92.0                           # mm; H is trimmed by the tight bounding box
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    r = fig.canvas.get_renderer()
    mm = 25.4 / fig.dpi
    fs = tps.raw_pt("flow")
    lw = tps.pad_pt(0.9)

    def measure(s):
        """Width and height of a text in mm."""
        t = ax.text(0, 0, s, fontsize=fs)
        bb = t.get_window_extent(r); t.remove()
        return bb.width * mm, bb.height * mm

    class Node:
        def __init__(self, lines, kind, shape="box", left=()):
            self.lines = [(l, INK) if isinstance(l, str) else l for l in lines]
            self.kind, self.shape, self.left = kind, shape, set(left)           # left: indices of lines that are left-aligned (numbered items)
            m = [measure(l) for l, _ in self.lines]
            self.tw, self.pitch = max(w for w, _ in m), max(h for _, h in m) + PITCH_GAP
            self.th = self.pitch * (len(m) - 1) + max(h for _, h in m)
            if shape == "diamond":
                rho = 1.75                                           # half width / half height
                self.b = (self.tw / rho + self.th) / 2 * 1.18        # all text corners inside the diamond (+ margin)
                self.a = rho * self.b
                self.w, self.h = 2 * self.a, 2 * self.b
            else:
                self.w, self.h = self.tw + 2 * PADX, self.th + 2 * PADY
            self.x = self.y = 0.0

        def draw(self):
            fc, ec = STYLE[self.kind]
            if self.shape == "diamond":
                ax.add_patch(Polygon([(self.x - self.a, self.y), (self.x, self.y + self.b), (self.x + self.a, self.y), (self.x, self.y - self.b)], closed=True, fc=fc, ec=ec, lw=lw, zorder=2))
            else:
                ax.add_patch(FancyBboxPatch((self.x - self.w / 2, self.y - self.h / 2), self.w, self.h, boxstyle="round,pad=0,rounding_size=2.2", fc=fc, ec=ec, lw=lw, zorder=2))
            y0 = self.y + self.pitch * (len(self.lines) - 1) / 2
            for i, (s, c) in enumerate(self.lines):
                ax.text(self.x - self.tw / 2 if i in self.left else self.x, y0 - i * self.pitch, s, ha="left" if i in self.left else "center", va="center", fontsize=fs, color=c, zorder=3)

        edge = lambda self, side: {"l": (self.x - self.w / 2, self.y), "r": (self.x + self.w / 2, self.y), "t": (self.x, self.y + self.h / 2), "b": (self.x, self.y - self.h / 2)}[side]

    def arrow(p, q):
        """Arrow from p to q."""
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.22", color=INK, lw=lw, shrinkA=0, shrinkB=0), zorder=1)

    start = Node(["Start", "(initial dataset)", r"$n_D=1$"], "neutral")
    train = Node(["Train $n_{\\mathrm{ens}}$ models", r"on $\mathcal{D}_{n_D}$"], "process")
    evalu = Node(["Evaluate models"], "process")
    dec = Node([r"$n_D\geq n_{D,\max}$", (r"or $E_{\mathrm{val}}\leq E_{\mathrm{tol}}$", SUB), ("(optional)", SUB)], "neutral", shape="diamond")
    stop = Node(["Stop"], "neutral")
    score = Node(["Score pool", r"1. UQ method $m$: $\mathbf{\Sigma}_{\mathrm{e},m}(\mathbf{z})$", r"2. Acquisition: $\alpha(\mathbf{z})$ for $\mathbf{z}\in\mathcal{P}_{n_D}$"], "process", left=(1, 2))
    select = Node(["Select candidate", r"$\mathbf{z}^{*}=\arg\max_{\mathbf{z}\in\mathcal{P}_{n_D}}\,\alpha(\mathbf{z})$"], "process")
    update = Node(["Acquire and add", "trajectory", r"$n_D\rightarrow n_D+1$"], "process")

    cols = [[start, train, update], [evalu, select], [dec, score], [stop]]
    for col in (cols[0], cols[1]):                                    # boxes of one column share the width, boxes of one row the height
        w = max(n.w for n in col)
        for n in col:
            n.w = w
    for row in ([train, evalu, stop], [update, select, score]):
        h = max(n.h for n in row)
        for n in row:
            n.h = h
    cw = [max(n.w for n in c) for c in cols]
    gap = (W - sum(cw) - 1.0) / 3
    if gap < 8.0:
        print(f"WARNING: horizontal gap {gap:.1f} mm < 8 mm; shorten labels or widen the figure")
    cx, x = [], 0.5
    for w in cw:
        cx.append(x + w / 2); x += w + gap
    for c, nodes in zip(cx, cols):
        for n in nodes:
            n.x = c
    rows = [[start], [train, evalu, dec, stop], [update, select, score]]
    rh = [max(n.h for n in row) for row in rows]
    y = H - 0.5
    for row, h in zip(rows, rh):
        yc = y - h / 2
        for n in row:
            n.y = yc
        y -= h + GAP_V
    for n in (start, train, evalu, dec, stop, score, select, update):
        n.draw()

    arrow(start.edge("b"), train.edge("t"))
    arrow(train.edge("r"), evalu.edge("l"))
    arrow(evalu.edge("r"), (dec.x - dec.a, dec.y))
    arrow((dec.x + dec.a, dec.y), stop.edge("l"))
    ax.text((dec.x + dec.a + stop.edge("l")[0]) / 2, dec.y + 1.0, "yes", ha="center", va="bottom", fontsize=fs, color=INK)
    arrow((dec.x, dec.y - dec.b), score.edge("t"))
    ax.text(dec.x + 1.6, (dec.y - dec.b + score.edge("t")[1]) / 2, "no", ha="left", va="center", fontsize=fs, color=INK)
    arrow(score.edge("l"), select.edge("r"))
    arrow(select.edge("l"), update.edge("r"))
    arrow(update.edge("t"), train.edge("b"))
    ax.set_ylim(y + GAP_V - 0.5, H)                                   # crop the unused bottom part (data units stay millimetres)
    return fig


if __name__ == "__main__":
    fig = build()
    y0, y1 = fig.axes[0].get_ylim()
    fig.set_size_inches(fig.get_size_inches()[0], (y1 - y0) / 25.4)
    tps.save_figure(fig, EXPORT, generator="03_thermal_food_processing/figures/experiment_graph/al_flowchart.py")
