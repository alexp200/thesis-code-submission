"""Computational graph of the aggregation behind the learning curves of the thermal food processing experiment (Figure B.6) (one fixed architecture, AL strategy and training-set size); notation only.
Two rows starting at the member-level test RMSEs:  AL: RMSE_k -> median over the n_ens members -> median and Q25-Q75 across the starts;
Random: RMSE_{r,k} of every random path r -> median over the n_ens members -> arithmetic mean over the n_rand random paths within the start (extra step, highlighted) -> median and Q25-Q75 across the starts.
No explanatory text: definitions belong into the caption.  Typography: box text 8 pt, headings 9 pt (final document size), box sizes follow from the measured text; content, symbols and
the order of all operations unchanged.  One data unit = 1 mm.  Output: the export folder (figA2_aggregation_graph).  Usage: python aggregation_graph.py"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "shared")]
import thesis_plot_style as tps

KEY = "figA2_aggregation_graph"
EXPORT = Path(os.environ.get("CHICKEN_EXPORT_ROOT", HERE.parents[1] / "thesis_figures/final")) / "00_Masterthesis/Knowledgebase/imgs/appendix/exp03" / KEY
INK = "0.15"
BLUE, YEL, ORG = ("#EEF3FA", "#6C8EBF"), ("#FFF2CC", "#D6B656"), ("#FFE6CC", "#D79B00")
PADX, PADY, GAPV = 3.0, 2.4, 11.0                                               # mm


def build():
    """Draw the two aggregation chains (AL and Random) as boxes and arrows."""
    tps.apply_style(KEY)
    W, H = tps.sizes.width_in(KEY) * 25.4, 60.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
    r = fig.canvas.get_renderer(); mm = 25.4 / fig.dpi
    fa, fh, lw = tps.raw_pt("annot"), tps.raw_pt("heading"), tps.pad_pt(0.9)

    def size(lines, fs):
        ws, hs = [], []
        for s in lines:
            t = ax.text(0, 0, s, fontsize=fs); bb = t.get_window_extent(r); t.remove(); ws.append(bb.width * mm); hs.append(bb.height * mm)
        return max(ws), max(hs)

    rows = {  # name -> list of (column, lines, colour)
        "AL": [(0, ["$\\mathrm{RMSE}_k$", "$k\\leq n_{\\mathrm{ens}}$"], BLUE), (1, [r"$E=\mathrm{med}_k(\mathrm{RMSE}_k)$"], BLUE), (3, [r"$\mathrm{med}_s,\,Q_{25},\,Q_{75}\,(E)$"], YEL)],
        "Random": [(0, ["$\\mathrm{RMSE}_{r,k}$", "$r\\leq n_{\\mathrm{rand}},\\ k\\leq n_{\\mathrm{ens}}$"], BLUE), (1, [r"$E_r=\mathrm{med}_k(\mathrm{RMSE}_{r,k})$"], BLUE), (2, [r"$\bar{E}=\mathrm{mean}_r(E_r)$"], ORG), (3, [r"$\mathrm{med}_s,\,Q_{25},\,Q_{75}\,(\bar{E})$"], YEL)]}
    heads = ["member RMSEs", "node error", "within start", "across starts\n$s=1,\\ldots,n_{\\mathrm{start}}$"]
    pitch = max(size([s], fa)[1] for rw in rows.values() for _, ls, _ in rw for s in ls) + 1.2
    boxes = {nm: [(c, ls, col, *size(ls, fa)) for c, ls, col in rw] for nm, rw in rows.items()}
    cw = [max(b[3] for nm in boxes for b in boxes[nm] if b[0] == c) + 2 * PADX for c in range(4)]
    hw = [size(h.split("\n"), fh)[0] for h in heads]
    label_w = max(size(["Random"], fh)[0], size(["AL"], fh)[0]) + 4.0
    cw = [max(a, b + 2.0) for a, b in zip(cw, hw)]
    gap = (W - label_w - sum(cw) - 1.0) / 3
    if gap < 7.0:
        print(f"WARNING: gap between the boxes {gap:.1f} mm < 7 mm")
    cx, x = [], label_w
    for w in cw:
        cx.append(x + w / 2); x += w + gap
    bh = 2 * pitch + 2 * PADY                                                     # all boxes of the height of the two-line boxes
    yA, yR = H - 20.0 - bh / 2, H - 20.0 - bh - GAPV - bh / 2
    for nm, y in (("AL", yA), ("Random", yR)):
        ax.text(0.5, y, nm, ha="left", va="center", fontsize=fh, color="0.1")
        prev = None
        for c, ls, col, w_, h_ in boxes[nm]:
            w = cw[c] if c in (0, 1, 2, 3) else w_ + 2 * PADX
            ax.add_patch(FancyBboxPatch((cx[c] - w / 2, y - bh / 2), w, bh, boxstyle="round,pad=0,rounding_size=1.6", fc=col[0], ec=col[1], lw=lw, zorder=2))
            y0 = y + pitch * (len(ls) - 1) / 2
            for i, s in enumerate(ls):
                ax.text(cx[c], y0 - i * pitch, s, ha="center", va="center", fontsize=fa, zorder=3)
            if prev is not None:
                ax.annotate("", xy=(cx[c] - w / 2, y), xytext=(prev, y), arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.2", color=INK, lw=lw, shrinkA=0, shrinkB=0), zorder=1)
            prev = cx[c] + w / 2
    for c, h in enumerate(heads):
        ax.text(cx[c], yA + bh / 2 + 3.0, h, ha="center", va="bottom", fontsize=fh, color="0.35", linespacing=1.15)
    ax.set_ylim(yR - bh / 2 - 0.8, yA + bh / 2 + 3.0 + 2 * pitch + 1.0)               # crop to the content (the axes rectangle counts for the tight bounding box)
    return fig


if __name__ == "__main__":
    fig = build()
    y0, y1 = fig.axes[0].get_ylim()
    fig.set_size_inches(fig.get_size_inches()[0], (y1 - y0) / 25.4)
    tps.save_figure(fig, EXPORT, generator="03_thermal_food_processing/figures/experiment_graph/aggregation_graph.py")
