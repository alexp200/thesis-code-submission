"""Plot elements shared by the two cubic-regression figures (verification figure and FFNN comparison): colours, line styles per parameter index, and the compact legend handle
(one short vertical line per line style, one label for the whole family)."""
import colorsys
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from thesis_plot_style import style as S

BLR_COLOR = S.ROLE_B                                           # exact BLR (reference): vermillion, distinct from all method colours and from the grey NN legend entry
BLR_LS = ["-", "--", "-.", ":"]                                # parameters theta_0..theta_3 of the cubic model
NN_LS = ["-", (0, (5, 2)), (0, (1, 1.2)), (0, (4, 1.5, 1, 1.5))]   # plotted FFNN parameters theta_0..theta_3 (deliberately different patterns)


def tints(col, n=4):
    """n lightness steps of the hue of `col` (the four parameters of one method)."""
    h, _, s_ = colorsys.rgb_to_hls(*mcolors.to_rgb(col))
    return [colorsys.hls_to_rgb(h, lv, s_) for lv in np.linspace(0.28, 0.62, n)]


class LStyles:
    """Legend handle of a parameter family: one short vertical line per line style (i = 0..3), drawn next to each other."""
    def __init__(self, color, styles):
        self.color, self.styles = color, styles


class VLinesHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        n = len(orig_handle.styles)
        xs = xdescent + width * (np.arange(n) + 0.5) / n
        h = 1.05 * fontsize                 # about the height of the label text, centred on the visual centre of the label (matplotlib places handles slightly above it)
        y0 = ydescent + 0.5 * (height - h) - 0.2 * fontsize
        return [Line2D([x, x], [y0, y0 + h], ls=ls, color=orig_handle.color, lw=S.LW["main"], transform=trans) for x, ls in zip(xs, orig_handle.styles)]


LEGEND_KW = dict(handlelength=2.2, handleheight=0.9, columnspacing=1.3, handler_map={LStyles: VLinesHandler()})
