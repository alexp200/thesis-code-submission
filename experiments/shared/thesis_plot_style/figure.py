"""Generic figure helpers shared by all thesis experiments (no experiment-specific logic).

typography is defined at the FINAL document size (style.TARGET_PT, normal weight everywhere); `apply_style(key)` converts it to the exported size with the
inclusion width of the figure (sizes.py); row / column headings are drawn OUTSIDE the data rectangles by `col_heading` / `row_headings`; `save_figure` measures the exported
PDF box and records raw and effective font sizes (size_calibration.json)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import sizes
from . import style
from .style import TEXTWIDTH_IN, WIDTH_FACTORS

_CUR = {"key": None, "scale": 1.0, "raw": dict(style.TARGET_PT)}      # state of the figure that is currently being drawn (set by apply_style)


def apply_style(key: str | None = None):
    """Typography (sans-serif, DejaVu mathtext, normal weight), axes / grid look, embedded TrueType fonts in PDF, 300 dpi export.
    key = file stem of the figure (figure_widths.json). Exported sizes = target sizes (the figure is drawn at its inclusion width, `figsize_for` / `sizes.width_in`, corrected so that the exported
    PDF box equals the inclusion width; effective = exported * included / box is recorded by save_figure). Call it before every figure of a script (it also sets the current key). Returns 1.0."""
    s = 1.0
    raw = {k: v / s for k, v in style.TARGET_PT.items()}
    _CUR.update(key=key, scale=s, raw=raw)
    plt.rcParams.update({
        "font.size": raw["tick"], "axes.labelsize": raw["label"], "axes.titlesize": raw["heading"], "axes.titleweight": "normal", "axes.labelweight": "normal",
        "xtick.labelsize": raw["tick"], "ytick.labelsize": raw["tick"], "legend.fontsize": raw["legend"], "figure.titlesize": raw["heading"], "figure.titleweight": "normal",
        "axes.linewidth": 0.6 / s, "xtick.major.width": 0.6 / s, "ytick.major.width": 0.6 / s, "xtick.minor.width": 0.4 / s, "ytick.minor.width": 0.4 / s,
        "xtick.major.size": 2.5 / s, "ytick.major.size": 2.5 / s, "xtick.minor.size": 1.5 / s, "ytick.minor.size": 1.5 / s, "xtick.major.pad": 2.5 / s, "ytick.major.pad": 2.5 / s,
        "axes.labelpad": 3.0 / s, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4 / s, "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
    })
    return s


def raw_pt(role: str) -> float:
    """Exported (raw) font size in points of a role of style.TARGET_PT for the figure that is being drawn."""
    return _CUR["raw"][role]


def pad_pt(effective: float) -> float:
    """Converts a distance given at the final document size (points) into the exported size of the current figure."""
    return effective / _CUR["scale"]


def figsize(kind: str = "text", height_in: float = 3.0):
    """(width, height) in inches; width = TEXTWIDTH_IN * WIDTH_FACTORS[kind]. (Figures with a fixed inclusion width use `figsize_for`.)"""
    return (TEXTWIDTH_IN * WIDTH_FACTORS[kind], height_in)


def figsize_for(key: str, height_in: float):
    """(width, height) in inches with width = the inclusion width of figure `key` (figure_widths.json), so that the exported box is close to the printed size."""
    return (sizes.width_in(key), height_in)


def save_figure(fig, base: Path, generator: str = ""):
    """png (300 dpi) + vector pdf next to each other; `base` without suffix; tight bounding box, 0.03 in padding. Closes the figure.
    If the figure is listed in figure_widths.json the exported PDF box and the effective font sizes are recorded in size_calibration.json."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    size_in = tuple(fig.get_size_inches())
    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(str(base) + ".pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    msg = ""
    if base.name in sizes.widths():
        if _CUR["key"] != base.name:
            msg = f"   WARNING: apply_style('{base.name}') was not called before drawing; sizes not recorded"
        else:
            e = sizes.record(base.name, Path(str(base) + ".pdf"), size_in, _CUR["raw"], generator)
            msg = f"   box {e['box_mm']:.1f} mm, included {e['included_mm']:.1f} mm, scale {e['scale']:.3f}" + ("   RERUN (box differs from the inclusion width by more than 1 %)" if abs(e["scale"] - 1.0) > 0.01 else "")
    print("wrote", base, msg)


def panel_tag(ax, text: str, offset_points=None):
    """Panel identifier inside the axes (headings outside the data rectangles are `col_heading` and `row_headings`). Normal weight."""
    box = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5)
    if offset_points is None:
        ax.text(0.02, 0.96, text, transform=ax.transAxes, va="top", ha="left", fontsize=8, bbox=box, zorder=10)
    else:
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=offset_points, textcoords="offset points", va="top", ha="left", fontsize=8, bbox=box, zorder=10)


# ---- headings outside the data rectangles ---------------------------------------------------------------------------------------------------------
def _as_list(target):
    return [target] if hasattr(target, "get_position") else list(target)


def col_heading(target, text: str, pad_effective: float | None = None, **kw):
    """Column heading: horizontal, centred ABOVE one axes or above a group of axes (list), normal weight, heading size, outside the data rectangle."""
    pad = pad_pt(style.HEADING_PAD_PT if pad_effective is None else pad_effective)
    if hasattr(target, "set_title"):
        return target.set_title(text, loc="center", pad=pad, fontsize=raw_pt("heading"), fontweight="normal", **kw)
    axes = _as_list(target)
    fig = axes[0].figure
    x0, x1 = min(a.get_position().x0 for a in axes), max(a.get_position().x1 for a in axes)
    y1 = max(a.get_position().y1 for a in axes)
    return fig.text(0.5 * (x0 + x1), y1 + pad / 72.0 / fig.get_figheight(), text, ha="center", va="bottom", fontsize=raw_pt("heading"), fontweight="normal", **kw)


def row_headings(rows, where: str = "left", pad_effective: float | None = None, rotation: float | None = None):
    """Row headings of the same style as `col_heading`. rows = [(axes_or_list_of_axes, text), ...].
    where='left': text in the left margin, rotated like the y axis labels (style.ROW_HEADING_ROTATION = 90; rotation=0 for horizontal text), all rows on ONE vertical line just left of the left-most
    tick labels / axis label of the whole block, centred on the row.
    where='above': the text sits left-aligned just above the row (use it when a left margin would crowd the axes)."""
    rot = style.ROW_HEADING_ROTATION if rotation is None else rotation
    pad = pad_pt(style.HEADING_PAD_PT if pad_effective is None else pad_effective)
    groups = [(_as_list(a), t) for a, t in rows]
    fig = groups[0][0][0].figure
    if where == "above":
        for g, t in groups:
            first = min(g, key=lambda a: a.get_position().x0)
            first.set_title(t, loc="left", pad=pad, fontsize=raw_pt("heading"), fontweight="normal")
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x = min(a.get_tightbbox(r).x0 for g, _ in groups for a in g) / fig.bbox.width - pad / 72.0 / fig.get_figwidth()
    out = []
    for g, t in groups:
        y = 0.5 * (min(a.get_position().y0 for a in g) + max(a.get_position().y1 for a in g))
        out.append(fig.text(x, y, t, ha="right", va="center", rotation=rot, fontsize=raw_pt("heading"), fontweight="normal"))
    return out


# ---- legends --------------------------------------------------------------------------------------------------------------------------------------


def legend_above(fig, handles, ncol, gap_pt=None, handlelength=1.8, columnspacing=1.2, **kwargs):
    """STANDARD legend position of all thesis figures: one frameless legend centred ABOVE the top row of panels, `gap_pt` (effective points, default style.LEGEND_GAP_PT)
    above the highest content: axes incl. their titles / column headings and figure-level headings. The position is computed from the artists after layout, so the gap is the same in
    every figure (no manual top margins). Call it after all panels and headings are drawn and after the axes layout is final."""
    gap = pad_pt(style.LEGEND_GAP_PT if gap_pt is None else gap_pt)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axes = [a for a in fig.axes if a.get_visible()]
    top = max([a.get_tightbbox(r).y1 for a in axes] + [t.get_window_extent(r).y1 for t in fig.texts if t.get_visible() and t.get_text()]) / fig.bbox.height
    x0 = min(a.get_position().x0 for a in axes)
    x1 = max(a.get_position().x1 for a in axes)
    return fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5 * (x0 + x1), top + gap / 72.0 / fig.get_figheight()), frameon=False,
                      handlelength=handlelength, columnspacing=columnspacing, borderaxespad=0, **kwargs)

