"""Shared plotting standard of the thesis experiments.

    import sys; sys.path.insert(0, "<repo>/experiments/shared")
    import thesis_plot_style as tps
    tps.apply_style("fig_name")             # typography at the final document size (sizes.py / figure_widths.json), normal weight
    fig, ax = plt.subplots(figsize=tps.figsize_for("fig_name", 2.4))
    ax.plot(t, y, color=tps.style.METHOD_COLOR["LA"], marker=tps.style.METHOD_MARKER["LA"])
    tps.col_heading(ax, "LA")               # headings outside the data rectangle (col_heading / row_headings)
    tps.save_figure(fig, Path("out/fig_name"))   # writes fig_name.png (300 dpi) and fig_name.pdf, records the PDF box and effective font sizes

See README.md in this folder.
"""
from . import sizes, style
from .figure import (WIDTH_FACTORS, apply_style, col_heading, figsize, figsize_for, legend_above, pad_pt, panel_tag, raw_pt, row_headings, save_figure)
from .style import TEXTWIDTH_IN
