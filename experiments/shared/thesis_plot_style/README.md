# thesis_plot_style

Shared plotting standard of the thesis figures: `style.py` (fonts, colours, line styles, markers), `figure.py` (helpers: headings, legends, export), `sizes.py` with `figure_widths.json` (inclusion width of every figure) and `size_calibration.json` (measured box sizes).

* Font sizes at the printed size: axis labels and headings 9 pt, ticks and legends 8 pt, all normal weight. Figures are exported at their inclusion width.
* Usage: `sys.path.insert(0, "<repository>/experiments/shared"); import thesis_plot_style as tps; tps.apply_style(key)`, then `tps.figsize_for(key, height)`, `tps.col_heading`, `tps.row_headings`, `tps.legend_above`, `tps.save_figure`.
* The calibration state is only updated if `TPS_CALIBRATION_WRITE=1` is set.
