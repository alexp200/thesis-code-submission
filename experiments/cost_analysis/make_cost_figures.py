"""Wall time per AL step of the thermal food processing ("Chicken" in the data) and the DED experiment, one figure per data set (left sPHNN, right NODE): Figures B.8 and B.10.
x = ensemble size N whose acquisition is measured (1..4); grey = training of the K = 3 ensemble (identical for all UQ methods), colour = acquisition with the UQ method (UQ fit + scoring + selection);
median over the runs with whiskers = quartiles, small dot at the median; log axis.  Data: results/cost_data.csv (cost_data.py).
Output: figA3_cost_wall_time / figA2_cost_wall_time (.png, .pdf) and results/cost_summary.csv.  Usage: python make_cost_figures.py"""
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ["COST_OUT_DIR"]) if os.environ.get("COST_OUT_DIR") else HERE / "results"          # submission: outputs below COST_OUT_DIR when set
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(HERE.parent / "shared")]
import thesis_plot_style as tps
from thesis_plot_style import style as TS

d = pd.read_csv(HERE / "results/cost_data.csv"); d = d[d.N <= 4]; METHODS = ("MoLA", "DE", "LA"); ARCHS = ("sPHNN", "NODE")
SER = [("train", "0.55", 0.0)] + [(m, TS.METHOD_COLOR[m], 0.0) for m in ("DE", "LA", "MoLA")]
d["acq_total_s"] = d.acq_s + d.extra_train_s.fillna(0)
KEY = {"Chicken": "figA3_cost_wall_time", "DED": "figA2_cost_wall_time"}        # file stems of the exports (figure_widths.json)
for ds in ("Chicken", "DED"):
    tps.apply_style(KEY[ds])
    fig, axs = plt.subplots(1, 2, figsize=tps.figsize_for(KEY[ds], 2.55), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, arch in zip(axs, ARCHS):
        a = d[(d.dataset == ds) & (d.arch == arch)]
        for key, col, dx in SER:
            col_name = "train_s" if key == "train" else "acq_s"; sub = a if key == "train" else a[a.method == key]
            if key == "train":
                sub = a[a.method == "DE"]                                              # training time is independent of the UQ method; one record per ensemble
            g = sub.groupby("N")[col_name]; med = g.median() / 60; q1 = g.quantile(.25) / 60; q3 = g.quantile(.75) / 60; x = med.index.to_numpy() + dx
            ax.plot(med.index, med.values, color=col, lw=1.0); ax.vlines(x, q1.values, q3.values, color=col, lw=0.5); ax.plot(x, q1.values, ls="", marker="_", ms=2.5, mew=0.4, color=col); ax.plot(x, q3.values, ls="", marker="_", ms=2.5, mew=0.4, color=col)
            ax.plot(x, med.values, ls="", marker="o", ms=1.8, mew=0, color=col, zorder=3)
        if ds == "DED":                                                                                # LA incl. the separately trained K = 1 acquisition model (dotted)
            sub = a[a.method == "LA"]; g = sub.groupby("N").acq_total_s; med = g.median() / 60; q1 = g.quantile(.25) / 60; q3 = g.quantile(.75) / 60; x = med.index.to_numpy() + 0.0
            ax.plot(med.index, med.values, color=TS.METHOD_COLOR["LA"], lw=1.0, ls=":"); ax.vlines(x, q1.values, q3.values, color=TS.METHOD_COLOR["LA"], lw=0.5, ls=":"); ax.plot(x, med.values, ls="", marker="o", ms=1.8, mew=0, color=TS.METHOD_COLOR["LA"], zorder=3)
        ax.set_yscale("log"); ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(0.7, 4.3); ax.set_xlabel(r"$n_D$"); tps.col_heading(ax, arch); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0].set_ylabel("Wall time per step [min]")
    tps.legend_above(fig, [Line2D([], [], color="0.55", lw=1.0, label="training ($n_{\\mathrm{ens}}=3$)")] + [Line2D([], [], color=TS.METHOD_COLOR[m], lw=1.0, label=f"{m} acquisition") for m in METHODS] + ([Line2D([], [], color=TS.METHOD_COLOR["LA"], lw=1.0, ls=":", label="LA incl. $n_{\\mathrm{ens}}{=}1$ model training")] if ds == "DED" else []), ncol=3 if ds == "DED" else 4)
    tps.save_figure(fig, OUT / KEY[ds], generator="cost_analysis/make_cost_figures.py")
d["ratio"] = d.acq_s / d.train_s
S = d.groupby(["dataset", "arch", "method"]).agg(n=("ratio", "size"), hosts=("host", "nunique"), cpus=("cpus", "median"), train_min=("train_s", lambda x: x.median() / 60), acq_min=("acq_s", lambda x: x.median() / 60), acq_total_min=("acq_total_s", lambda x: x.median() / 60), acq_over_train=("ratio", "median"),
                                                 acq_q25=("acq_s", lambda x: x.quantile(.25) / 60), acq_q75=("acq_s", lambda x: x.quantile(.75) / 60)).reset_index(); S.to_csv(OUT / "cost_summary.csv", index=False); print(S.round(3).to_string())
