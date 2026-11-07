"""Harmonic oscillator: period-maximum predictive standard deviation of the conservative sPHNN (Figure 4.6, Figure B.4) and the member-frequency table (Table B.10).

Stored results are re-plotted and re-tabulated; nothing is trained or re-evaluated. Outputs (paths from common/export_paths.py):
  fig03a_max_std_lower_half_linear, a03_max_std_left_half_linear    maximum of the predictive Std per true period, q and p, every UQ method
  tableA3_sphnn_member_frequencies.tex                               orbit frequencies of the retained sPHNN members
Usage: python -u make_envelope_and_frequencies.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ensembles"))
sys.path.insert(0, str(HERE.parent / "common"))
import export_paths as EP
import ensemble_lib as ens_lib  # (enables float64 before any model construction)
import plot_lib as pl
import make_tables as MT  # (write())

import matplotlib.pyplot as plt
import numpy as np

COMP_DIR = HERE.parent / "swag_dependence" / "comparison"
LABEL = dict(pl.style.METHOD_LABEL)
KEYS = ("LA", "MoLA", "DE", "SWAG", "MultiSWAG")
IMG_MAIN, IMG_APP, TAB_APP = EP.IMG_MAIN, EP.IMG_APP, EP.TAB_APP
YLABEL_MAX = r"$\mathrm{{max}}_T\,\sqrt{{\mathbb{{V}}[{c}]}}$"
COORD = {0: "q", 1: "p"}


def load(model, dataset):
    """Time grid and stored predictive Std of the five methods."""
    z = np.load(ens_lib.RESULT_DIR / f"methods_{model}_{dataset}.npz")
    swag = np.load(COMP_DIR / f"{model}_{dataset}.npz")
    std = {m: z[f"{m}__std"] for m in ("LA", "MoLA", "DE")}
    std["SWAG"], std["MultiSWAG"] = swag["SWAG__std"], swag["MultiSWAG__std"]
    return {"t": z["t"], "std": std}


# ---------------------------------------------------------------- maximum of the predictive Std per period
def max_std_sphnn_figure(ds, key, out, raw_std=False):
    """Conservative sPHNN, q (left) and p (right): every UQ method, with one shared legend above. Unconnected markers for the period maxima and the actual Std as thin transparent line (Figures 4.6 and B.4);
    raw_std=False: maxima joined by lines."""
    pl.tps.apply_style(key)
    d = load("sPHNN", ds)
    fig, axes = plt.subplots(1, 2, figsize=pl.tps.figsize_for(key, 2.3), gridspec_kw={"wspace": 0.4})
    for ci in (0, 1):
        ax = axes[ci]
        for m in KEYS:
            if raw_std:
                ax.plot(d["t"], d["std"][m][:, ci], color=pl.COLOR[m], lw=0.5, alpha=0.5, zorder=2)
            tk, ek = ens_lib.envelope(d["t"], d["std"][m][:, ci])      # exactly the existing statistic, plotted at the actual peak time
            ax.plot(tk, ek, color=pl.COLOR[m], marker=pl.MARKER[m], ms=3, lw=0 if raw_std else 1.3, zorder=3, label=LABEL[m])
        ax.set_xlim(0, d["t"][-1])
        ax.set_ylim(bottom=0)                                      # own y range per panel; no clipping, no transformation
        ax.set_ylabel(YLABEL_MAX.format(c=COORD[ci]))
        ax.set_xlabel("$t$")
    for ax in axes:
        ax.set_box_aspect(0.55)                                    # landscape panels: the x axis is longer than the y axis
    h, lab = axes[0].get_legend_handles_labels()
    pl.tps.legend_above(fig, h, 5, labels=lab)
    pl.tps.save_figure(fig, out, generator="02_harmonic_oscillator/figures/make_envelope_and_frequencies.py")


# ---------------------------------------------------------------- member frequencies
def table_a3():
    """Table B.10: orbit frequencies of the retained sPHNN ensemble members (= DE members = MoLA components), from the first return of each member's rollout."""
    rows, tex = [], ["\\begin{tabular}{@{}lrccc@{}}", "\\toprule",
                     "Dataset & $n_{\\mathrm{ens}}$ & $\\bar\\Omega$ & $[\\Omega_{\\min},\\Omega_{\\max}]$ & $\\sigma_\\Omega/\\bar\\Omega$ [$\\%$] \\\\", "\\midrule"]
    for ds in ens_lib.DATASETS:
        v = np.array(json.loads((ens_lib.RESULT_DIR / f"methods_sPHNN_{ds}.json").read_text())["Omega_MAP"], dtype=float)
        rec = ["\\emph{" + ds.replace("_", " ") + "}", len(v), v.mean(), v.min(), v.max(), 100 * v.std() / v.mean()]
        rows.append([rec[0], rec[1]] + [f"{x:.6g}" for x in rec[2:]])
        tex.append(f"{rec[0]} & {rec[1]} & ${rec[2]:.3f}$ & $[{rec[3]:.3f}, {rec[4]:.3f}]$ & ${rec[5]:.2f}$ \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    MT.write("tableA3_sphnn_member_frequencies", [TAB_APP], tex, ["dataset", "n", "Omega_bar", "Omega_min", "Omega_max", "sigma_over_Omega_percent"], rows)
    return rows

if __name__ == "__main__":
    pl.set_style()
    EP.ensure()
    max_std_sphnn_figure("lower_half", "fig03a_max_std_lower_half_linear", IMG_MAIN / "fig03a_max_std_lower_half_linear", raw_std=True)
    max_std_sphnn_figure("left_half", "a03_max_std_left_half_linear", IMG_APP / "a03_max_std_left_half_linear", raw_std=True)
    table_a3()
