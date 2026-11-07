"""Dataset introduction figure of the Chicken experiments (exp03): the data actually used - three selection-pool trajectories (sinAPRBS with the median mean oven temperature of the 13 pool sinAPRBS, APRBS 745, multi-sine 795) and one AP15 test trajectory (median mean oven temperature; AP15 = 15 APRBS).
One row of panels: the input T_oven(t) (dotted, in the background) and the probe temperatures T_A (core) and T_B (surface) in the same axes. Examples: 745 (APRBS, 4 steps; Kannapinn Fig. 5.2, training signal of Roth et al. 2025),
1056 (sinAPRBS obtained from 745, Kannapinn Fig. 5.7; folder name ...id745), 795 (multi-sine with 6 components; Roth et al. 2025). Data: data/chicken_data_full_clean (805 simulations, contains the 30 of data/chicken_data).
Classes by folder name (details: data/chicken_data_full_clean/EXCITATION_CLASSES.md): a1 = single step, a2-a4 = APRBS with 2-4 steps, sr = sinAPRBS, sra = step signals paired with the sr signals,
m1 = single sine, m2.. = multi-sine.
Writes imgs/main/exp03/fig04_dataset_excitation_types.(png,pdf) (Figure 4.7) and final_audit/dataset_class_shares.csv.
Usage: python figures/make_dataset_figure.py"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, str(HERE.parents[1] / "shared"))
import export_paths as EP
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.lines import Line2D

import thesis_plot_style as tps

DATA = EP.REPO / "data/chicken_data_full_clean"
POOL = [745, 795, 835, 851, 853, 861, 868, 873, 874, 888, 899, 902, 905, 906, 917]
TEST = [313, 320, 344, 378, 383, 407, 412, 415, 461, 462, 466, 467, 474, 508, 528]
CLASSES = ["APRBS (2-4 steps)", "sinAPRBS", "multi-sine", "single step", "single sine", "APRBS base of the sinAPRBS (prefix sra)"]
COLOR_A, COLOR_B, COLOR_IN = "#0072B2", "#D55E00", "0.15"
NAME = "fig04_dataset_excitation_types"


def classify(folder):
    """Excitation class of a simulation from its folder name."""
    name = re.sub(r"^\d+-", "", folder)
    if name.startswith("sra"):
        return CLASSES[5]
    if name.startswith("sr"):
        return "sinAPRBS"
    if name.startswith("a"):
        return "single step" if name.startswith("a1") else "APRBS (2-4 steps)"
    return "single sine" if name.startswith("m1t") else "multi-sine"


def main():
    """Class shares of the catalogue and the dataset figure."""
    tps.apply_style(); EP.ensure()
    # excitation class of every simulation and the shares in the catalogue, the selection pool and the test set
    recs = json.loads((DATA / "manifest.json").read_text())["records"]
    ds = xr.open_dataset(DATA / "dataset.nc").load()
    cls = {r["simulation_id"]: classify(r["folder"]) for r in recs}
    with open(EP.AUDIT / "dataset_class_shares.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["class", "n_large_catalogue", "share_large_catalogue", "n_small_pool", "n_small_test_AP15"])
        for c in CLASSES:
            n = sum(1 for i in cls if cls[i] == c); w.writerow([c, n, f"{n / len(cls):.3f}", sum(cls[i] == c for i in POOL), sum(cls[i] == c for i in TEST)])
    # the examples shown: median sinAPRBS of the pool, APRBS 745, multi-sine 795, median APRBS of the test set
    sin_ids = [i for i in POOL if cls[i] == "sinAPRBS"]; mean_t = lambda ids: {i: float(ds.oven_temperature.sel(simulation_id=i).mean()) for i in ids}
    m = mean_t(sin_ids); sin_id = sorted(sin_ids, key=lambda i: m[i])[len(sin_ids) // 2]; m = mean_t(TEST); test_id = sorted(TEST, key=lambda i: m[i])[len(TEST) // 2]
    shown = [(f"sinAPRBS (pool)", sin_id), ("APRBS (pool)", 745), ("Multi-sine (pool)", 795), ("APRBS (test)", test_id)]; print("shown IDs:", shown)
    tps.apply_style(NAME)
    fig, axs = plt.subplots(1, len(shown), figsize=tps.figsize_for(NAME, 2.0), sharex=True, sharey=True, gridspec_kw={"wspace": 0.2}); t = ds.time.values
    for ax, (title, sid) in zip(axs, shown):
        s = ds.sel(simulation_id=sid)
        ax.plot(t, s.oven_temperature.values, color=COLOR_IN, lw=0.9, ls=":", alpha=0.85, zorder=1)           # excitation in the background
        ax.plot(t, s.temperature.values[:, 0], color=COLOR_A, lw=1.2, zorder=3); ax.plot(t, s.temperature.values[:, 1], color=COLOR_B, lw=1.2, ls="--", zorder=3)
        tps.col_heading(ax, title); ax.set_xlabel("$t$ [s]"); ax.set_xlim(0, 1395); ax.set_xticks([0, 500, 1000])
    axs[0].set_ylabel(r"$T$ [K]")
    tps.legend_above(fig, [Line2D([], [], color=COLOR_IN, lw=1.0, ls=":", label=r"input $T_\mathrm{oven}$"), Line2D([], [], color=COLOR_A, lw=1.3, label=r"core $T_A$"), Line2D([], [], color=COLOR_B, lw=1.3, ls="--", label=r"surface $T_B$")], ncol=3)
    tps.save_figure(fig, EP.IMG_MAIN / NAME, generator="03_thermal_food_processing/figures/make_dataset_figure.py"); plt.close(fig)
    print("wrote", EP.IMG_MAIN / NAME)


if __name__ == "__main__":
    main()
