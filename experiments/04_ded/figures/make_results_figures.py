"""Result figures and tables of the DED experiment (Section 4.2.2, Appendix B.4); all numbers of the text come from here.
Figures:  fig04a_saturation_whiskers (Figure 4.13)   RMSE over n_D = 1..5 of the three AL strategies (MGE) against the Random expectation; medians over the starts, quartiles as whiskers
          fig07_path_ecdf (Figure 4.14)              distribution of the path outcomes: Random paths against the AL path of each UQ method
          fig05_pick_map (Figure 4.15)               acquired simulations (AL pooled over LA / DE / MoLA vs Random) on the 5 x 5 operating grid
          fig06_error_ratio (Figure B.10)            per-simulation RMSE ratio AL / Random expectation (geometric mean over starts and methods)
Tables (tabular only):  tableA_al_vs_random, tableA_path_ecdf; the authored tables table_ded_{settings,data,training} are copied from figures/authored_tables/.
Cohort: all starts whose three AL streams and whose Random tails are complete for n_D = 1..5 in both architectures (ten starts). Error of a node = median of the K = 3 member RMSEs over ALL 25 simulations (training trajectories included, no held-out set);
Random expectation of a start = mean over its Random tails. Run records are read through ded_results_io.py; the numbers behind every figure and table are written to final_audit/figure_data/*.csv. No bootstrap intervals.
Usage: python figures/make_results_figures.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, str(HERE.parents[1] / "shared"))
import export_paths as EP
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import ded_results_io as IO
import thesis_plot_style as tps
from thesis_plot_style import style as TS

GEN = "04_ded/figures/make_results_figures.py"
ARCHS = ("sPHNN", "NODE"); METHODS = ("MoLA", "DE", "LA")
FIGDATA = {}                                   # name -> DataFrame written to thesis_figures/final_audit/figure_data/<name>.csv


def cohort(d, nmax=5):
    """per-start table: one row per (model, start, method, N) with the AL node error and the Random expectation (mean over the tails that exist for all n_D = 1..5 in both architectures); returns (PS, runs)"""
    d = d.assign(rand=d.stream.str.startswith("random")); d = d[d.pkg.isin(["A", "T", "G6", "F"]) & ~d.stream.str.startswith("candidate")]
    NSC = tuple(range(1, nmax + 1))
    ok_al = d[~d.rand].groupby(["start", "stream"]).apply(lambda g: all(set(g[g.model == m].N) >= set(NSC) for m in ARCHS))
    al_starts = set(s for s in d.start.unique() if all(ok_al.get((s, x), False) for x in METHODS)) - {0}
    tails = d[d.rand].groupby(["start", "stream"]).apply(lambda g: all(set(g[g.model == m].N) >= set(NSC) for m in ARCHS))
    rows, runs = [], []
    for s in sorted(al_starts):
        tl = [t for (st, t), ok in tails.items() if st == s and ok]
        r = d[(d.start == s) & d.stream.isin(tl) & d.rand]; runs.append(r[["model", "start", "stream", "N", "err"]])
        ex = r.groupby(["model", "N"]).err.mean()
        for x in d[(d.start == s) & ~d.rand & d.stream.isin(METHODS)].itertuples():
            if x.N in NSC:
                rows.append(dict(arch=x.model, stream=x.stream, start=s, N=x.N, al=x.err, member_min=min(x.members), member_max=max(x.members), rand=float(ex[(x.model, x.N)]), n_tails=len(tl)))
    PS = pd.DataFrame(rows); PS["ratio"] = PS.al / PS.rand
    return PS, pd.concat(runs, ignore_index=True).rename(columns={"model": "arch", "err": "rmse"})


LEG = lambda: [Line2D([], [], color="k", lw=1.2, label="sPHNN"), Line2D([], [], color="k", lw=1.2, ls="--", label="NODE"), Line2D([], [], color="0.6", lw=0.9, label="Random expectation")]    # method colours: column headings


# ------------------------------------------------------------------------------------------------ Figure 4.13: learning curves
def fig_saturation(PS, nmax=5):
    """Figure 4.13: one column per UQ method (MGE), lines = medians over the starts, whiskers = quartiles over the starts (AL and Random offset horizontally by 0.07)."""
    NSl = tuple(range(1, nmax + 1)); key = "fig04a_saturation_whiskers"; xlim = (0.7, nmax + 0.3)
    tps.apply_style(key)
    fig, axs = plt.subplots(1, 3, figsize=tps.figsize_for(key, 2.15), sharex=True, sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, m in zip(axs, METHODS):
        for arch, ls in (("sPHNN", "-"), ("NODE", "--")):
            d = PS[(PS.arch == arch) & (PS.stream == m)]
            for col, k, lw, dx in (("0.6", "rand", 0.9, 0.07), (TS.METHOD_COLOR[m], "al", 1.2, -0.07)):
                g = d.groupby("N")[k]; med = g.median(); ax.plot(med.index, med.values, color=col, lw=lw, ls=ls, zorder=2)
                q1, q3 = g.quantile(.25), g.quantile(.75); x = med.index.to_numpy() + dx
                ax.vlines(x, q1.values, q3.values, color=col, lw=0.5, ls=ls, zorder=1); ax.plot(x, q1.values, ls="", marker="_", ms=2.5, mew=0.4, color=col); ax.plot(x, q3.values, ls="", marker="_", ms=2.5, mew=0.4, color=col)
                ax.plot(x, med.values, ls="", marker="o", ms=1.6, mew=0, color=col, zorder=3)
        ax.set_yscale("log"); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100", "200"]); ax.set_xlim(*xlim); ax.set_ylim(1.1, 250); ax.set_xticks(NSl); ax.set_xlabel(r"$n_D$")
        tps.col_heading(ax, m)
    axs[0].set_ylabel("RMSE [K]")
    tps.row_headings([(list(axs), "MGE")])
    tps.legend_above(fig, LEG(), ncol=3); tps.save_figure(fig, EP.IMG_MAIN / key, generator=GEN)
    FIGDATA["fig04_per_start"] = PS


# ------------------------------------------------------------------------------------------------ Table B.21: AL against the Random expectation
def tables(PS, nmax=5):
    """Table B.21: geometric means of the paired ratios AL / Random expectation per method and architecture, and over all methods and starts."""
    NS = tuple(range(1, nmax + 1))
    P = PS[PS.N >= 2].assign(lr=lambda x: np.log(x.ratio)); R = P.groupby(["arch", "stream", "N"]).lr.mean().apply(np.exp).rename("ratio").reset_index()
    Ns = NS[1:]
    # geometric means of the paired ratios per UQ method and architecture, and over all methods and starts (mean of the unrounded log ratios, not of the rounded rows)
    k = len(Ns); C = ["\\begin{tabular}{@{}l" + "r" * (2 * k) + "@{}}", "\\toprule", f" & \\multicolumn{{{k}}}{{c}}{{sPHNN}} & \\multicolumn{{{k}}}{{c}}{{NODE}} \\\\", f"\\cmidrule(lr){{2-{k + 1}}}\\cmidrule(lr){{{k + 2}-{2 * k + 1}}}",
                       "& " + " & ".join(f"$n_D={n}$" for n in Ns) + " & " + " & ".join(f"$n_D={n}$" for n in Ns) + " \\\\", "\\midrule"]
    for m in METHODS:
        C.append(f"{m}, MGE & " + " & ".join(f"{R[(R.arch == a) & (R.stream == m) & (R.N == n)].ratio.iloc[0]:.2f}" for a in ARCHS for n in Ns) + " \\\\")
    allm = {(a, n): float(np.exp(P[(P.arch == a) & (P.N == n)].lr.mean())) for a in ARCHS for n in Ns}                 # = geometric mean over the three methods and the starts
    C += ["\\midrule", "All methods & " + " & ".join(f"{allm[(a, n)]:.2f}" for a in ARCHS for n in Ns) + " \\\\", "\\bottomrule", "\\end{tabular}"]
    (EP.TAB_APP / "tableA_al_vs_random.tex").write_text("\n".join(C) + "\n")


# ------------------------------------------------------------------------------------------------ Figures 4.15 and B.10: the 5 x 5 operating grid (x = source speed v, y = source power Q; cell id = 5 * power index + speed index)
V = [10, 12.5, 15, 17.5, 20]; Q = [300, 350, 400, 450, 500]; CORNERS = [0, 4, 20, 24]


def grid_axes(ax, every=1):
    """5 x 5 operating grid: x = source speed v, y = source power Q; tick labels of every `every`-th cell (the grid always has 5 x 5 cells)."""
    ix = list(range(0, 5, every))
    ax.set_xticks(ix); ax.set_xticklabels([f"{V[i]:g}" for i in ix]); ax.set_yticks(ix); ax.set_yticklabels([str(Q[i]) for i in ix])
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True); ax.set_yticks(np.arange(-0.5, 5, 1), minor=True); ax.grid(which="minor", color="0.85", lw=0.5); ax.grid(which="major", visible=False); ax.tick_params(which="minor", length=0); ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.5)


def fig_pick_map(d, runs, nmax=4):
    """Figure 4.15: which simulations are acquired in the steps n_D = 2..nmax: AL (LA, DE, MoLA pooled) for sPHNN and NODE, and Random (identical draws for both architectures).
    Shading (sequential Blues, 0..1) = share of the eligible paths that acquire the simulation (paths in which the simulation is the start trajectory are excluded from the denominator);
    cell number = mean acquisition step when selected (shown for shares >= 25 %); dashed frames = start trajectories."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    key = "fig05_pick_map"
    tps.apply_style(key)
    starts = sorted(int(x) for x in runs.start.unique()); tails = runs.groupby("start").stream.unique(); A = d[(d.N == nmax) & d.start.isin(starts)]
    panels = [("sPHNN", "AL", "sPHNN, AL"), ("NODE", "AL", "NODE, AL"), ("sPHNN", "Random", "Random")]
    fig, axs = plt.subplots(1, 3, figsize=tps.figsize_for(key, 2.35), sharex=True, sharey=True, gridspec_kw={"wspace": 0.12}); rows = []
    for ax, (arch, kind, title) in zip(axs, panels):
        if kind == "Random":
            sel = [x for x in A[A.model == arch].itertuples() if x.stream in tails[x.start]]
        else:
            sel = [x for x in A[A.model == arch].itertuples() if x.stream in METHODS]
        cnt = np.zeros(25); avail = np.zeros(25); steps = [[] for _ in range(25)]
        for x in sel:
            avail += 1; avail[x.train_ids[0]] -= 1
            for k, t in enumerate(x.train_ids[1:], start=2):
                cnt[t] += 1; steps[t].append(k)
        share = np.divide(cnt, avail, out=np.zeros(25), where=avail > 0)
        ax.imshow(share.reshape(5, 5), origin="lower", cmap="Blues", vmin=0, vmax=1)
        for t in range(25):
            rows.append(dict(panel={"sPHNN, AL": "sPHNN, AL (LA, DE, MoLA)", "NODE, AL": "NODE, AL (LA, DE, MoLA)"}.get(title, title), cell=t, v=V[t % 5], Q=Q[t // 5], runs_available=int(avail[t]), acquisitions=int(cnt[t]), share=share[t], mean_step=float(np.mean(steps[t])) if steps[t] else np.nan))
            if share[t] >= 0.25:
                ax.text(t % 5, t // 5, f"{np.mean(steps[t]):.1f}", ha="center", va="center", fontsize=tps.raw_pt("annot"), color="w" if share[t] > 0.5 else "k")
        for st in starts:
            ax.add_patch(plt.Rectangle((st % 5 - 0.42, st // 5 - 0.42), 0.84, 0.84, fill=False, ec="k", lw=0.7, ls=(0, (2.5, 1.5)), zorder=4))
        grid_axes(ax); ax.set_xlabel("$v$ [mm/s]"); tps.col_heading(ax, title)
        print("pick map", title, "runs", len(sel), "corner share of the acquisitions %.2f" % (cnt[CORNERS].sum() / cnt.sum()))
    axs[0].set_ylabel("$Q$ [W]")
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="Blues"), ax=axs, fraction=0.03, pad=0.02); cb.set_label("Selection frequency"); cb.set_ticks([0, 0.25, 0.5, 0.75, 1]); cb.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])
    tps.legend_above(fig, [Patch(fill=False, ec="k", ls=(0, (2.5, 1.5)), lw=0.7, label="start trajectory")], ncol=1); tps.save_figure(fig, EP.IMG_MAIN / key, generator=GEN); plt.close(fig)
    FIGDATA["fig05_pick_map"] = pd.DataFrame(rows)


def fig_error_ratio(d, runs, nmax=4):
    """Figure B.10: per simulation, RMSE of the AL model divided by the Random expectation (mean over the Random tails of the start) of the same start; geometric mean over the starts and the three UQ methods.
    Diverging RdBu_r on log2 of the ratio (centred on 1, -2..2), unchanged."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    key = "fig06_error_ratio"
    tps.apply_style(key)
    starts = sorted(int(x) for x in runs.start.unique()); tails = runs.groupby("start").stream.unique(); dd = d[d.start.isin(starts)]
    norm = Normalize(-2, 2); fig, axs = plt.subplots(2, nmax - 1, figsize=tps.figsize_for(key, 3.15), sharex=True, sharey=True, gridspec_kw={"wspace": 0.1, "hspace": 0.18}); rows = []
    for r, arch in enumerate(ARCHS):
        g = dd[dd.model == arch]
        for c, n in enumerate(range(2, nmax + 1)):
            ax = axs[r, c]; L = []
            for st in starts:
                rnd = np.mean([np.array(x.pt) for x in g[(g.start == st) & (g.N == n) & g.stream.isin(list(tails[st]))].itertuples()], axis=0)
                for m in METHODS:
                    L.append(np.log(np.array(g[(g.start == st) & (g.N == n) & (g.stream == m)].iloc[0].pt) / rnd))
            M = np.exp(np.mean(L, axis=0)); ax.imshow(np.log2(M).reshape(5, 5), origin="lower", cmap="RdBu_r", norm=norm)
            for t in range(25):
                ax.text(t % 5, t // 5, f"{M[t]:.1f}", ha="center", va="center", fontsize=tps.raw_pt("annot"), color="w" if abs(np.log2(M[t])) > 1.2 else "k"); rows.append(dict(arch=arch, N=n, cell=t, v=V[t % 5], Q=Q[t // 5], ratio=M[t]))
            grid_axes(ax, every=2)
            if r == 0:
                tps.col_heading(ax, f"$n_D={n}$")
            if r == 1:
                ax.set_xlabel("$v$ [mm/s]")
        axs[r, 0].set_ylabel("$Q$ [W]")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="RdBu_r"), ax=axs, fraction=0.025, pad=0.02); cb.set_ticks([-2, -1, 0, 1, 2]); cb.set_ticklabels(["0.25", "0.5", "1", "2", "4"]); cb.set_label("RMSE AL / RMSE Random expectation")
    tps.row_headings([(list(axs[r]), a) for r, a in enumerate(ARCHS)])
    tps.save_figure(fig, EP.IMG_MAIN / key, generator=GEN); plt.close(fig); FIGDATA["fig06_error_ratio"] = pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ Figure 4.14 and Table B.22: path outcomes
def path_errors(d, runs, nmax=4):
    """one value per path: error at n_D = nmax and path error = geometric mean of the node error over n_D = 2..nmax; AL: one path per start and method, Random: every Random run of every start"""
    starts = sorted(int(x) for x in runs.start.unique()); tails = runs.groupby("start").stream.unique(); dd = d[d.start.isin(starts)]; rows = []
    for arch in ARCHS:
        for st in starts:
            for strm in list(METHODS) + sorted(tails[st]):
                g = dd[(dd.model == arch) & (dd.start == st) & (dd.stream == strm)].sort_values("N"); e = np.array([g[g.N == n].err.iloc[0] for n in range(1, nmax + 1)])
                rows.append(dict(arch=arch, start=st, stream=strm, kind="AL" if strm in METHODS else "Random", path_error=float(np.exp(np.mean(np.log(e[1:])))), end_error=float(e[-1])))
    return pd.DataFrame(rows)


def fig_path_ecdf(d, runs, nmax=4):
    """Figure 4.14 and Table B.22: empirical distribution of the path outcomes. Rows: sPHNN, NODE; left column: path error (geometric mean of the node error over n_D = 2..nmax), right column: error at n_D = nmax.
    Random: all paths of all starts (one curve), AL: the paths of each UQ method (one per start); dashed vertical line = mean of the Random paths."""
    from matplotlib.lines import Line2D
    key = "fig07_path_ecdf"
    tps.apply_style(key)
    P = path_errors(d, runs, nmax); fig, axs = plt.subplots(2, 2, figsize=tps.figsize_for(key, 3.0), sharey="row", gridspec_kw={"wspace": 0.12, "hspace": 0.3}); rows = []
    ticks = {("sPHNN", "path_error"): [2, 3, 5, 10], ("NODE", "path_error"): [14, 20, 30, 50], ("sPHNN", "end_error"): [1.5, 2, 3, 5, 10], ("NODE", "end_error"): [12, 14, 16, 19]}
    for r, arch in enumerate(ARCHS):
        for c, col in enumerate(("path_error", "end_error")):
            ax = axs[r, c]; g = P[P.arch == arch]; rnd = np.sort(g[g.kind == "Random"][col].values); med = float(np.median(rnd)); mean = float(np.mean(rnd)); lo = rnd.min()
            startmean_median = float(np.median(g[g.kind == "Random"].groupby("start")[col].mean().values))        # Random expectation of the learning curves (median over the starts of the per-start mean); not drawn, kept in the summary
            ax.step(rnd, np.arange(1, len(rnd) + 1) / len(rnd), where="post", color="0.45", lw=1.0)
            ax.axvline(mean, color="0.45", lw=0.8, ls="--")
            for m in METHODS:
                v = np.sort(g[g.stream == m][col].values); ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=TS.METHOD_COLOR[m], lw=1.1); lo = min(lo, v.min())
            worst = g[g.kind == "AL"][col].max(); rows.append(dict(arch=arch, measure=col, random_startmean_median=startmean_median, random_min=rnd.min(), random_median=med, random_mean=mean, random_max=rnd.max(), n_random=len(rnd), share_random_worse_than_worst_AL=float(np.mean(rnd > worst)), worst_AL=worst,
                                                                    **{f"worst_{m}": g[g.stream == m][col].max() for m in METHODS}, share_AL_better_than_random_median=float(np.mean(g[g.kind == "AL"][col].values < med)), share_AL_better_than_random_mean=float(np.mean(g[g.kind == "AL"][col].values < mean))))
            ax.set_xscale("log"); ax.set_xticks(ticks[(arch, col)]); ax.set_xticklabels([f"{t:g}" for t in ticks[(arch, col)]]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlim(lo * 0.93, rnd.max() * 1.1); ax.set_ylim(0, 1.02)
            if r == 1:                                                                                 # x label only in the bottom row (the tick values stay: the rows have different ranges)
                ax.set_xlabel("Path error [K]" if col == "path_error" else f"RMSE at $n_D={nmax}$ [K]")
            if c == 0:
                ax.set_ylabel("Share of paths")
    tps.row_headings([(list(axs[r]), a) for r, a in enumerate(ARCHS)])
    tps.legend_above(fig, [Line2D([], [], color="0.45", lw=1.2, label="Random")] + [Line2D([], [], color=TS.METHOD_COLOR[m], lw=1.4, label=m) for m in METHODS] + [Line2D([], [], color="0.45", lw=0.9, ls="--", label="Random mean")], ncol=5)
    tps.save_figure(fig, EP.IMG_MAIN / key, generator=GEN); plt.close(fig); S = pd.DataFrame(rows); FIGDATA["fig07_path_ecdf_paths"] = P; FIGDATA["fig07_path_ecdf_summary"] = S
    f = lambda x: f"{x:.2f}" if x < 10 else f"{x:.1f}"
    L = ["\\begin{tabular}{@{}llrrrrrrr@{}}", "\\toprule", "& & \\multicolumn{4}{c}{Random [K]} & \\multicolumn{3}{c}{worst AL path [K]} \\\\", "\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}", "& & min & median & mean & max & MoLA & DE & LA \\\\", "\\midrule"]
    for arch in ARCHS:
        for col, nm in (("path_error", "path error"), ("end_error", f"RMSE at $n_D={nmax}$")):
            x = S[(S.arch == arch) & (S.measure == col)].iloc[0]; L.append(f"{arch if col == 'path_error' else ''} & {nm} & {f(x.random_min)} & {f(x.random_median)} & {f(x.random_mean)} & {f(x.random_max)} & {f(x.worst_MoLA)} & {f(x.worst_DE)} & {f(x.worst_LA)} \\\\")
        L.append("\\midrule" if arch == ARCHS[0] else "")
    L = [l for l in L if l != ""] + ["\\bottomrule", "\\end{tabular}"]
    (EP.TAB_APP / "tableA_path_ecdf.tex").write_text("\n".join(L) + "\n")


def export_authored_tables():
    """Copy the authored tables (figures/authored_tables/) unchanged to the export folder."""
    import shutil
    for n in ("table_ded_data", "table_ded_settings", "table_ded_training"):
        EP.TAB_APP.mkdir(parents=True, exist_ok=True); shutil.copyfile(EP.ROOT / "figures/authored_tables" / f"{n}.tex", EP.TAB_APP / f"{n}.tex")


def main():
    """Load the run records, make all figures and tables and write the numbers behind them to final_audit/figure_data/."""
    tps.apply_style(); EP.ensure(); d = IO.load_new()
    NMAX = 5                                            # all starts complete to n_D = 5 (10 starts, 6 Random runs per start)
    PS, runs = cohort(d, NMAX); print(f"n_D <= {NMAX}: cohort starts", [int(x) for x in sorted(PS.start.unique())], "tails per start:", [int(x) for x in sorted(PS.n_tails.unique())])
    fig_saturation(PS, NMAX); tables(PS, NMAX)
    fig_pick_map(d, runs, NMAX); fig_error_ratio(d, runs, NMAX); fig_path_ecdf(d, runs, NMAX)
    export_authored_tables()
    out = EP.AUDIT / "figure_data"; out.mkdir(parents=True, exist_ok=True)
    for k, df in FIGDATA.items():
        df.to_csv(out / f"{k}.csv", index=False)


if __name__ == "__main__":
    main()
