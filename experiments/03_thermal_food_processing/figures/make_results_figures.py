"""Result figures and tables of the thermal food processing experiment (Section 4.2.1, Appendix B.3); all numbers of the text come from here.
Figures:  fig05a_saturation_whiskers (Figure 4.8)   AP15 test RMSE over n_D = 1..5 for the six AL strategies against the Random expectation (15 starts; Random = mean of 10 random paths per start, n_D = 2 exact over the 14 candidates, n_D = 1 the shared start ensemble)
          fig08_path_ecdf (Figure 4.9)              distribution of the path outcomes: 150 Random paths against the 15 AL paths of each strategy
          fig06_picks_and_outcome (Figure 4.10)     left: gain over the Random expectation per temperature bucket of the picked trajectory; right: mean oven temperature of the start against that of the pick (n_D = 1 -> 2)
          fig07_pick_quality (Figure 4.11)          share of the gap between the Random expectation and the best candidate that the pick closes
          figA1_saturation_full (Figure B.8)        AP15 test RMSE over n_D = 1..15 against the single stored Random path of each start (ten original starts)
Tables (tabular only):  table_start_dependence, tableA_{al_vs_random, al_vs_random_runs, path_ecdf, temperature_selection, uq_agreement, fig07_closure, seed_reproducibility}.tex;
          the authored tables table_chicken_settings and table_chicken_splits are copied from figures/authored_tables/. Figure 4.7 (dataset) is made by make_dataset_figure.py, the schematics by experiment_graph/, the cost figure by cost_analysis/.
Sources: AP15 evaluations of the AL runs (al_overview/studies_io.py), n2_landscape/merged (landscape of the second trajectory, seed sets A and B), criteria_offline/results (offline scores of the n_D = 1 ensembles, seed sets A, B, C), random_repeats/results.
The numbers behind every figure and table are written to final_audit/figure_data/*.csv. Usage: python figures/make_results_figures.py"""
import shutil
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, str(HERE.parents[1] / "shared")); sys.path.insert(0, str(HERE / "al_overview"))
import export_paths as EP
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr

import studies_io as S
import thesis_plot_style as tps
from thesis_plot_style import style as TS

SIDE = EP.ROOT / "side_studies"
LAND = SIDE / "n2_landscape/merged"
CRIT = SIDE / "criteria_offline/results"
RR = SIDE / "random_repeats/results"
POOL = [745, 795, 835, 851, 853, 861, 868, 873, 874, 888, 899, 902, 905, 906, 917]
ARCHS = ("sPHNN", "NODE")
STREAMS = [(m, c) for m in ("MoLA", "DE", "LA") for c in ("sum_marginal", "sum_variance")]
CRIT_LABEL = {"sum_marginal": "MGE", "sum_variance": "VAR"}                 # naming convention of the thesis
GEN = "03_thermal_food_processing/figures/make_results_figures.py"
FIGDATA = {}                                   # name -> DataFrame written to thesis_figures/final_audit/figure_data/<name>.csv


# ------------------------------------------------------------------------------------------------------------------------------ data helpers
def oven_means():
    """Mean oven temperature of every pool trajectory (from the raw data if available, otherwise from the frozen means in metadata/)."""
    raw = EP.REPO / "data/chicken_data_full_clean/dataset.nc"
    if not raw.exists():                                                                  # submission: frozen per-simulation means (derived from the raw data, see DATA_ACCESS.md)
        t = pd.read_csv(Path(__file__).resolve().parents[1] / "metadata/oven_means.csv").set_index("simulation_id").oven_temperature_mean_degC
        return {i: float(t[i]) for i in POOL}
    ds = xr.open_dataset(raw).load()
    return {i: float(ds.oven_temperature.sel(simulation_id=i).mean()) for i in POOL}


def landscape():
    """n_D = 2 landscape: AP15 RMSE of the ensemble trained on (start, candidate) for seed sets A and B, with best / worst / Random-expectation level of every start."""
    L = pd.read_csv(LAND / "landscape_candidates.csv"); L["rmse"] = L[["A", "B"]].mean(axis=1)
    g = L.groupby(["arch", "start"]).rmse
    L["best"], L["worst"], L["rand"] = g.transform("min"), g.transform("max"), g.transform("mean")
    L["pos"] = (L.rmse - L.best) / (L.worst - L.best)
    return L


def scores():
    """Offline acquisition scores of the n_D = 1 ensemble (seed set A) for every start."""
    return {a: pd.concat([pd.read_csv(f) for f in sorted(CRIT.glob(f"v2_{a}_start[0-9][0-9][0-9].csv"))]) for a in ARCHS}


def pick_table(L, W, temp):
    """one row per (architecture, start, UQ method, criterion): the second trajectory actually chosen by the stored AL path (n_D = 2), its mean oven temperature, RMSE_2 of the landscape, gain over the Random expectation of the start"""
    A = W[(W.N == 2) & (W.method != "Random")]; rows = []
    for r in A.itertuples():
        pick = int(r.train_ids[1]); st = int(r.train_ids[0]); g = L[(L.arch == r.arch) & (L.start == st) & (L.cand == pick)].iloc[0]
        rows.append(dict(arch=r.arch, start=st, method=r.method, crit=r.criterion, pick=pick, T_start=temp[st], T_pick=temp[pick], rmse=g.rmse, rand=g.rand, best=g.best, gain=g.rand - g.rmse))
    return pd.DataFrame(rows).sort_values(["arch", "start", "method", "crit"]).reset_index(drop=True)


def pct(x):
    """Share as a LaTeX percentage."""
    return f"{100 * x:.0f}\\,\\%"


def write_tex(name, lines, folder):
    """Write a tabular-only fragment to `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.tex").write_text("\n".join(lines) + "\n")


def pick_table_offline(V, L, temp):
    """Pick of every offline score (argmax over the 14 candidates) per architecture, start, method and criterion."""
    rows = []
    for arch in ARCHS:
        for st, d in V[arch].groupby("start"):
            d = d.reset_index(drop=True)
            for m, c in STREAMS:
                rows.append(dict(arch=arch, start=st, method=m, crit=c, pick=int(d.cand.iloc[int(np.argmax(d[f"{m}_{c}"].to_numpy()))])))
    return pd.DataFrame(rows)


def check_picks(W, P):
    """consistency of the stored AL runs with the offline scores: the second trajectory of every AL path (n_D = 2) must be the argmax of the offline score of the same start / stream"""
    A = W[(W.N == 2) & (W.method != "Random")].copy(); A["pick_al"] = A.train_ids.apply(lambda t: t[1]); m = A.merge(P.rename(columns={"crit": "criterion", "pick": "pick_off"}), on=["arch", "start", "method", "criterion"])
    print(f"check: AL pick of the stored paths = offline pick in {int((m.pick_al == m.pick_off).sum())} of {len(m)} cases ({len(A)} AL paths)")


# ------------------------------------------------------------------------------------------------------------------------------ Figure 4.8: saturation, n_D = 1..5, Random expectation
def saturation_per_start(W, rs):
    """one row per (architecture, AL stream, start, n_D = 1..5): the node error of the single AL path of that start (median of the K = 3 member RMSEs on AP15), the range of the members,
    and the Random expectation of the start (mean over the 10 random paths, n_D = 2: exact mean over the 14 candidates; n_D = 1: the shared start ensemble); ratio = AL / Random expectation"""
    A = W[(W.method != "Random") & (W.N <= 5)]; ex = rs.set_index(["arch", "start", "N"])["mean"]; rows = []
    for r in A.itertuples():
        rand = r.ap15 if r.N == 1 else float(ex[(r.arch, r.start, r.N)])
        rows.append(dict(arch=r.arch, stream=f"{r.method}/{r.criterion}", start=r.start, N=r.N, al=r.ap15, member_min=min(r.members), member_max=max(r.members), rand=rand, ratio=r.ap15 / rand))
    return pd.DataFrame(rows)


SAT_LEGEND = lambda rand_label: [Line2D([], [], color="k", lw=1.2, label="sPHNN"), Line2D([], [], color="k", lw=1.2, ls="--", label="NODE"), Line2D([], [], color="0.6", lw=0.9, label=rand_label)]    # method colours: column headings
SAT_PANELS = [(m, c) for c in ("sum_variance", "sum_marginal") for m in ("MoLA", "DE", "LA")]           # rows: VAR (top), MGE (bottom); columns: MoLA, DE, LA (colour); stated in the caption, not in the panels


def sat_axes(ax, k, xlim, xticks):
    """Common axes of the learning-curve panels (log RMSE axis, labels on the outer panels, UQ method as column heading)."""
    ax.set_yscale("log"); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_yticks([1, 2, 4]); ax.set_yticklabels(["1", "2", "4"]); ax.set_xlim(*xlim); ax.set_ylim(0.65, 6.5); ax.set_xticks(xticks)
    if k % 3 == 0:
        ax.set_ylabel("Test RMSE [K]")
    if k // 3 == 1:
        ax.set_xlabel(r"$n_D$")
    if k < 3:
        tps.col_heading(ax, SAT_PANELS[k][0])                                      # UQ method, outside the axes


def random_runs(L):
    """all single Random runs: n_D = 2: the 14 candidate ensembles of every start (seed set A, the exact distribution of a random second trajectory); n_D = 3..5: the ten random paths of every start"""
    C = pd.read_csv(RR / "random_cells.csv"); R2 = L[["arch", "start", "A"]].rename(columns={"A": "rmse"}).assign(N=2)
    return pd.concat([R2, C[C.N >= 3][["arch", "start", "N", "rmse"]]], ignore_index=True)


def fig_saturation(PS):
    """Figure 4.8: medians over the 15 starts with the quartiles over the starts as whiskers (AL and Random offset horizontally by 0.07 as a grouping offset).
    Random = Random expectation of every start (mean over its random paths). Columns MoLA, DE, LA, rows VAR (top) and MGE (bottom)."""
    name = "fig05a_saturation_whiskers"
    tps.apply_style(name)
    fig, axs = plt.subplots(2, 3, figsize=tps.figsize_for(name, 3.45), sharex=True, sharey=True, gridspec_kw={"hspace": 0.15, "wspace": 0.1})
    for k, (m, c) in enumerate(SAT_PANELS):
        ax = axs[k // 3, k % 3]
        for arch, ls in (("sPHNN", "-"), ("NODE", "--")):
            d = PS[(PS.arch == arch) & (PS.stream == f"{m}/{c}")]
            for col, key, lw, dx in (("0.6", "rand", 0.9, 0.07), (TS.METHOD_COLOR[m], "al", 1.2, -0.07)):
                g = d.groupby("N")[key]; med = g.median(); ax.plot(med.index, med.values, color=col, lw=lw, ls=ls, zorder=2)
                q1, q3 = g.quantile(.25), g.quantile(.75); x = med.index.to_numpy() + dx
                ax.vlines(x, q1.values, q3.values, color=col, lw=0.5, ls=ls, zorder=1); ax.plot(x, q1.values, ls="", marker="_", ms=2.5, mew=0.4, color=col); ax.plot(x, q3.values, ls="", marker="_", ms=2.5, mew=0.4, color=col)
                ax.plot(x, med.values, ls="", marker="o", ms=1.6, mew=0, color=col, zorder=3)                  # small dot at the median on every whisker
        sat_axes(ax, k, (0.7, 5.3), range(1, 6))
    tps.row_headings([(list(axs[0]), "VAR"), (list(axs[1]), "MGE")])
    tps.legend_above(fig, SAT_LEGEND("Random expectation"), ncol=3)
    tps.save_figure(fig, EP.IMG_MAIN / name, generator=GEN)
    q = lambda x, a: x.quantile(a)
    FIGDATA["fig05_saturation"] = PS.groupby(["arch", "stream", "N"]).agg(AL_median_K=("al", "median"), AL_q25_K=("al", lambda x: q(x, .25)), AL_q75_K=("al", lambda x: q(x, .75)), AL_min_K=("al", "min"), AL_max_K=("al", "max"), AL_mean_K=("al", "mean"),
                                                                          Random_expectation_median_K=("rand", "median"), Random_expectation_q25_K=("rand", lambda x: q(x, .25)), Random_expectation_q75_K=("rand", lambda x: q(x, .75)), Random_expectation_mean_K=("rand", "mean"), n_starts=("start", "nunique")).reset_index()
    FIGDATA["fig05_per_start"] = PS


def fig_saturation_full(W):
    """appendix: n_D = 1..15; both architectures in one panel per stream (sPHNN solid, NODE dashed); grey = the single stored Random path of each start (median over the starts)"""
    rows = []; name = "figA1_saturation_full"; tps.apply_style(name); fig, axs = plt.subplots(2, 3, figsize=tps.figsize_for(name, 3.45), sharex=True, sharey=True, gridspec_kw={"hspace": 0.15, "wspace": 0.1})
    for k, (m, c) in enumerate(SAT_PANELS):
        ax = axs[k // 3, k % 3]
        for arch, ls in (("sPHNN", "-"), ("NODE", "--")):
            g = W[(W.arch == arch) & (W.method == m) & (W.criterion == c)]; rnd = W[(W.arch == arch) & (W.method == "Random")]
            for d, col, lw, nm in ((rnd, "0.6", 0.9, "Random"), (g, TS.METHOD_COLOR[m], 1.2, f"{m}/{c}")):
                te = d.groupby("N")["ap15"].median(); q = d.groupby("N")["ap15"].quantile([.25, .75]).unstack(); ax.plot(te.index, te.values, color=col, lw=lw, ls=ls)
                rows += [dict(arch=arch, stream=nm, N=int(n), test_ap15_median_K=te[n], test_ap15_q25_K=q.loc[n, 0.25], test_ap15_q75_K=q.loc[n, 0.75]) for n in te.index]
        sat_axes(ax, k, (1, 15), [1, 5, 10, 15]); ax.set_yticks([0.6, 1, 2, 4]); ax.set_yticklabels(["0.6", "1", "2", "4"])
    FIGDATA["figA1_saturation_full"] = pd.DataFrame(rows).drop_duplicates()
    tps.row_headings([(list(axs[0]), "VAR"), (list(axs[1]), "MGE")])
    tps.legend_above(fig, SAT_LEGEND("Random (one stored path per start)"), ncol=3)
    tps.save_figure(fig, EP.IMG_APP / name, generator=GEN)


# ------------------------------------------------------------------------------------------------------------------------------ Figure 4.10 and Table B.13: picks (start vs pick temperature, gain per temperature bucket)
def fig_picks(L, P, temp):
    """Figure 4.10 (gain per temperature bucket of the pick; start against pick temperature) and Table B.13; returns the bucket edges."""
    q = np.quantile(list(temp.values()), [1 / 3, 2 / 3]); tmax = max(temp.values())
    BUCKET = [f"$\\leq$ {q[0]:.0f}", f"$\\leq$ {q[1]:.0f}", f"$\\leq$ {tmax:.0f}"]; terc = lambda x: np.where(np.asarray(x) <= q[0], 0, np.where(np.asarray(x) <= q[1], 1, 2))
    L = L.assign(D=L.rand - L.rmse, bucket=terc([temp[c] for c in L.cand])); P = P.assign(bucket=terc(P.T_pick))
    figkey = "fig06_picks_and_outcome"; tps.apply_style(figkey)
    fig, axs = plt.subplots(2, 2, figsize=tps.figsize_for(figkey, 4.2), sharex="col", gridspec_kw={"wspace": 0.45, "hspace": 0.2}); w = 0.115; rows, stat = [], []
    Tb = r"$\overline{T}_\mathrm{oven}$"
    def mark(ax, x, y, m, c, s_filled, s_open):
        if c == "sum_marginal":
            ax.scatter(x, y, s=s_filled, marker="o", color=TS.METHOD_COLOR[m], alpha=0.6, lw=0, zorder=2)
        else:
            ax.scatter(x, y, s=s_open, marker="s", facecolors="none", edgecolors=TS.METHOD_COLOR[m], linewidths=0.6, alpha=0.9, zorder=2)
    for r_, arch in enumerate(ARCHS):
        Pa, La = P[P.arch == arch], L[L.arch == arch]; ax = axs[r_, 0]                                  # left: gain per temperature bucket of the picked trajectory
        for b in range(3):
            for k, key in enumerate(["all"] + STREAMS):
                x = b + (k - 3) * w
                if key == "all":
                    v = La[La.bucket == b]; ax.scatter(np.full(len(v), x), v.D, s=3, color="0.7", lw=0, zorder=1); ax.hlines(v.D.median(), x - w * 0.42, x + w * 0.42, color="k", lw=1.3, zorder=3)
                    rows.append(dict(arch=arch, group="all candidates", bucket=BUCKET[b], n=len(v), share=len(v) / len(La), median_gain_K=v.D.median(), mean_gain_K=v.D.mean())); continue
                m, c = key; s = Pa[(Pa.method == m) & (Pa.crit == c)]; v = s[s.bucket == b]
                if len(v):
                    mark(ax, np.full(len(v), x), v.gain, m, c, 9, 8); ax.hlines(v.gain.median(), x - w * 0.42, x + w * 0.42, color="k", lw=1.3, zorder=3)
                rows.append(dict(arch=arch, group=f"{m}, {CRIT_LABEL[c]}", bucket=BUCKET[b], n=len(v), share=len(v) / len(s), median_gain_K=v.gain.median() if len(v) else np.nan, mean_gain_K=v.gain.mean() if len(v) else np.nan))
        for xb in (0.5, 1.5):
            ax.axvline(xb, color="0.85", lw=0.7, zorder=0)
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels(BUCKET); ax.set_xlim(-0.55, 2.55); ax.grid(False)
        ax.set_ylabel("RMSE gain [K]")
        ax = axs[r_, 1]                                                                                 # right: mean oven temperature of the n_D = 1 trajectory against that of the pick
        for (m, c) in STREAMS:
            s = Pa[(Pa.method == m) & (Pa.crit == c)]; mark(ax, s.T_start, s.T_pick, m, c, 14, 11)
        ref = Pa[(Pa.method == "MoLA") & (Pa.crit == "sum_marginal")]; rp, pp = pearsonr(ref.T_start, ref.T_pick); rs_, ps = spearmanr(ref.T_start, ref.T_pick); stat.append(dict(arch=arch, n=len(ref), pearson=rp, p_pearson=pp, spearman=rs_, p_spearman=ps))
        ax.set_ylabel(f"{Tb} of pick [K]")
    axs[1, 0].set_xlabel(f"{Tb} of pick [K]"); axs[1, 1].set_xlabel(f"{Tb} of $n_D=1$ trajectory [K]")
    tps.row_headings([(list(axs[r]), a) for r, a in enumerate(ARCHS)])
    tps.legend_above(fig, [Line2D([], [], marker="|", ls="", ms=9, mew=2.2, color=TS.METHOD_COLOR[m], label=m) for m in ("LA", "DE", "MoLA")] + [Line2D([], [], marker="o", ls="", ms=4, color="0.35", label="MGE"), Line2D([], [], marker="s", ls="", ms=4, mfc="none", mec="0.35", label="VAR"), Line2D([], [], marker="o", ls="", ms=2.5, color="0.7", label="all candidates"), Line2D([], [], color="k", lw=1.3, label="median")], ncol=4)
    tps.save_figure(fig, EP.IMG_MAIN / figkey, generator=GEN)
    S_, B_ = pd.DataFrame(stat), pd.DataFrame(rows); FIGDATA["fig06_picks"] = P; FIGDATA["fig06_correlation"] = S_; FIGDATA["fig06_bucket_table"] = B_
    cnt = []
    for arch in ARCHS:
        a = P[P.arch == arch]; cnt.append([np.mean(a.bucket == b) for b in range(3)])
    pearson = {r.arch: (r.pearson, r.spearman) for r in S_.itertuples()}
    lines = ["\\begin{tabular}{@{}lrrr@{}}", "\\toprule", "\\multicolumn{4}{@{}l}{\\emph{Start and pick temperature: MoLA/MGE}} \\\\", "Architecture & $n_{\\mathrm{start}}$ & Pearson $r$ & Spearman $\\rho$ \\\\", "\\midrule"]
    lines += [f"{a} & {int(S_[S_.arch == a].n.iloc[0])} & {pearson[a][0]:.2f} & {pearson[a][1]:.2f} \\\\" for a in ARCHS]
    lines += ["\\midrule", "\\multicolumn{4}{@{}l}{\\emph{Selection shares: all six strategies}} \\\\", "Architecture & " + " & ".join(f"$\\leq{u:.0f}$" for u in (q[0], q[1], tmax)) + " \\\\", "\\midrule"]
    lines += [f"{a} & " + " & ".join(pct(x) for x in c) + " \\\\" for a, c in zip(ARCHS, cnt)] + [f"Uniform candidate reference & " + " & ".join(pct(np.mean(terc(list(temp.values())) == b)) for b in range(3)) + " \\\\", "\\bottomrule", "\\end{tabular}"]
    write_tex("tableA_temperature_selection", lines, EP.TAB_APP)
    return q


# ------------------------------------------------------------------------------------------------------------------------------ Figure 4.11 and Table B.15: gap closed by the pick
def fig_pick_quality(P):
    """Figure 4.11 (boxplots of the gap closure over the 15 starts) and Table B.15 (median [Q25, Q75])."""
    P = P.assign(closure=(P.rand - P.rmse) / (P.rand - P.best)); key = "fig07_pick_quality"; tps.apply_style(key); fig, axs = plt.subplots(1, 2, figsize=tps.figsize_for(key, 2.75), sharey=True, gridspec_kw={"wspace": 0.22})
    for ax, arch in zip(axs, ARCHS):
        for j, c in enumerate(("sum_variance", "sum_marginal")):
            for k, m in enumerate(("MoLA", "LA", "DE")):
                v = P[(P.arch == arch) & (P.crit == c) & (P.method == m)].closure.to_numpy(); x = j + (k - 1) * 0.27
                ax.boxplot([v], positions=[x], widths=0.22, patch_artist=True, showfliers=True, flierprops=dict(marker="o", ms=2, mfc="none", mec="0.4", mew=0.5), medianprops=dict(color="k", lw=0.9), boxprops=dict(facecolor=TS.METHOD_COLOR[m], alpha=0.8, lw=0.5), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        ax.axhline(0, color="0.35", ls="--", lw=0.9); ax.axhline(1, color="#009E73", lw=0.9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["VAR", "MGE"]); tps.col_heading(ax, arch); ax.set_xlim(-0.6, 1.6)
    axs[0].set_ylabel(r"Gap closure $\gamma(s,c)$")
    mh = {m: Patch(color=TS.METHOD_COLOR[m], label=m) for m in ("MoLA", "LA", "DE")}; rh, bh = Line2D([], [], color="0.35", ls="--", lw=0.9, label="Random expectation (0)"), Line2D([], [], color="#009E73", lw=0.9, label="best candidate (1)")
    tps.legend_above(fig, [mh["MoLA"], rh, mh["LA"], bh, mh["DE"]], ncol=3)                              # column-major fill: row 1 MoLA, LA, DE; row 2 Random expectation, best candidate
    tps.save_figure(fig, EP.IMG_MAIN / key, generator=GEN)
    T = P.groupby(["arch", "method", "crit"]).closure.agg(median="median", mean="mean", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), minimum="min").reset_index(); FIGDATA["fig07_pick_quality_summary"] = T; FIGDATA["fig07_pick_quality_per_pick"] = P
    lines = ["\\begin{tabular}{@{}llrrr@{}}", "\\toprule", "Architecture & Criterion & LA & DE & MoLA \\\\", "\\midrule"]
    for arch in ARCHS:
        for c in ("sum_marginal", "sum_variance"):
            t = T[(T.arch == arch) & (T.crit == c)].set_index("method"); lines.append(f"{arch} & {CRIT_LABEL[c]} & " + " & ".join(f"{t.loc[m, 'median']:.2f} [{t.loc[m, 'q25']:.2f}, {t.loc[m, 'q75']:.2f}]" for m in ("LA", "DE", "MoLA")) + " \\\\")
    write_tex("tableA_fig07_closure", lines + ["\\bottomrule", "\\end{tabular}"], EP.TAB_APP)


# ------------------------------------------------------------------------------------------------------------------------------ Tables B.16 and B.17: start dependence and seed reproducibility
def seed_set_agreement():
    """Per architecture, start, method and criterion: do the offline scores of the n_D = 1 ensembles of the seed sets A, B and C select the same candidate?"""
    rows = []
    for arch in ARCHS:
        for st in POOL:
            d = [pd.read_csv(CRIT / f"v2_{arch}_start{st}{suf}.csv") for suf in ("", "_setB", "_setC")]
            for m, c in STREAMS:
                picks = [int(x.cand.iloc[int(np.argmax(x[f"{m}_{c}"].to_numpy()))]) for x in d]
                rows.append(dict(arch=arch, start=st, method=m, crit=c, pA=picks[0], pB=picks[1], pC=picks[2], same_all=len(set(picks)) == 1))
    return pd.DataFrame(rows)


def tables_start_dependence(L, V):
    """Table B.16 (agreement of the rankings of two different starts) and Table B.17 (candidate ranking under seed sets A and B; picks under seed sets A, B, C)."""
    rows = []
    for arch in ARCHS:
        la = L[L.arch == arch]; true = la.pivot(index="cand", columns="start", values="rmse"); A = la.pivot(index="cand", columns="start", values="A"); B = la.pivot(index="cand", columns="start", values="B")
        sc = {(m, c): V[arch].pivot(index="cand", columns="start", values=f"{m}_{c}") for m, c in STREAMS}
        for s, t in combinations(POOL, 2):
            sh = [i for i in POOL if i not in (s, t)]
            rows.append(dict(arch=arch, kind="true gain, two different starts", same=int(true.loc[sh, s].idxmin() == true.loc[sh, t].idxmin()), rho=spearmanr(true.loc[sh, s], true.loc[sh, t])[0]))
            for (m, c), tab in sc.items():
                rows.append(dict(arch=arch, kind=f"{m}, {CRIT_LABEL[c]}: score, two different starts", same=int(tab.loc[sh, s].idxmax() == tab.loc[sh, t].idxmax()), rho=spearmanr(tab.loc[sh, s], tab.loc[sh, t])[0]))
        for s in POOL:
            sh = [i for i in POOL if i != s]; rows.append(dict(arch=arch, kind="true gain, same start, seed set A vs B", same=int(A.loc[sh, s].idxmin() == B.loc[sh, s].idxmin()), rho=spearmanr(A.loc[sh, s], B.loc[sh, s])[0]))
    R7 = pd.DataFrame(rows); T7 = R7.groupby(["arch", "kind"]).agg(n=("same", "size"), top1=("same", "mean"), rho_median=("rho", "median"), rho_q25=("rho", lambda x: x.quantile(.25)), rho_q75=("rho", lambda x: x.quantile(.75))).reset_index(); FIGDATA["table_start_dependence"] = T7
    order = ["true gain, two different starts"] + [f"{m}, {CRIT_LABEL[c]}: score, two different starts" for m, c in STREAMS]
    lab = {order[0]: "Candidate test error"}; lab.update({o: o.replace(": score, two different starts", "") for o in order[1:]})
    lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", " & \\multicolumn{2}{c}{sPHNN} & \\multicolumn{2}{c}{NODE} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Quantity / strategy & top-1 & $\\rho$ [Q25, Q75] & top-1 & $\\rho$ [Q25, Q75] \\\\", "\\midrule"]
    for o in order:
        cells = []
        for arch in ("sPHNN", "NODE"):
            r = T7[(T7.arch == arch) & (T7.kind == o)].iloc[0]; cells += [pct(r.top1), f"{r.rho_median:.2f} [{r.rho_q25:.2f}, {r.rho_q75:.2f}]"]
        lines.append(f"{lab[o]} & " + " & ".join(cells) + " \\\\")
        if o == order[0]:
            lines += ["\\midrule", "\\multicolumn{5}{@{}l}{\\emph{Acquisition scores}} \\\\"]
    write_tex("table_start_dependence", lines + ["\\midrule", f"chance level & {pct(1 / (len(POOL) - 2))} & 0 & {pct(1 / (len(POOL) - 2))} & 0 \\\\", "\\bottomrule", "\\end{tabular}"], EP.TAB_APP)
    FIGDATA["seed_set_picks"] = ST = seed_set_agreement(); ab = "true gain, same start, seed set A vs B"; r_ = {a_: T7[(T7.arch == a_) & (T7.kind == ab)].iloc[0] for a_ in ARCHS}
    lines = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule", "Quantity / strategy & sPHNN & NODE \\\\", "\\midrule", "Same lowest-error candidate, A vs B & " + " & ".join(pct(r_[a_].top1) for a_ in ("sPHNN", "NODE")) + " \\\\",
             "Error-rank correlation, A vs B & " + " & ".join(f"{r_[a_].rho_median:.2f} [{r_[a_].rho_q25:.2f}, {r_[a_].rho_q75:.2f}]" for a_ in ("sPHNN", "NODE")) + " \\\\",
             "Median absolute error difference [K] & " + " & ".join(f"{(L[L.arch == a_].A - L[L.arch == a_].B).abs().median():.3f}" for a_ in ("sPHNN", "NODE")) + " \\\\", "\\midrule", "\\multicolumn{3}{@{}l}{\\emph{Identical pick under seed sets A, B, C}} \\\\"]
    for m, c in STREAMS:
        lines.append(f"{m}, {CRIT_LABEL[c]} & " + " & ".join(pct(ST[(ST.arch == a_) & (ST.method == m) & (ST.crit == c)].same_all.mean()) for a_ in ("sPHNN", "NODE")) + " \\\\")
    write_tex("tableA_seed_reproducibility", lines + ["\\bottomrule", "\\end{tabular}"], EP.TAB_APP)


# ------------------------------------------------------------------------------------------------------------------------------ Tables B.14, B.10, B.11: UQ agreement, AL against Random
def table_uq_agreement(P):
    """Table B.14: how often LA, DE and MoLA pick the same second trajectory."""
    ag = []
    for arch in ARCHS:
        for c in ("sum_marginal", "sum_variance"):
            pk = P[(P.arch == arch) & (P.crit == c)].pivot(index="start", columns="method", values="pick"); mode = int(pk.stack().mode().iloc[0])
            ag.append(dict(arch=arch, criterion=c, LA_eq_DE=(pk.LA == pk.DE).mean(), LA_eq_MoLA=(pk.LA == pk.MoLA).mean(), DE_eq_MoLA=(pk.DE == pk.MoLA).mean(), all_three=((pk.LA == pk.DE) & (pk.DE == pk.MoLA)).mean(), modal_candidate=mode, modal_share=float((pk.stack() == mode).mean())))
    A = pd.DataFrame(ag); FIGDATA["uq_pick_agreement"] = A
    write_tex("tableA_uq_agreement", ["\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule", "Architecture & Criterion & LA = DE & LA = MoLA & DE = MoLA & all three & most frequent pick \\\\", "\\midrule"]
              + [f"{r.arch} & {CRIT_LABEL[r.criterion]} & {pct(r.LA_eq_DE)} & {pct(r.LA_eq_MoLA)} & {pct(r.DE_eq_MoLA)} & {pct(r.all_three)} & ID {r.modal_candidate} ({pct(r.modal_share)}) \\\\" for r in A.itertuples()] + ["\\bottomrule", "\\end{tabular}"], EP.TAB_APP)


TIE_TOL_K = 1e-12                                                  # a single Random run is only counted as beaten if Random - AL > 1e-12 K (practically identical errors are no wins; manuscript convention)


def table_al_vs_runs(PS, RUNS):
    """Table B.11: share of the single Random runs of a start whose error is larger than the error of the AL run of that start (mean over the 15 starts); 0.5 = AL is not better than a random run"""
    Ns = (2, 3, 4, 5); rows = []
    for r in PS[PS.N >= 2].itertuples():
        ru = RUNS[(RUNS.arch == r.arch) & (RUNS.start == r.start) & (RUNS.N == r.N)].rmse.to_numpy(); rows.append(dict(arch=r.arch, stream=r.stream, start=r.start, N=r.N, win=float(((ru - r.al) > TIE_TOL_K).mean())))
    Wn = pd.DataFrame(rows); FIGDATA["al_vs_random_runs"] = Wn; m = Wn.groupby(["arch", "stream", "N"]).win.mean().reset_index()
    lines = ["\\begin{tabularx}{\\linewidth}{@{}Xrrrrrrrr@{}}", "\\toprule", " & \\multicolumn{4}{c}{sPHNN} & \\multicolumn{4}{c}{NODE} \\\\", "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
             "Strategy & " + " & ".join(f"$n_D={n}$" for n in Ns) + " & " + " & ".join(f"$n_D={n}$" for n in Ns) + " \\\\", "\\midrule"]
    for mm, c in STREAMS:
        lines.append(f"{mm}, {CRIT_LABEL[c]} & " + " & ".join(pct(m[(m.arch == a) & (m.stream == f'{mm}/{c}') & (m.N == n)].win.iloc[0]) for a in ARCHS for n in Ns) + " \\\\")
    six = Wn.groupby(["arch", "N"]).win.mean(); lines += ["\\midrule", "Mean across strategies & " + " & ".join(pct(six[(a, n)]) for a in ARCHS for n in Ns) + " \\\\"]
    write_tex("tableA_al_vs_random_runs", lines + ["\\bottomrule", "\\end{tabularx}"], EP.TAB_APP)


def table_al_vs_random(PS):
    """Table B.10: paired ratio AL / Random expectation per start; per stream the geometric mean over the 15 starts"""
    Ns = (2, 3, 4, 5); P = PS[PS.N >= 2].assign(lr=lambda d: np.log(d.ratio)); R = P.groupby(["arch", "stream", "N"]).lr.mean().apply(np.exp).rename("ratio").reset_index(); FIGDATA["al_vs_random_ratio"] = R
    lines = ["\\begin{tabularx}{\\linewidth}{@{}Xrrrrrrrr@{}}", "\\toprule", " & \\multicolumn{4}{c}{sPHNN} & \\multicolumn{4}{c}{NODE} \\\\", "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
             "Strategy & " + " & ".join(f"$n_D={n}$" for n in Ns) + " & " + " & ".join(f"$n_D={n}$" for n in Ns) + " \\\\", "\\midrule"]
    for m, c in STREAMS:
        lines.append(f"{m}, {CRIT_LABEL[c]} & " + " & ".join(f"{R[(R.arch == a) & (R.stream == f'{m}/{c}') & (R.N == n)].ratio.iloc[0]:.2f}" for a in ARCHS for n in Ns) + " \\\\")
    write_tex("tableA_al_vs_random", lines + ["\\bottomrule", "\\end{tabularx}"], EP.TAB_APP)


# ------------------------------------------------------------------------------------------------------------------------------ Figure 4.9 and Table B.12: path outcomes
def fig_path_ecdf(PS, blocks):
    """Figure 4.9 and Table B.12: empirical distribution of the path outcomes. Rows: sPHNN, NODE; left column: path error (geometric mean of the node error over n_D = 2..5 of one path), right column: node error at n_D = 5.
    Random: all 150 random paths (15 starts x 10 paths), grey; AL: the 15 paths of each of the six streams (colour = UQ method, thick = MGE, thin = VAR); dashed vertical line = mean of the Random paths."""
    C = pd.read_csv(RR / "random_cells.csv"); C = C[C.start.isin(blocks) & (C.N >= 2) & (C.N <= 5)]
    R = C.groupby(["arch", "start", "r"]).rmse.agg(path_error=lambda x: float(np.exp(np.log(x).mean()))).reset_index()
    R = R.merge(C[C.N == 5][["arch", "start", "r", "rmse"]].rename(columns={"rmse": "end5"}), on=["arch", "start", "r"]); R["end_error"] = R.end5
    A = PS[PS.N >= 2].groupby(["arch", "stream", "start"]).al.agg(path_error=lambda x: float(np.exp(np.log(x).mean()))).reset_index().merge(PS[PS.N == 5][["arch", "stream", "start", "al"]].rename(columns={"al": "end_error"}), on=["arch", "stream", "start"])
    key = "fig08_path_ecdf"; tps.apply_style(key)
    fig, axs = plt.subplots(2, 2, figsize=tps.figsize_for(key, 3.0), sharey="row", gridspec_kw={"wspace": 0.12, "hspace": 0.3}); rows = []
    nice = [0.8, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]
    for r, arch in enumerate(ARCHS):
        for c, col in enumerate(("path_error", "end_error")):
            ax = axs[r, c]; rnd = np.sort(R[R.arch == arch][col].values); med = float(np.median(rnd)); mean = float(np.mean(rnd)); lo = rnd.min()
            startmean_median = float(np.median(R[R.arch == arch].groupby("start")[col].mean().values))        # Random expectation of the learning curves (median over the starts of the per-start mean); not drawn, kept in the summary
            ax.step(rnd, np.arange(1, len(rnd) + 1) / len(rnd), where="post", color="0.45", lw=1.0); ax.axvline(mean, color="0.45", lw=0.8, ls="--")
            for m, crit in STREAMS:
                v = np.sort(A[(A.arch == arch) & (A.stream == f"{m}/{crit}")][col].values); lo = min(lo, v.min())
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=TS.METHOD_COLOR[m], lw=1.3 if crit == "sum_marginal" else 0.6)
            al = A[A.arch == arch][col].values; rows.append(dict(arch=arch, measure=col, random_startmean_median=startmean_median, random_min=rnd.min(), random_median=med, random_mean=mean, random_max=rnd.max(), n_random=len(rnd), worst_AL_MGE=A[(A.arch == arch) & A.stream.str.endswith("sum_marginal")][col].max(), worst_AL_VAR=A[(A.arch == arch) & A.stream.str.endswith("sum_variance")][col].max(),
                                                                    share_random_worse_than_worst_AL=float(np.mean(rnd > al.max())), share_AL_better_than_random_median=float(np.mean(al < med)), share_AL_better_than_random_mean=float(np.mean(al < mean))))
            hi = rnd.max() * 1.1; xl = (lo * 0.93, hi); tk = [t for t in nice if xl[0] < t < xl[1]]
            ax.set_xscale("log"); ax.set_xlim(*xl); ax.set_xticks(tk); ax.set_xticklabels([f"{t:g}" for t in tk]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_ylim(0, 1.02)
            if r == 1:                                                                                 # x label only in the bottom row (the tick values stay: the rows have different ranges)
                ax.set_xlabel("Path error [K]" if col == "path_error" else r"RMSE at $n_D=5$ [K]")
            if c == 0:
                ax.set_ylabel("Share of paths")
    tps.row_headings([(list(axs[r]), a) for r, a in enumerate(ARCHS)])
    leg = [Line2D([], [], color="0.45", lw=1.2, label="Random")] + [Line2D([], [], color=TS.METHOD_COLOR[m], lw=1.4, label=m) for m in ("MoLA", "DE", "LA")] + [Line2D([], [], color="k", lw=1.3, label="MGE"), Line2D([], [], color="k", lw=0.6, label="VAR"), Line2D([], [], color="0.45", lw=0.9, ls="--", label="Random mean")]
    tps.legend_above(fig, leg, ncol=4); tps.save_figure(fig, EP.IMG_MAIN / key, generator=GEN); plt.close(fig); S_ = pd.DataFrame(rows); FIGDATA["fig08_path_ecdf_paths_random"] = R; FIGDATA["fig08_path_ecdf_paths_AL"] = A; FIGDATA["fig08_path_ecdf_summary"] = S_
    f = lambda x: f"{x:.2f}"; L = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule", "& & \\multicolumn{4}{c}{Random [K]} & \\multicolumn{2}{c}{worst AL path [K]} \\\\", "\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}", "& & min & median & mean & max & MGE & VAR \\\\", "\\midrule"]
    for a in ARCHS:
        for col, nm in (("path_error", "path error"), ("end_error", r"RMSE at $n_D=5$")):
            x = S_[(S_.arch == a) & (S_.measure == col)].iloc[0]; L.append(f"{a if col == 'path_error' else ''} & {nm} & {f(x.random_min)} & {f(x.random_median)} & {f(x.random_mean)} & {f(x.random_max)} & {f(x.worst_AL_MGE)} & {f(x.worst_AL_VAR)} \\\\")
        if a == ARCHS[0]:
            L.append("\\midrule")
    write_tex("tableA_path_ecdf", L + ["\\bottomrule", "\\end{tabular}"], EP.TAB_APP)


# ------------------------------------------------------------------------------------------------------------------------------ export and main
def export_authored_tables():
    """Copy the authored tables (figures/authored_tables/) unchanged to the export folder."""
    for n in ("table_chicken_settings", "table_chicken_splits"):
        EP.TAB_APP.mkdir(parents=True, exist_ok=True); shutil.copyfile(EP.ROOT / "figures/authored_tables" / f"{n}.tex", EP.TAB_APP / f"{n}.tex")


def main():
    """Load the run records, make all figures and tables and write the numbers behind them to final_audit/figure_data/."""
    tps.apply_style(); EP.ensure()
    # AP15 evaluations of all AL runs (ten original starts + five additional starts), one row per path and training-set size
    D0 = S.load_all()
    W = D0[D0.study.isin(["warm_small_K3", "warm_extra5_K3"])].rename(columns={"median": "ap15"})[["arch", "method", "criterion", "path", "N", "ap15", "train_ids", "members"]]
    W["start"] = W.train_ids.apply(lambda t: t[0]); blocks = sorted(W.start.unique()); blocks_orig = sorted(W[D0.loc[W.index, "study"] == "warm_small_K3"].start.unique())
    rs = pd.read_csv(RR / "random_summary.csv"); rs = rs[rs.start.isin(blocks)]; L, V, temp = landscape(), scores(), oven_means()
    # learning curves and path outcomes (Figures 4.8, 4.9, B.8; Tables B.10-B.12)
    PS = saturation_per_start(W, rs); fig_saturation(PS); fig_path_ecdf(PS, blocks); fig_saturation_full(W[W.start.isin(blocks_orig)])
    RUNS = random_runs(L); table_al_vs_random(PS); table_al_vs_runs(PS, RUNS)
    # first acquisition: picks against the landscape and the offline scores (Figures 4.10, 4.11; Tables B.13-B.17)
    P = pick_table(L, W, temp); check_picks(W, pick_table_offline(V, L, temp))
    q = fig_picks(L, P, temp); print("temperature bucket edges:", q)
    fig_pick_quality(P); tables_start_dependence(L, V); table_uq_agreement(P)
    export_authored_tables()
    out = EP.AUDIT / "figure_data"; out.mkdir(parents=True, exist_ok=True)
    for k, df in FIGDATA.items():
        df.to_csv(out / f"{k}.csv", index=False)


if __name__ == "__main__":
    main()
