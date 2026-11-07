"""Table of the n_D = 2 landscape (landscape_run.py): for every architecture, start and candidate the AP15 test RMSE [K] (median over the K = 3 members) of the ensemble trained on {start, candidate}, for seed sets A and B.
Output: <results_dir>/landscape_candidates.csv (columns arch, start, cand, A, B, mean; read by figures/make_results_figures.py), a printed summary per start (Random expectation = mean over the candidates,
best and worst candidate) and the check that seed set A reproduces the n_D = 2 values stored in the AL runs. Only starts with a complete set A are used.
Usage: python landscape_analysis.py [results_dir]"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
WARM = HERE.parents[3] / "runs/03_AL_chicken/warmstart_path_AL/small/results"
BLOCK = {853: 0, 917: 1, 874: 2, 906: 3, 888: 4, 835: 5, 868: 6, 899: 7, 902: 8, 851: 9}
POOL = [745, 795, 835, 851, 853, 861, 868, 873, 874, 888, 899, 902, 905, 906, 917]
STREAMS = [(m, c) for m in ("LA", "DE", "MoLA") for c in ("sum_marginal", "sum_variance")]
pd.set_option("display.width", 220)


def load_landscape():
    """One row per trained ensemble: architecture, start, candidate, seed set and median member RMSE."""
    rows = []
    for f in glob.glob(str(RES / "*" / "start*" / "cand*_set?.json")):
        d = json.loads(Path(f).read_text())
        if d.get("smoke"):
            continue
        rows.append(dict(arch=d["arch"], start=d["start"], cand=d["candidate"], set=d["seed_set"], rmse=d["median_K"]))
    return pd.DataFrame(rows)


def stored_values(arch, p):
    """Pick and test RMSE at n_D = 2 of the stored AL runs (Random and the six AL streams) of block p."""
    out = {}
    for m, c in [("random", "random")] + STREAMS:
        f = WARM / arch / m / c / f"path{p:02d}" / "iter01.json"
        if f.exists():
            d = json.loads(f.read_text()); out[(m, c)] = (d["train_ids"][-1], d["evaluation"]["all_members"]["median_K"])
    return out


def main():
    L = load_landscape()
    if L.empty:
        print("no results yet"); return
    cand_rows, repro = [], []
    for (arch, start), g in L.groupby(["arch", "start"]):
        piv = g.pivot_table(index="cand", columns="set", values="rmse")
        cands = [c for c in POOL if c != start]
        if "A" not in piv or piv["A"].reindex(cands).isna().any():
            print(f"{arch} start {start}: seed set A incomplete ({int(piv['A'].notna().sum()) if 'A' in piv else 0}/14), skipped"); continue
        hasB = "B" in piv and not piv["B"].reindex(cands).isna().any()
        A = piv["A"].reindex(cands); B = piv["B"].reindex(cands) if hasB else None; M = (A + B) / 2 if hasB else A
        for c in cands:
            cand_rows.append(dict(arch=arch, start=start, cand=c, A=A[c], B=B[c] if hasB else np.nan, mean=M[c]))
        print(f"{arch}, start {start} (block {BLOCK.get(start, '-')}): Random expectation {M.mean():.3f} K, best candidate {M.idxmin()} {M.min():.3f} K, worst {M.idxmax()} {M.max():.3f} K")
        # seed set A against the stored n_D = 2 values of the AL runs of the same start
        info = stored_values(arch, BLOCK[start]) if start in BLOCK else {}
        for (m, c), (pick, stored) in info.items():
            repro.append(float(A[pick]) - stored)
            print(f"   {m + '/' + c:22s} pick {pick}: set A {A[pick]:.3f} K, stored {stored:.3f} K (diff {repro[-1]:+.3f})")
    pd.DataFrame(cand_rows).to_csv(RES / "landscape_candidates.csv", index=False)
    if repro:
        print("reproduction of the stored n_D = 2 values with seed set A: median |diff| =", round(float(np.median(np.abs(repro))), 3), "K, max", round(float(np.max(np.abs(repro))), 3), "K")


if __name__ == "__main__":
    main()
