"""Random baseline from the repeated random paths (plan.py, random_repeat_run.py): the Random expectation of every start and training-set size.
Cells: Random path r at start s and size N: N = 2 -> the landscape ensemble (s, tail0(r)), seed set A (same keys); N = 3, 4, 5 -> the trained ensembles of the plan (results/{arch}/start{S}/N{N}_task{T}.json, AP15 member median in K).
Outputs (results/): random_cells.csv (arch, start, r, N, rmse) and random_summary.csv (per arch, start, N: number, mean, sd, min and max of the Random paths; read by figures/make_results_figures.py).
For N = 2 the summary is exact: it uses all 14 candidates of the landscape instead of the ten sampled tail-0 draws.
Usage: python random_repeats_analysis.py [results_dir_of_random_repeats [landscape_dir]]"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BK = HERE.parents[3] / "runs/03_AL_chicken/investigations"
RR = Path(sys.argv[1]) if len(sys.argv) > 1 else BK / "random_repeats/cluster/run/results"
LD = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "n2_landscape/merged"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)


def random_cells():
    """RMSE of every Random path r at every size N = 2..5 (N = 2 from the landscape, N >= 3 from the trained ensembles of the plan)."""
    plan = json.loads((HERE / "plan.json").read_text()); rows = []
    for t in plan["tasks"]:
        f = RR / t["arch"] / f"start{t['start']}" / f"N{t['N']}_task{t['task']}.json"
        if f.exists():
            v = json.loads(f.read_text())["median_K"]
            for r in t["paths"]:
                rows.append(dict(arch=t["arch"], start=t["start"], r=r, N=t["N"], rmse=v))
    for c in plan["landscape_n2"]:
        f = LD / c["arch"] / f"start{c['start']}" / f"cand{c['candidate']}_setA.json"
        if f.exists():
            rows.append(dict(arch=c["arch"], start=c["start"], r=c["r"], N=2, rmse=json.loads(f.read_text())["median_K"]))
    return pd.DataFrame(rows)


def main():
    C = random_cells(); C.to_csv(OUT / "random_cells.csv", index=False)
    print("Random cells:", len(C), "of", 15 * 10 * 2 * 4, "(missing trainings are skipped)"); print(C.groupby(["arch", "N"]).size().unstack().to_string())
    # N >= 3: mean over the ten random paths of a start
    R = pd.DataFrame([dict(arch=a, start=s, N=N, n=len(g), mean=g.rmse.mean(), sd=g.rmse.std(), min=g.rmse.min(), max=g.rmse.max()) for (a, s, N), g in C[C.N != 2].groupby(["arch", "start", "N"])])
    # N = 2: exact mean over all 14 candidates of the landscape (seed set A, the keys of the AL runs)
    Lc = pd.concat([pd.DataFrame([json.loads(f.read_text())]) for f in LD.glob("*/start*/cand*_setA.json")])[["arch", "start", "candidate", "median_K"]]
    ex2 = Lc.groupby(["arch", "start"]).median_K.agg(["mean", "std", "min", "max", "size"]).reset_index().rename(columns={"size": "n", "std": "sd"}); ex2["N"] = 2
    R = pd.concat([R, ex2[R.columns]]).sort_values(["arch", "start", "N"]).reset_index(drop=True)
    R.to_csv(OUT / "random_summary.csv", index=False)


if __name__ == "__main__":
    main()
