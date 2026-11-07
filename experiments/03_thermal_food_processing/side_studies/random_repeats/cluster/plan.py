"""Plan of the Random-repeats study: a fair Random baseline for the warm-start design (the warm-start study has ONE Random run per start, i.e. one noisy draw).
For each of the 15 pool trajectories as START s and r = 0..9: a random tail = permutation of the other 14 pool IDs (numpy default_rng([20263012, s, r]), identical for NODE and sPHNN). The prefixes define the Random training sets
  n_D = 2: [s, tail0]   (= the landscape ensemble (s, tail0), seed set A: SAME keys, no new training; the value is read from the landscape)
  n_D = 3, 4, 5: [s, tail0, tail1], ...  -> NEW trainings (K = 3 members, fresh training for every n_D, no weight warm-starting).
Training keys (identical for all tails r and for the AL streams of the start -> only the DATA differ between paths, no training-seed variance mixed in):
  key(s, N) = n1_key(s) + 1000 (N - 1), members + k (k = 0, 1, 2), optimizer key key + 1        n1_key(s) = 7_000_000 + 100_000 p (warm-start block p of s) or 8_000_000 + 100_000 i (starts without a block 745, 795, 861, 873, 905)
  -> for the 10 block starts exactly the keys of the warm-start study at the same N; at N = 2 exactly the landscape set A.
Recipe: NODE 30 000 steps, sPHNN 20 000, Adam lr 1e-4, L2 1e-6 (as landscape_run.RECIPE).
Output: plan.json (all tasks) and tasks.tsv (arch start r N train_ids unique_task_id). Tasks with an identical (arch, start, N, set of IDs) are deduplicated (one training, several paths use it).
Usage: python plan.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from landscape_run import POOL, RECIPE, n1_key

R, NS, SEED = 10, (3, 4, 5), 20263012


def key(start, N):
    return n1_key(start) + 1000 * (N - 1)


def main():
    tails = {s: [np.random.default_rng([SEED, s, r]).permutation(sorted(i for i in POOL if i != s)).tolist() for r in range(R)] for s in POOL}
    tasks, seen = [], {}
    for arch in ("NODE", "sPHNN"):
        for s in POOL:
            for r in range(R):
                for N in NS:
                    ids = [s] + tails[s][r][:N - 1]; sig = (arch, s, N, tuple(sorted(ids)))
                    if sig not in seen:
                        seen[sig] = len(seen)
                        rec = RECIPE[arch]
                        tasks.append(dict(task=seen[sig], arch=arch, start=s, N=N, train_ids=ids, training_key=key(s, N), optimizer_key=key(s, N) + 1, steps=rec["steps"], lr=rec["lr"], l2=rec["l2"], paths=[]))
                    tasks[seen[sig]]["paths"].append(r)
    plan = dict(seed=SEED, R=R, N=list(NS), K=3, pool=POOL, tails={str(s): tails[s] for s in POOL}, tasks=tasks,
                landscape_n2=[dict(arch=a, start=s, r=r, candidate=tails[s][r][0]) for a in ("NODE", "sPHNN") for s in POOL for r in range(R)])
    (HERE / "plan.json").write_text(json.dumps(plan))
    with open(HERE / "tasks.tsv", "w") as f:
        f.write("task\tarch\tstart\tN\ttrain_ids\tpaths\n")
        for t in tasks:
            f.write(f"{t['task']}\t{t['arch']}\t{t['start']}\t{t['N']}\t{','.join(map(str, t['train_ids']))}\t{','.join(map(str, t['paths']))}\n")
    print(f"{len(tasks)} unique trainings ({sum(len(t['paths']) for t in tasks)} path-N cells), {len(plan['landscape_n2'])} n_D = 2 cells read from the landscape")
    for a in ("NODE", "sPHNN"):
        print(a, sum(t["arch"] == a for t in tasks))


if __name__ == "__main__":
    main()
