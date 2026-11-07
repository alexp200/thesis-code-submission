"""n_D = 2 landscape: train ONE K = 3 ensemble on {start, candidate} with exactly the code and recipe of the AL runs (al_runs: tc.train_ensemble, the
member-train-RMSE guard, Kelvin RMSE on the fixed AP15 test set) and write one JSON.
Keys: seed set A = the key of the AL runs at n_D = 2 for the path of that start (7_000_000 + 100_000 p + 1000 + k; optimizer key base + 1) -> a candidate that AL/Random picked in that block must reproduce the stored
n_D = 2 value; seed set B = 7_050_000 + 100_000 p + 1000 + k. Starts without a path in the original design (745, 795, 861, 873, 905; EXTRA): A = 8_001_000 + 100_000 i, B = 8_051_000 + 100_000 i. Within (start, architecture, seed set) all candidates share the initialisation seeds (paired).
Usage (repository root, CPU / x64 / one native thread, see run_landscape.sh):
  python experiments/03_thermal_food_processing/side_studies/n2_landscape/landscape_run.py --arch NODE --start 853 --cand 874 --set A
Output: results/{arch}/start{S}/cand{C}_set{A|B}.json (+ .eqx next to it). Resumable: an existing JSON is not recomputed."""
import argparse
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
WARM = REPO / "experiments/03_thermal_food_processing/al_runs"
STORED_WARM = REPO / "runs/03_AL_chicken/warmstart_path_AL/small/results"


def warm_result(*parts):
    """AL record of the ten original starts: the local run (al_runs/study/results) if it has the file, else the stored record of the thesis runs."""
    local = WARM / "study/results" / Path(*parts)
    return local if local.exists() else STORED_WARM / Path(*parts)


DATA = REPO / "data" / "chicken_data"                                                           # directory with dataset.nc (30 simulations)
POOL = [745, 795, 835, 851, 853, 861, 868, 873, 874, 888, 899, 902, 905, 906, 917]
TEST = [313, 320, 344, 378, 383, 407, 412, 415, 461, 462, 466, 467, 474, 508, 528]
BLOCK = {853: 0, 917: 1, 874: 2, 906: 3, 888: 4, 835: 5, 868: 6, 899: 7, 902: 8, 851: 9}      # start -> path index p of the original design
RECIPE = {"NODE": dict(steps=30000, lr=1e-4, l2=1e-6), "sPHNN": dict(steps=20000, lr=1e-4, l2=1e-6)}


EXTRA = {745: 0, 795: 1, 861: 2, 873: 3, 905: 4}      # starts without a path in the original design: index i -> n_D = 1 key 8_000_000 + 100_000 i, set A 8_001_000 + 100_000 i, set B 8_051_000 + 100_000 i


SET_OFFSET = {"A": 0, "B": 50_000, "C": 70_000}      # offsets inside the 100_000 key block of a start: set A = the keys of the AL runs, B and C independent sets (no overlap, no collision with the random-repeat keys); the landscape uses A and B, the n_D = 1 models A, B and C


def key_base(start, seed_set):
    """Training key of the n_D = 2 ensemble of a start and seed set (members +k, optimizer key +1)."""
    off = SET_OFFSET[seed_set]
    if start in BLOCK:
        return 7_000_000 + off + 100_000 * BLOCK[start] + 1000
    return 8_001_000 + off + 100_000 * EXTRA[start]


def n1_key(start, seed_set="A"):
    """key base of the n_D = 1 ensemble of a start: the warm-start study's iter00 key for a block start, else 8_000_000 + 100_000 i (members +k, optimizer +1)"""
    return key_base(start, seed_set) - 1000                  # the n_D = 1 key of seed set X is the n_D = 2 key of set X minus 1000 (paired key schedule of the warm-start study); A = the stored iter00 / n1_models keys


def check_env():
    """Require the single-thread CPU setting of the validated runs."""
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    if any(os.environ.get(n) != "1" for n in names) or "--xla_cpu_parallel_codegen_split_count=1" not in os.environ.get("XLA_FLAGS", "").split():
        raise SystemExit("use run_landscape.sh (single native thread, XLA flag) so that the numerics match the cluster runs")


def main():
    """Train one landscape ensemble and write its JSON (and checkpoint)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["NODE", "sPHNN"])
    ap.add_argument("--start", required=True, type=int)
    ap.add_argument("--cand", required=True, type=int)
    ap.add_argument("--set", dest="seed_set", required=True, choices=["A", "B"])
    ap.add_argument("--steps", type=int, default=None, help="override the number of optimizer steps (smoke tests only)")
    ap.add_argument("--outdir", type=Path, default=HERE / "results")
    a = ap.parse_args()
    if a.start not in POOL or a.cand not in POOL or a.start == a.cand:
        raise SystemExit("start and candidate must be two different pool IDs")
    out = a.outdir / a.arch / f"start{a.start}" / f"cand{a.cand}_set{a.seed_set}.json"
    if out.exists():
        print("exists", out); return
    check_env()
    sys.path.insert(0, str(WARM))
    import equinox as eqx
    import jax
    import numpy as np
    numerical = importlib.import_module("numerical")
    numerical.DATA = DATA                  # same dataset file as the warm-start small study (sha256 checked below)
    tc, _cd = numerical.helpers()
    from src.ensembles import filter_members_by_rmse, member_train_rmse
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("CPU / x64 required")
    ids = [a.start, a.cand]
    data = numerical.prepare_data(ids)
    rec = dict(RECIPE[a.arch]); steps = a.steps or rec["steps"]
    base = key_base(a.start, a.seed_set)
    t0 = time.perf_counter()
    ensemble, history = tc.train_ensemble(a.arch, list(range(3)), data, steps=steps, lr=rec["lr"], key_offset=base, log_every=max(steps // 10, 1), l2_regularization=rec["l2"])
    jax.block_until_ready(ensemble)
    if not all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves(ensemble) if eqx.is_array(x)):
        raise SystemExit("nonfinite trained parameters")
    train_s = time.perf_counter() - t0
    members = tc.split_ensemble(ensemble, 3)
    losses = member_train_rmse(members, data["t"], data["x0s_train"], data["ys_train"], us_train=data["us_train"])
    _, filt = filter_members_by_rmse(members, losses)
    fn = numerical.predictor(data)
    idx = np.asarray([data["all_ids"].index(s) for s in TEST])
    pred = numerical.predict_batch(fn, ensemble, data["x"][idx], data["u"][idx], 15)
    err = pred * np.asarray(data["ys_std"]) + 279.15 - data["target_K"][idx][None]
    if not np.all(np.isfinite(err)):
        raise SystemExit("nonfinite evaluation")
    per_traj = np.sqrt(np.mean(err ** 2, axis=(2, 3)))                 # (3, 15)
    member = np.sqrt(np.mean(err ** 2, axis=(1, 2, 3)))                # (3,)
    kept = [k for k in range(3) if k not in filt["dropped_indices"]]
    res = dict(arch=a.arch, start=a.start, candidate=a.cand, seed_set=a.seed_set, train_ids=ids, training_key=base, optimizer_key=base + 1, steps=steps, smoke=a.steps is not None,
               member_rmse_K=member.tolist(), median_K=float(np.median(member)), per_trajectory_rmse_K=per_traj.tolist(), ensemble_mean_rmse_K=float(np.sqrt(np.mean((err.mean(axis=0)) ** 2))),
               train_rmse_latent=np.asarray(losses).tolist(), dropped_indices=filt["dropped_indices"], filter_cutoff=filt["cutoff"], effective_members=len(kept), train_seconds=train_s,
               total_seconds=time.perf_counter() - t0, host=platform.node(), warm_block=BLOCK.get(a.start))
    out.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(out.with_suffix(".eqx"), ensemble)
    if os.environ.get("LANDSCAPE_SAVE_HISTORY") == "1":
        history.save(out.with_name(out.stem + "_history.pkl"))
    out.write_text(json.dumps(res))
    print(f"{a.arch} start {a.start} cand {a.cand} set {a.seed_set}: member median {res['median_K']:.3f} K, members {np.round(member, 3).tolist()}, train {train_s:.0f} s, dropped {filt['dropped_indices']}", flush=True)


if __name__ == "__main__":
    main()
