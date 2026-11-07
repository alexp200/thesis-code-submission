"""Train ONE task of the Random-repeats plan (plan.json, made by plan.py): a K = 3 ensemble on the given training IDs with the given keys (same code and recipe as landscape_run.py / the AL runs:
tc.train_ensemble, member-train-RMSE guard, Kelvin RMSE on the fixed AP15 test set). Output results/{arch}/start{S}/N{N}_task{T}.json (+ .eqx next to it); resumable.
Usage (repository root, single-thread CPU / x64 environment of run_landscape.sh):  python random_repeat_run.py --task 17          (smoke: --steps 50 --outdir /tmp/x)
Reproduction check: python random_repeat_run.py --reproduce 0 3   trains the stored Random path 0 of the AL runs at N = 3 with its keys and compares with the stored value."""
import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "n2_landscape"))
from landscape_run import BLOCK, DATA, TEST, WARM, check_env, warm_result


def run(arch, ids, base, steps, lr, l2, out, meta):
    """Train a K = 3 ensemble on the training ids, evaluate it on AP15 and write the JSON and checkpoint."""
    import equinox as eqx
    import jax
    import numpy as np
    sys.path.insert(0, str(WARM))
    numerical = importlib.import_module("numerical"); numerical.DATA = DATA
    tc, _ = numerical.helpers()
    from src.ensembles import filter_members_by_rmse, member_train_rmse
    data = numerical.prepare_data(ids); t0 = time.perf_counter()
    ens, _ = tc.train_ensemble(arch, [0, 1, 2], data, steps=steps, lr=lr, key_offset=base, log_every=max(steps // 10, 1), l2_regularization=l2)
    jax.block_until_ready(ens)
    if not all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves(ens) if eqx.is_array(x)):
        raise SystemExit("nonfinite trained parameters")
    members = tc.split_ensemble(ens, 3); losses = member_train_rmse(members, data["t"], data["x0s_train"], data["ys_train"], us_train=data["us_train"])
    _, filt = filter_members_by_rmse(members, losses)
    fn = numerical.predictor(data); idx = np.asarray([data["all_ids"].index(s) for s in TEST])
    pred = numerical.predict_batch(fn, ens, data["x"][idx], data["u"][idx], 15)
    err = pred * np.asarray(data["ys_std"]) + 279.15 - data["target_K"][idx][None]
    if not np.all(np.isfinite(err)):
        raise SystemExit("nonfinite evaluation")
    member = np.sqrt(np.mean(err ** 2, axis=(1, 2, 3)))
    res = dict(arch=arch, train_ids=ids, training_key=base, optimizer_key=base + 1, steps=steps, lr=lr, l2=l2, member_rmse_K=member.tolist(), median_K=float(np.median(member)), per_trajectory_rmse_K=np.sqrt(np.mean(err ** 2, axis=(2, 3))).tolist(),
               train_rmse_latent=np.asarray(losses).tolist(), dropped_indices=filt["dropped_indices"], train_seconds=time.perf_counter() - t0, host=platform.node(), **meta)
    out.parent.mkdir(parents=True, exist_ok=True); eqx.tree_serialise_leaves(out.with_suffix(".eqx"), ens); out.write_text(json.dumps(res))
    print(f"{arch} {ids}: AP15 member median {res['median_K']:.3f} K, members {np.round(member, 3).tolist()}, {res['train_seconds']:.0f} s", flush=True)
    return res


def main():
    """Run one task of the plan, or the reproduction check."""
    ap = argparse.ArgumentParser(); ap.add_argument("--task", type=int); ap.add_argument("--reproduce", nargs=2, type=int, metavar=("BLOCK", "N")); ap.add_argument("--arch", default="sPHNN")
    ap.add_argument("--steps", type=int, default=None); ap.add_argument("--outdir", type=Path, default=HERE / "results"); a = ap.parse_args()
    if a.reproduce:
        p, N = a.reproduce; src = warm_result(a.arch, "random/random", f"path{p:02d}")
        meta = json.loads((src / f"iter{N - 1:02d}_train.json").read_text()); rep = json.loads((src / f"iter{N - 1:02d}.json").read_text())
        from landscape_run import RECIPE
        rec = RECIPE[a.arch]; check_env()
        res = run(a.arch, meta["train_ids"], meta["initialization_seeds"][0], a.steps or rec["steps"], rec["lr"], rec["l2"], a.outdir / "reproduce" / f"{a.arch}_b{p}_N{N}.json", dict())
        print(f"stored warm-start value {rep['evaluation']['all_members']['median_K']:.4f} K, recomputed {res['median_K']:.4f} K"); return
    plan = json.loads((HERE / "plan.json").read_text()); t = plan["tasks"][a.task]
    out = a.outdir / t["arch"] / f"start{t['start']}" / f"N{t['N']}_task{t['task']}.json"
    if out.exists():
        print("exists", out); return
    check_env()
    run(t["arch"], t["train_ids"], t["training_key"], a.steps or t["steps"], t["lr"], t["l2"], out, dict(task=t["task"], start=t["start"], N=t["N"], paths=t["paths"], smoke=a.steps is not None))


if __name__ == "__main__":
    main()
