"""n_D = 1 ensemble (K = 3) of a start with the recipe of the iter00 models of the AL runs, needed as the current model for the offline criteria scoring (criteria_offline).
Seed set A is needed only for the five starts without a path in the original design (745, 795, 861, 873, 905); the other starts use iter00.eqx of the AL runs (key 7_000_000 + 100_000 p). Seed sets B and C: all 15 starts.
Key: landscape_run.n1_key (members +k, optimizer +1); the n_D = 1 key of set X is the n_D = 2 key of landscape set X minus 1000, so the n_D = 1 model of a set is paired with the n_D = 2 landscape of the same set.
Usage (repository root, environment of run_landscape.sh):  python n1_run.py --arch sPHNN --start 795 [--set B]        Output: n1_models/{arch}/start{S}[_set{B|C}].eqx + .json (resumable)."""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from landscape_run import DATA, RECIPE, WARM, check_env, n1_key


def main():
    """Train the n_D = 1 ensemble of one start and seed set and write its checkpoint and JSON."""
    ap = argparse.ArgumentParser(); ap.add_argument("--arch", required=True, choices=["NODE", "sPHNN"]); ap.add_argument("--start", required=True, type=int); ap.add_argument("--set", dest="seed_set", default="A", choices=["A", "B", "C"]); ap.add_argument("--outdir", type=Path, default=HERE / "n1_models")
    a = ap.parse_args(); tag = "" if a.seed_set == "A" else f"_set{a.seed_set}"; out = a.outdir / a.arch / f"start{a.start}{tag}.json"
    if out.exists():
        print("exists", out); return
    check_env(); sys.path.insert(0, str(WARM))
    import equinox as eqx
    import jax
    import numpy as np
    numerical = importlib.import_module("numerical"); numerical.DATA = DATA
    tc, _ = numerical.helpers()
    from src.ensembles import member_train_rmse
    data = numerical.prepare_data([a.start]); rec = RECIPE[a.arch]; base = n1_key(a.start, a.seed_set); t0 = time.perf_counter()
    ens, _ = tc.train_ensemble(a.arch, [0, 1, 2], data, steps=rec["steps"], lr=rec["lr"], key_offset=base, log_every=max(rec["steps"] // 10, 1), l2_regularization=rec["l2"])
    jax.block_until_ready(ens)
    losses = np.asarray(member_train_rmse(tc.split_ensemble(ens, 3), data["t"], data["x0s_train"], data["ys_train"], us_train=data["us_train"]))
    out.parent.mkdir(parents=True, exist_ok=True); eqx.tree_serialise_leaves(out.with_suffix(".eqx"), ens)
    out.write_text(json.dumps(dict(arch=a.arch, start=a.start, train_ids=[a.start], seed_set=a.seed_set, training_key=base, optimizer_key=base + 1, steps=rec["steps"], train_rmse_latent=losses.tolist(), train_seconds=time.perf_counter() - t0)))
    print(f"{a.arch} n_D=1 start {a.start}: train RMSE (latent) {np.round(losses, 4).tolist()}, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
