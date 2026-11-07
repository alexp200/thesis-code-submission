"""Level 2 (thermal food processing): recompute the test errors of stored active-learning nodes from checkpoint + raw data + preprocessing and compare them with the stored records.

For every selected node (arch / method / criterion / path / iteration) the script
  1. reads the stored record runs/03_AL_chicken/.../iterNN.json (training IDs, initialisation keys, stored member RMSEs [K]),
  2. rebuilds the training-only normalisation from the raw data (data/chicken_data/dataset.nc, supervisor-provided, see docs/DATA_ACCESS.md),
  3. loads the K = 3 member checkpoint iterNN.eqx, rolls every member out on the 15 fixed AP15 holdout trajectories and computes the member RMSE exactly as al_runs/run.py::evaluate does,
  4. compares member RMSEs and the node error (median of the three member RMSEs, the quantity of all thermal figures and tables) with the stored values.

    python tools/recompute_thermal_nodes.py                      # 8 nodes spread over both studies / architectures / methods (about 1-2 min)
    python tools/recompute_thermal_nodes.py --sample 40 --max-iter 5
    python tools/recompute_thermal_nodes.py --record runs/03_AL_chicken/warmstart_path_AL/small/results/NODE/LA/sum_marginal/path00/iter02.json

The original runs were produced on x86-64 Linux CPUs (float64, single native thread); a different CPU / BLAS gives small floating-point differences, amplified by the adaptive step-size control of the ODE solver.
A node passes if every member RMSE agrees with the stored value within 1e-6 K + rtol x stored value (default rtol = 1e-3; the thesis displays node errors with two decimals, i.e. a relative
resolution of about 1e-3 to 1e-2).  No training is performed.  Exit code 1 if a node exceeds the tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WARM = ROOT / "experiments/03_thermal_food_processing/al_runs"
RUNS = ROOT / "runs/03_AL_chicken/warmstart_path_AL"
STUDIES = {"small": (RUNS / "small/results", WARM / "study/plan.json"), "extra5": (RUNS / "small_extra5/run/results", RUNS / "small_extra5/run/plan.json")}
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WARM))


def sha(p: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def records(max_iter: int) -> list[Path]:
    """Stored evaluation records up to training-set size max_iter + 1."""
    out = []
    for study, (res, _) in STUDIES.items():
        out += [p for p in sorted(res.glob("*/*/*/path??/iter??.json")) if int(p.stem[4:]) <= max_iter and p.with_suffix(".eqx").exists()]
    return out


def study_of(rec: Path) -> str:
    """Study ('small' or 'extra5') a record belongs to."""
    return "extra5" if "small_extra5" in rec.parts else "small"


def recompute(rec: Path, plans: dict) -> dict:
    """Re-evaluate the stored ensemble of one record on the AP15 test set and compare with the stored errors."""
    import equinox as eqx
    from numerical import predict_batch, predictor, prepare_data, skeleton
    study = study_of(rec)
    arch, method, criterion, path = rec.parts[-5:-1]
    d = json.loads(rec.read_text())
    plan = plans[study]
    ids = d["train_ids"]
    t0 = time.time()
    data = prepare_data(ids)
    ens = eqx.tree_deserialise_leaves(rec.with_suffix(".eqx"), skeleton(arch, d["initialization_seeds"]))
    fn, batch = predictor(data), plan["acquisition"]["evaluation_batch"]
    idx = np.asarray([data["all_ids"].index(s) for s in plan["evaluation_ids"]])
    x, u, target = data["x"][idx], data["u"][idx], data["target_K"][idx]
    pred = np.concatenate([predict_batch(fn, ens, x[s:s + batch], u[s:s + batch], batch) for s in range(0, len(idx), batch)], axis=1)
    err = pred * np.asarray(data["ys_std"]) + 279.15 - target[None]
    mse = np.mean(err ** 2, axis=2)                                   # [member, trajectory, probe]
    mem = np.sqrt(mse.mean(axis=(1, 2)))
    stored = np.asarray([m["rmse_K"] for m in sorted(d["evaluation"]["members"], key=lambda m: m["member_index"])])
    return dict(study=study, arch=arch, method=method, criterion=criterion, path=path, iteration=d["iteration"], n_D=d["dataset_size"], stored_members=stored.tolist(), recomputed_members=mem.tolist(),
                stored_node_error=float(np.median(stored)), recomputed_node_error=float(np.median(mem)), max_abs_member_diff=float(np.abs(mem - stored).max()), max_rel_member_diff=float((np.abs(mem - stored) / np.abs(stored)).max()),
                abs_node_diff=float(abs(np.median(mem) - np.median(stored))), seconds=round(time.time() - t0, 1), record=str(rec.relative_to(ROOT)))


def main():
    """Recompute a sample of stored records."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=8, help="number of nodes (deterministic, evenly spread over the sorted record list)")
    ap.add_argument("--max-iter", type=int, default=4, help="largest iteration (n_D = iteration + 1... see record) considered for the sample")
    ap.add_argument("--record", action="append", default=[], help="explicit record path(s) (relative to the submission root); overrides --sample")
    ap.add_argument("--all", action="store_true", help="every record (hours)")
    ap.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance of a member RMSE (plus 1e-6 K absolute)")
    ap.add_argument("--out", type=Path, default=ROOT / "regenerated/level2_thermal_nodes.json")
    a = ap.parse_args()
    data_file = ROOT / "data/chicken_data/dataset.nc"
    if not data_file.exists():
        sys.exit("data/chicken_data/dataset.nc is missing: Level 2 needs the supervisor-provided raw data (docs/DATA_ACCESS.md).")
    plans = {k: json.loads(p.read_text()) for k, (_, p) in STUDIES.items()}
    for k, p in plans.items():
        if p["dataset_sha256"] != sha(data_file):
            sys.exit(f"dataset.nc differs from the data the {k} study was run with (plan: {p['dataset_sha256']})")
    if a.record:
        sel = [ROOT / r for r in a.record]
    else:
        allr = records(10 ** 6 if a.all else a.max_iter)
        sel = allr if a.all else [allr[int(i)] for i in np.linspace(0, len(allr) - 1, a.sample)]
    rows = []
    for rec in sel:
        r = recompute(rec, plans)
        rows.append(r)
        print(f"{r['study']:6s} {r['arch']:5s} {r['method']:6s} {r['criterion']:13s} {r['path']} n_D={r['n_D']:2d}  stored {r['stored_node_error']:.9f}  recomputed {r['recomputed_node_error']:.9f}  max|dMember| {r['max_abs_member_diff']:.2e} K ({r['max_rel_member_diff']:.1e} rel)  ({r['seconds']} s)", flush=True)
    for r in rows:
        r["within_tolerance_members"] = bool(np.all(np.abs(np.asarray(r["recomputed_members"]) - np.asarray(r["stored_members"])) <= 1e-6 + a.rtol * np.abs(r["stored_members"])))
    ok = all(r["within_tolerance_members"] for r in rows)
    worst = max(r["max_rel_member_diff"] for r in rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(rtol=a.rtol, worst_member_rel_diff=worst, n_nodes=len(rows), all_within_tolerance=ok, nodes=rows), indent=1))
    print(f"\n{len(rows)} nodes, largest relative member difference {worst:.2e} (tolerance {a.rtol:g}): {'OK' if ok else 'EXCEEDED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
