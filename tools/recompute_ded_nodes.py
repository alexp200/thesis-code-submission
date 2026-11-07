"""Level 2 (directed energy deposition): recompute the reporting errors of stored AL-path nodes from checkpoint + POD basis + raw data and compare them with the stored node records.

For every selected reporting node (runs/ded_start_study/run/main/shared/<id>/ or runs/ded_final_extension/run/shared/<id>/)
  1. reads train.json (model class, training IDs, stored member RMSEs [K] over all 25 simulations),
  2. loads the node-specific POD basis pod.npz (restored from large_assets/ded_pod_bases.tar.gz on demand) and encodes the training and the 25 evaluation trajectories (data/directed_energy_deposition/max_simulation/dataset.nc),
  3. loads the K = 3 member checkpoint model.eqx, applies the production training-RMSE filter and evaluates the members exactly as the production code does (start_study/study_numerical.py -> repeat_adapter.filter_and_report;
     global squared error over the 25 simulations x times x 1836 nodes after decoding to Kelvin),
  4. compares member RMSEs and the node error (median of the member RMSEs) with the stored values.
With --refit-pod the POD basis is additionally recomputed from the raw data and the training IDs and compared with the stored basis (after aligning the SVD sign convention of every mode, see pod_refit_comparison).

    python tools/recompute_ded_nodes.py                      # 6 nodes (both architectures, sizes 1-5), about 2-4 min
    python tools/recompute_ded_nodes.py --sample 20 --refit-pod
    python tools/recompute_ded_nodes.py --record runs/ded_start_study/run/main/shared/<id>/train.json

The production modules insist on the validated single-native-thread float64 CPU configuration; this script sets it (OMP/MKL/OPENBLAS threads = 1, serial XLA codegen) before JAX is imported.
The original runs were produced on x86-64 Linux CPUs; other CPUs give small floating-point differences.  A node passes if every member RMSE agrees with the stored value within
1e-6 K + rtol x stored value (default rtol = 1e-3; the thesis displays node errors with two decimals, i.e. a relative resolution of about 1e-3 to 1e-2).  No training is performed.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_parallel_codegen_split_count=1").strip()
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHARED = [ROOT / "runs/ded_start_study/run/main/shared", ROOT / "runs/ded_final_extension/run/shared"]
ARCHIVE = ROOT / "large_assets/ded_pod_bases.tar.gz"
DATA = ROOT / "data/directed_energy_deposition/max_simulation/dataset.nc"
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/04_ded/al_runs/start_study"), str(ROOT / "experiments/shared/al_path_repeats")]


def reporting_nodes() -> list[Path]:
    """Stored reporting nodes (train.json) with member errors."""
    out = []
    for base in SHARED:
        for f in sorted(base.glob("*/train.json")):
            j = json.loads(f.read_text())
            if j["request"]["role"] == "reporting" and j.get("evaluation") and j["evaluation"].get("members"):
                out.append(f)
    return out


def pod_for(node: Path, scratch: Path) -> Path:
    """pod.npz of a node: from the restored tree if present, else extracted from the large-asset archive into the scratch directory"""
    local = node.with_name("pod.npz")
    if local.exists():
        return local
    rel = str(local.relative_to(ROOT))
    target = scratch / rel
    if not target.exists():
        if not ARCHIVE.exists():
            sys.exit(f"{rel} is missing and {ARCHIVE.relative_to(ROOT)} is not present (large assets are delivered separately; unpack large_assets/ded_pod_bases.tar.gz, see docs/REPRODUCTION.md)")
        with tarfile.open(ARCHIVE, "r:gz") as tf:
            member = tf.getmember(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, target.open("wb") as dst:
                dst.write(src.read())
    return target


def pod_refit_comparison(fresh, pod: Path) -> dict:
    """compares a freshly fitted POD basis with the stored one. The SVD fixes the sign of every mode only up to the platform's convention (a different CPU / LAPACK returns some modes with the opposite sign, and x_eq follows
    the signs of modes_T), so the comparison aligns the sign of every column first and reports the remaining difference relative to the largest entry of the column."""
    with np.load(pod, allow_pickle=False) as saved:
        T, Q, X = (np.asarray(a) for a in fresh)
        sT, sQ = saved["modes_T"], saved["modes_q"]
        sign = np.sign((T * sT).sum(0))
        sign[sign == 0] = 1.0
        signq = np.sign((Q * sQ).sum(0))
        signq[signq == 0] = 1.0
        rel = lambda a, b: float((np.abs(a - b).max(0) / np.abs(b).max(0)).max())
        return dict(refit_sign_flipped_columns=int((sign < 0).sum()), refit_modes_T_max_rel_diff=rel(T * sign, sT), refit_modes_q_max_rel_diff=rel(Q * signq, sQ),
                    refit_x_eq_max_rel_diff=float(np.abs(X * sign - saved["x_eq"]).max() / np.abs(saved["x_eq"]).max()), refit_signs_of_T_and_q_agree=bool((sign == signq).all()))


def recompute(node: Path, scratch: Path, refit: bool) -> dict:
    """Re-evaluate one stored node from its checkpoint and POD basis and compare with the stored member errors."""
    import equinox as eqx
    import study_numerical as SN
    j = json.loads(node.read_text())
    request = j["request"]
    t0 = time.time()
    adapter, base = SN.modules(request)
    try:
        pod = pod_for(node, scratch)
        data = SN.prepare(base, request["train_ids"], pod)
        ensemble = eqx.tree_deserialise_leaves(node.with_name("model.eqx"), adapter.skeleton(request["ensemble_size"]))
        metrics, _ = adapter.filter_and_report(ensemble, data)
        row = dict(refit_modes_T_max_rel_diff=None)
        if refit:
            from experiments.shared.cluster_recovery.pod_cache import load_or_fit
            t, ds = base.load_dataset()
            fresh = load_or_fit(ds, request["train_ids"], base.core.N_MODES, fitter=base.core.fit_pods, dataset_path=base.original.DATA_PATH, cache_dir=scratch / "pod_fit",
                                implementation_paths=(ROOT / "src/preprocessing.py", SN.PRIMARY / "core/ded_data.py"))
            row.update(pod_refit_comparison(fresh, pod))
            ds.close()
    finally:
        adapter.close()
    stored = np.asarray([m["rmse_K"] for m in sorted(j["evaluation"]["members"], key=lambda m: m["member_index"]) if m["status"] == "passed"])
    mem = np.asarray([m["rmse_K"] for m in sorted(metrics["members"], key=lambda m: m["member_index"]) if m["status"] == "passed"])
    return dict(node=str(node.relative_to(ROOT)), model=request["model_class"], start=request["start"], n_D=request["dataset_size"], train_ids=request["train_ids"], stored_members=stored.tolist(), recomputed_members=mem.tolist(),
                stored_node_error=float(np.median(stored)), recomputed_node_error=float(np.median(mem)), max_abs_member_diff=float(np.abs(mem - stored).max()), max_rel_member_diff=float((np.abs(mem - stored) / np.abs(stored)).max()), within_tolerance_members=None, abs_node_diff=float(abs(np.median(mem) - np.median(stored))),
                dropped_stored=j["dropped_indices"], seconds=round(time.time() - t0, 1), **row)


def main():
    """Recompute a sample of stored nodes (or the given records)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=6)
    ap.add_argument("--record", action="append", default=[], help="explicit train.json path(s) relative to the submission root")
    ap.add_argument("--refit-pod", action="store_true")
    ap.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance of a member RMSE (plus 1e-6 K absolute)")
    ap.add_argument("--out", type=Path, default=ROOT / "regenerated/level2_ded_nodes.json")
    a = ap.parse_args()
    if not DATA.exists():
        sys.exit("data/directed_energy_deposition/max_simulation/dataset.nc is missing: Level 2 needs the supervisor-provided raw data (docs/DATA_ACCESS.md).")
    scratch = ROOT / "regenerated/restored_pods"
    if a.record:
        sel = [ROOT / r for r in a.record]
    else:
        allr = reporting_nodes()
        sel = [allr[int(i)] for i in np.linspace(0, len(allr) - 1, a.sample)]
    rows = []
    for node in sel:
        r = recompute(node, scratch, a.refit_pod)
        rows.append(r)
        extra = "" if r["refit_modes_T_max_rel_diff"] is None else (f"  POD refit (sign-aligned, relative per column): modes_T {r['refit_modes_T_max_rel_diff']:.1e}, modes_q {r['refit_modes_q_max_rel_diff']:.1e}, "
                                                                     f"x_eq {r['refit_x_eq_max_rel_diff']:.1e}; {r['refit_sign_flipped_columns']} sign-flipped columns")
        print(f"{r['model']:5s} start {r['start']:2d} n_D={r['n_D']}  stored {r['stored_node_error']:.6f} K  recomputed {r['recomputed_node_error']:.6f} K  max|dMember| {r['max_abs_member_diff']:.2e} K ({r['max_rel_member_diff']:.1e} rel)  ({r['seconds']} s){extra}", flush=True)
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
