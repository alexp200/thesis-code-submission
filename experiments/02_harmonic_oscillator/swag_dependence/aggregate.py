"""Moments of the SWAG comparison and the DE reference of the learning-rate study.

  python -u aggregate.py    after `run_sweep.py --stage comparison`: SWAG and MultiSWAG moments at the learning rate of the comparisons (comparison/{model}_{dataset}.npz)
                            and the failure counts of these runs (comparison/comparison_summary.json)
"""

from __future__ import annotations

import json

import swag_dep_lib as L

import numpy as np

HERE = L.HERE
COMP = HERE / "comparison"
ens_lib = L.ens_lib


def cfg(model, ds, k, lr):
    """Cached result of one configuration (None if it was not run)."""
    p = L.CACHE / (L.stem(model, ds, k, lr) + ".json")
    return json.loads(p.read_text()) if p.exists() else None


def de_reference(model, ds):
    """H_q, H_p of the deep ensemble (all retained members) and its size."""
    z = np.load(ens_lib.RESULT_DIR / f"methods_{model}_{ds}.npz")
    meta = json.loads((ens_lib.RESULT_DIR / f"methods_{model}_{ds}.json").read_text())
    t = z["t"]
    return {"H_q": float(np.mean(ens_lib.envelope(t, z["DE__std"][:, 0])[1])), "H_p": float(np.mean(ens_lib.envelope(t, z["DE__std"][:, 1])[1])),
            "n_members": len(meta["filter"]["kept"])}


def pooled(mcs, ns):
    """Mean and variance of the pooled samples of several members (law of total variance with the sample counts as weights)."""
    N = float(sum(ns))
    mean = sum(n * m for n, m in zip(ns, mcs["mean"])) / N
    var = sum(n * (v + (m - mean) ** 2) for n, m, v in zip(ns, mcs["mean"], mcs["var"])) / N
    return mean, var


def run_comparison():
    """SWAG (member 0) and MultiSWAG (all retained members pooled) moments at the learning rate of the comparisons."""
    lr = ens_lib.SWAG_LR
    COMP.mkdir(exist_ok=True)
    summary = {}
    for model in L.MODELS:
        for ds in L.DATASETS:
            kept = L.kept_members(model, ds)
            cfgs = [cfg(model, ds, k, lr) for k in kept]
            arr = [np.load(L.CACHE / (L.stem(model, ds, k, lr) + ".npz")) for k, c in zip(kept, cfgs) if c["n_kept"] > 0]
            ns = [c["n_kept"] for c in cfgs if c["n_kept"] > 0]
            out = {"t": np.load(ens_lib.RESULT_DIR / f"methods_{model}_{ds}.npz")["t"]}
            out["SWAG__mean"], out["SWAG__std"] = arr[0]["mean"], np.sqrt(arr[0]["var"])
            m, v = pooled({"mean": [a["mean"] for a in arr], "var": [a["var"] for a in arr]}, ns)
            out["MultiSWAG__mean"], out["MultiSWAG__std"] = m, np.sqrt(v)
            np.savez(COMP / f"{model}_{ds}.npz", **out)
            summary[f"{model}/{ds}"] = {"model": model, "dataset": ds, "lr": lr, "members": kept, "samples_per_member": L.N_SAMPLES,
                                        "frac_failed_per_member": [c["frac_failed"] for c in cfgs]}
    (COMP / "comparison_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for key, info in summary.items():
        print(key, "lr", info["lr"], "| failed fraction", max(info["frac_failed_per_member"]))


if __name__ == "__main__":
    run_comparison()
