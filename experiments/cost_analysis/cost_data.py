"""Compute-cost data of the AL experiments (Chicken and DED) from the stored cluster records.  One row per (dataset, architecture, UQ method, ensemble size N):
  train_s   wall time of the training of the K = 3 ensemble of size N (train_seconds of the record; independent of the UQ method)
  acq_s     wall time of the acquisition that scores the pool with this ensemble (UQ fit + scoring + selection):  Chicken: selections/iterNN.json wall_seconds;  DED: acquisition cache selection.json wall_seconds
  extra_train_s  DED, LA only: training time of the separate K = 1 acquisition model (Chicken: LA uses member 0 of the reporting ensemble, no extra training)
  cpus      number of CPUs of the job (DED: length of the recorded cpu_affinity; Chicken: 1-CPU tasks)
Sources: thermal: runs/03_AL_chicken/warmstart_path_AL/{small, small_extra5/run} (iterNN_train.json, selections/iterNN.json);  DED: runs/ded_start_study/run/main + runs/ded_final_extension/run (iterNN.json, acquisition cache selection.json).
Records of unsuccessful attempts are not stored; timings are wall times on a shared cluster (see the caveats in the figure caption / README).  Output: results/cost_data.csv.  Usage: python cost_data.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
FE = REPO / "runs"


def chicken():
    """Training and acquisition times of the thermal AL paths (LA, DE, MoLA)."""
    rows = []
    for root in (FE / "03_AL_chicken/warmstart_path_AL/small/results", FE / "03_AL_chicken/warmstart_path_AL/small_extra5/run/results"):
        for sel in sorted(root.glob("*/*/*/path??/selections/iter??.json")):
            arch, method, crit, pdir = sel.parts[-6:-2]; it = sel.stem[-2:]; base = sel.parent.parent
            tr = base / f"iter{it}_train.json"
            if method == "random" or not tr.exists():
                continue
            s = json.loads(sel.read_text()); t = json.loads(tr.read_text())
            rows.append(dict(dataset="Chicken", arch=arch, method=method, crit=crit, path=pdir, N=int(t["dataset_size"]), train_s=t["train_seconds"], acq_s=s["wall_seconds"], cpus=1, host=t.get("hostname")))
    return pd.DataFrame(rows)


def ded():
    """Training and acquisition times of the DED paths, including the separate single-member acquisition model of LA."""
    rows = []
    for root in (FE / "ded_start_study/run/main/paths", FE / "ded_final_extension/run/paths"):
        for f in sorted(root.rglob("iter??.json")):
            parts = f.parts
            try:
                arch = next(p for p in parts if p in ("sPHNN", "NODE")); stream = f.parent.name
            except StopIteration:
                continue
            if stream not in ("LA", "DE", "MoLA"):
                continue
            j = json.loads(f.read_text()); acq = j.get("acquisition") or {}; c = acq.get("cache", {}).get("path")
            # the acquisition stored in iterNN.json scored the ensemble of size N-1 and created this data set: attach it to that ensemble (size N-1)
            if c is None or j["dataset_size"] < 2:
                continue
            sel = json.loads((f.parent / c).resolve().read_text()); ex = sel.get("execution", {})
            tmm = sel.get("timings") or {}; cpu_sec = sum(x.get("cpu_seconds", 0) for x in (tmm.get("group", {}) or {}).get("components", []))
            rows.append(dict(dataset="DED", arch=arch, method=stream, crit="sum_marginal", path=str(f.parent.relative_to(root)), N=int(j["dataset_size"]) - 1, acq_s=sel["wall_seconds"], acq_cpu_s=cpu_sec or np.nan, cpus=len(ex.get("cpu_affinity", [])) or np.nan, host=ex.get("hostname"), peak_rss_gb=max([x.get("peak_rss_bytes", 0) for x in (tmm.get("group", {}) or {}).get("components", [])] or [0]) / 2 ** 30 or np.nan))
    a = pd.DataFrame(rows).drop_duplicates(["arch", "method", "path", "N"])
    tr = []
    for root in (FE / "ded_start_study/run/main/paths", FE / "ded_final_extension/run/paths"):
        for f in sorted(root.rglob("iter??.json")):
            arch = next((p for p in f.parts if p in ("sPHNN", "NODE")), None); stream = f.parent.name
            if arch is None or stream not in ("LA", "DE", "MoLA"):
                continue
            j = json.loads(f.read_text()); tr.append(dict(arch=arch, method=stream, path=str(f.parent.relative_to(root)), N=int(j["dataset_size"]), train_s=j["train_seconds"], train_cpus=len(j.get("execution", {}).get("cpu_affinity", [])) or np.nan))
    out = a.merge(pd.DataFrame(tr).drop_duplicates(["arch", "method", "path", "N"]), on=["arch", "method", "path", "N"], how="left")
    k1 = []                                                                                    # DED: LA uses its own freshly trained K = 1 acquisition model (iterNN_acq.json): its training time is an extra cost of LA
    for root in (FE / "ded_start_study/run/main/paths", FE / "ded_final_extension/run/paths"):
        for f in sorted(root.rglob("iter??_acq.json")):
            arch = next((p for p in f.parts if p in ("sPHNN", "NODE")), None)
            if arch is None:
                continue
            j = json.loads(f.read_text()); k1.append(dict(arch=arch, method="LA", path=str(f.parent.relative_to(root)), N=int(j["dataset_size"]), extra_train_s=j["train_seconds"]))
    out = out.merge(pd.DataFrame(k1).drop_duplicates(["arch", "method", "path", "N"]), on=["arch", "method", "path", "N"], how="left")
    out["extra_train_s"] = out.extra_train_s.fillna(0.0); return out


if __name__ == "__main__":
    c, d = chicken(), ded(); df = pd.concat([c, d], ignore_index=True); df.to_csv(HERE / "results/cost_data.csv", index=False); pd.set_option("display.width", 250)
    print(len(c), len(d)); print(df.groupby(["dataset", "arch", "method", "N"]).agg(n=("acq_s", "size"), train_min=("train_s", lambda x: np.nanmedian(x) / 60), acq_min=("acq_s", lambda x: np.nanmedian(x) / 60), cpus=("cpus", "median")).round(2).to_string())
