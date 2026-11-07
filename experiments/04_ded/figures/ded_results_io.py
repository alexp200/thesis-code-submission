"""Tidy loader of the DED run records: one row per (package, model, start, stream, N) with the member errors of the K = 3 ensemble.
Sources (under runs/):
  A   ded_start_study/run/main/paths/{model}/startXX/{stream}/iterNN.json : the five first starts [12, 4, 24, 22, 10], LA / DE / MoLA (MGE) and Random 0..2
  T, G6   ded_final_extension/run/paths/{package}/{model}/startXX/{stream}/iterNN.json : T = Random 3..5 of the first five starts, G6 = the five further starts (LA / DE / MoLA and six Random paths)
iterNN.json holds the data set of size N = NN + 1; evaluation = all 25 operating trajectories INCLUDING the training trajectories (no held-out set); node error = median of the member RMSEs (status passed)."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
FE = REPO / "runs"
PAT = re.compile(r"paths/(?:(?P<pkg>[A-Z0-9]+)/)?(?P<model>sPHNN|NODE)/start(?P<start>\d+)/(?P<stream>[^/]+)/iter(?P<it>\d+)\.json$")


def _row(pkg, model, start, stream, j):
    """One record of the loader table from an iterNN.json, or None if it has no passed members."""
    ev = j.get("evaluation") or {}
    mem = [m["rmse_K"] for m in sorted(ev.get("members", []), key=lambda m: m["member_index"]) if m.get("status") == "passed"]
    if not mem:
        return None
    pts = [m["per_trajectory_rmse_K"] for m in ev["members"] if m.get("status") == "passed" and m.get("per_trajectory_rmse_K")]
    return dict(pkg=pkg, model=model, start=int(start), stream=stream, N=int(j["dataset_size"]), train_ids=tuple(j["train_ids"]), members=tuple(mem), err=float(np.median(mem)), n_members=len(mem), ens_mean_rmse=j.get("ensemble_mean_rmse_K"),
                pt=tuple(np.median(np.array(pts, float), axis=0)) if pts else None)          # pt = member-median RMSE [K] per simulation ID 0..24 (grid cell)


def load_new():
    """Table of all records of the start study and the extension."""
    rows = []
    for root, pkgdefault in ((FE / "ded_start_study/run/main/paths", "A"), (FE / "ded_final_extension/run/paths", None)):
        for f in sorted(root.rglob("iter??.json")):
            m = PAT.search(str(f))
            if not m:
                continue
            r = _row(m["pkg"] or pkgdefault, m["model"], m["start"], m["stream"], json.loads(f.read_text()))
            if r:
                rows.append(r)
    return pd.DataFrame(rows)
