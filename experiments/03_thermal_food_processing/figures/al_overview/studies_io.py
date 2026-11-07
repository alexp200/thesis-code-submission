"""Loader of the AL run records of the thesis (long table: one row per architecture / method / criterion / path / n_D).

Studies loaded:
  warm_small_K3   original ten starts, n_D = 1..15, AL streams and one stored Random path per start, K = 3, AP15 test set
  warm_extra5_K3  five additional starts (745, 795, 861, 873, 905), n_D = 1..5, AL streams only, K = 3
Columns: study, arch, method (Random/LA/DE/MoLA), criterion, path, N, members (tuple of member RMSEs [K]), median, mean, sd (ddof 1), train_ids (tuple).
Records: runs/03_AL_chicken/warmstart_path_AL/{small/results, small_extra5/run/results}/{arch}/{method}/{criterion}/pathXX/iterNN.json (evaluation.members).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import export_paths as EP

FE = EP.RESULTS_ROOT.parent
WARM = {"warm_small_K3": FE / "03_AL_chicken/warmstart_path_AL/small/results", "warm_extra5_K3": FE / "03_AL_chicken/warmstart_path_AL/small_extra5/run/results"}


def _row(study, arch, method, crit, path, N, mem, ids):
    mem = np.asarray(mem, float)
    return dict(study=study, arch=arch, method=method, criterion=crit, path=path, N=N, members=tuple(mem), median=float(np.median(mem)), mean=float(mem.mean()), sd=float(mem.std(ddof=1)), train_ids=tuple(ids) if ids is not None else None)


def load_warm(study):
    """Rows of one study: member RMSEs on AP15 of every stored evaluation."""
    root, rows = WARM[study], []
    if not root.exists():
        return rows
    for f in sorted(root.glob("*/*/*/path??/iter??.json")):
        arch, method, crit, pdir = f.parts[-5:-1]
        d = json.loads(f.read_text()); ev = d.get("evaluation") or {}
        if not ev.get("members"):
            continue
        mem = [m["rmse_K"] for m in sorted(ev["members"], key=lambda m: m["member_index"])]
        rows.append(_row(study, arch, {"random": "Random"}.get(method, method), crit, int(pdir[4:]), d["dataset_size"], mem, d["train_ids"]))
    return rows


def load_all():
    """Rows of both studies."""
    rows = load_warm("warm_small_K3") + load_warm("warm_extra5_K3")
    return pd.DataFrame(rows)
