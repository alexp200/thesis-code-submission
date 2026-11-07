"""Level 2 (harmonic oscillator): recompute the long-horizon moments and the short-horizon data from the stored ensembles and compare them with the frozen results.

Stage `long`: `long_horizon/compute_methods.py` is run for one (model class, dataset) with an EMPTY result directory (HO_RESULT_DIR), i.e. the linearised Laplace fits of all retained members
(GGN, marginal-likelihood calibration, pushforward), the MoLA mixture, the deep-ensemble moments and, for the sPHNN, the orbit frequencies of the members are recomputed. The new
methods_*.npz / .json are compared with long_horizon/results/.
Stage `short`: `short_horizon/compute_short_rollout.py` is run with an EMPTY data directory (HO_SHORT_DATA_DIR): LA, MoLA, DE, SWAG (mini-batch SGD continuation, learning rate
ens_lib.SWAG_LR) and MultiSWAG on t in [0, 10] and the vector-field uncertainty on the 51 x 31 grid are recomputed and compared with short_horizon/data/.
Both start from the checkpoints in ensembles/checkpoints/ and the self-generated training data; no training is performed.

    python tools/recompute_ho.py                                  # sPHNN, left_half, both stages (about 4 min)
    python tools/recompute_ho.py --stage short --model-class NODE --dataset lower_half
    python tools/recompute_ho.py --all                            # four combinations, both stages (about 20 min)

Result: for every array the maximum absolute difference, relative to the largest absolute value of the frozen array.  Acceptance: relative difference <= --rtol (default 1e-6).
On the tested machine (arm64 macOS, the machine that also produced the frozen results) the recomputation is bit-identical up to the last bit of single entries; other CPUs may differ in the last bits.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HO = ROOT / "experiments/02_harmonic_oscillator"


def compare_arrays(a, b) -> tuple[float, dict]:
    """Worst relative difference of two .npz files and the per-array record."""
    worst, rows = 0.0, {}
    for k in sorted(a.files):
        x, y = a[k], b[k]
        if x.dtype.kind != "f":
            rows[k] = dict(equal=bool(np.array_equal(x, y)))
            continue
        scale = float(np.nanmax(np.abs(x))) or 1.0
        d = float(np.nanmax(np.abs(x - y)))
        rows[k] = dict(max_abs_diff=d, scale=scale, rel=d / scale)
        worst = max(worst, d / scale)
    return worst, rows


def run_script(script: Path, args: list[str], env_extra: dict, log: Path) -> int:
    """Run a script of the harmonic-oscillator experiment in its own process; returns the exit code."""
    env = dict(os.environ, PYTHONPATH=str(ROOT), JAX_PLATFORMS="cpu", MPLBACKEND="Agg", **env_extra)
    with open(log, "w") as f:
        return subprocess.run([sys.executable, "-u", script.name, *args], cwd=script.parent, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def run_long(mc: str, ds: str, work: Path, rtol: float) -> dict:
    """Stage 'long': recompute the long-horizon moments and compare them with the stored ones."""
    out = work / f"long_{mc}_{ds}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    t0 = time.time()
    if run_script(HO / "long_horizon/compute_methods.py", ["--model-class", mc, "--dataset", ds], dict(HO_RESULT_DIR=str(out)), out / "log.txt"):
        return dict(stage="long", model_class=mc, dataset=ds, ok=False, error=f"compute_methods.py failed, see {out / 'log.txt'}")
    worst, rows = compare_arrays(np.load(HO / f"long_horizon/results/methods_{mc}_{ds}.npz"), np.load(out / f"methods_{mc}_{ds}.npz"))
    ja, jb = (json.loads((d / f"methods_{mc}_{ds}.json").read_text()) for d in (HO / "long_horizon/results", out))
    om = float(np.max(np.abs(np.asarray(ja.get("Omega_MAP", [0.0])) - np.asarray(jb.get("Omega_MAP", [0.0])))))      # sPHNN only
    kept_equal = ja["filter"]["kept"] == jb["filter"]["kept"]
    la = max(abs(c["prior_precision"] - d["prior_precision"]) / abs(c["prior_precision"]) for c, d in zip(ja["la_components"], jb["la_components"]))
    ok = worst <= rtol and om <= rtol and kept_equal and la <= rtol
    return dict(stage="long", model_class=mc, dataset=ds, ok=bool(ok), worst_relative_array_difference=worst, max_abs_omega_difference=om,
                la_prior_precision_max_rel_difference=la, retained_members_equal=kept_equal, n_arrays=len(rows), seconds=round(time.time() - t0), arrays=rows)


def run_short(mc: str, ds: str, work: Path, rtol: float) -> dict:
    """Stage 'short': recompute the short-horizon data and compare them with the stored ones."""
    out = work / f"short_{mc}_{ds}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    t0 = time.time()
    if run_script(HO / "short_horizon/compute_short_rollout.py", ["--model-class", mc, "--dataset", ds], dict(HO_SHORT_DATA_DIR=str(out)), out / "log.txt"):
        return dict(stage="short", model_class=mc, dataset=ds, ok=False, error=f"compute_short_rollout.py failed, see {out / 'log.txt'}")
    worst, rows = compare_arrays(np.load(HO / f"short_horizon/data/{mc}_{ds}.npz"), np.load(out / f"{mc}_{ds}.npz"))
    return dict(stage="short", model_class=mc, dataset=ds, ok=bool(worst <= rtol), worst_relative_array_difference=worst, n_arrays=len(rows), seconds=round(time.time() - t0), arrays=rows)


def main():
    """Run the selected stages for the selected model classes and data sets."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="both", choices=["long", "short", "both"])
    ap.add_argument("--model-class", default="sPHNN", choices=["NODE", "sPHNN"])
    ap.add_argument("--dataset", default="left_half", choices=["lower_half", "left_half"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--work", type=Path, default=ROOT / "regenerated/ho_level2")
    ap.add_argument("--out", type=Path, default=ROOT / "regenerated/level2_ho.json")
    a = ap.parse_args()
    combos = [(m, d) for m in ("NODE", "sPHNN") for d in ("lower_half", "left_half")] if a.all else [(a.model_class, a.dataset)]
    stages = {"long": run_long, "short": run_short}
    res = []
    for mc, ds in combos:
        for name, fn in stages.items():
            if a.stage not in (name, "both"):
                continue
            r = fn(mc, ds, a.work, a.rtol)
            res.append(r)
            print(f"{name:5s} {mc:6s} {ds:10s} {'OK' if r['ok'] else 'FAILED'}  "
                  + (f"worst relative array difference {r['worst_relative_array_difference']:.2e}, {r['n_arrays']} arrays, {r['seconds']} s" if "error" not in r else r["error"]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(rtol=a.rtol, results=res), indent=1))
    return 0 if all(r["ok"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
