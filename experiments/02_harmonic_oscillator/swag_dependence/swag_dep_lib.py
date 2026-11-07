"""Shared code of the SWAG learning-rate study.

One configuration = (model class, data set, member k, SWAG continuation learning rate). It reuses the long-horizon pipeline unchanged: the retained MAP members of
long_horizon/compute_methods.py, the SWAG class and settings of ens_lib.SWAG_KW (only the learning rate varies), the PRNG keys (fit 20000 + k, sampling 30000 + k), the rollout grid (15 true
periods, 3000 points, dynax default solver and tolerances) and the bounded-sample filter (finite and max|x| < 1e6; solver exceptions count as failed). The rollout is wrapped in
eqx.filter_jit so that it compiles once per process.

Measured per configuration:
  H_q, H_p   horizon-averaged Std: mean over the 15 per-period maxima (windows of the true period) of the sample Std of q (p)
  n_kept     number of the 64 samples that are finite and bounded
and the sample mean and variance over t (arrays) for the curves of the comparison.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ensembles"))
import ensemble_lib as ens_lib  # (enables float64 before any model construction)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrd
import numpy as np

CACHE = HERE / "cache"
LR_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)  # fixed before the first run
N_MEMBERS = 4
N_SAMPLES = int(ens_lib.SWAG_MC_SAMPLES)
SNAPSHOT_FREQ = int(ens_lib.SWAG_KW["swag_freq"])
EPS = float(ens_lib.SWAG_SAMPLE_EPS)
DATASETS, MODELS = ens_lib.DATASETS, ens_lib.MODEL_CLASSES


def log(msg):
    """Print a message with a time stamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kept_members(model, dataset):
    """Members retained by the convergence filter of compute_methods.py."""
    meta = json.loads((ens_lib.RESULT_DIR / f"methods_{model}_{dataset}.json").read_text())
    return list(meta["filter"]["kept"])


def sweep_members(model, dataset):
    """The first N_MEMBERS converged members; not selected on any result."""
    return kept_members(model, dataset)[:N_MEMBERS]


def stem(model, dataset, k, lr):
    """Cache file name of one configuration."""
    return f"{model}_{dataset}_k{k}_lr{lr:g}_n{N_SAMPLES}_f{SNAPSHOT_FREQ}_eps{EPS:g}"


@eqx.filter_jit
def _roll(model, ts, x0):
    return model(ts, x0)


_MEMBER_CACHE: dict = {}


def _members(model, dataset):
    key = (model, dataset)
    if key not in _MEMBER_CACHE:
        _MEMBER_CACHE[key] = ens_lib.load_members(model, dataset)
    return _MEMBER_CACHE[key]


def run_config(model, dataset, k, lr, force=False):
    """Fit SWAG around member k with continuation learning rate lr, roll out the samples and cache H_q, H_p, the failure counts and the sample mean and variance."""
    name = stem(model, dataset, k, lr)
    path_json, path_npz = CACHE / (name + ".json"), CACHE / (name + ".npz")
    if path_json.exists() and not force:
        res = json.loads(path_json.read_text())
        if res["n_kept"] == 0 or path_npz.exists():
            return res
    ds = ens_lib.build_dataset(dataset)
    x0 = ds["representative_x0"]
    t_eval = jnp.linspace(0.0, ens_lib.horizon(), ens_lib.N_EVAL)
    t = np.asarray(t_eval)
    member = _members(model, dataset)[k]
    res = {"model": model, "dataset": dataset, "member": k, "lr": lr, "n_samples": N_SAMPLES, "fit_key": 20_000 + k, "sample_key": 30_000 + k}
    arrays = {}
    try:
        swag = ens_lib.swag_class(model)(member, ds["t_train"], ds["x0s"], ds["ys_noisy"])
        kw = dict(ens_lib.SWAG_KW)
        kw["swag_lr"], kw["swag_freq"] = lr, SNAPSHOT_FREQ
        swag.fit_swag(l2=ens_lib.SWAG_L2, key=jrd.key(20_000 + k), log_every=kw["swag_steps"], **kw)
        thetas = swag.sample_params(jrd.key(30_000 + k), n_samples=N_SAMPLES, eps=EPS)
        xs_kept, n_exc = [], 0
        for i in range(thetas.shape[0]):
            try:
                xs = np.asarray(_roll(swag.rebuild(thetas[i]), t_eval, x0))
                if np.all(np.isfinite(xs)) and np.max(np.abs(xs)) < ens_lib.SWAG_MAX_ABS_VALUE:
                    xs_kept.append(xs)
            except Exception:  # (diffrax max_steps / non-finite parameters)
                n_exc += 1
        nk = len(xs_kept)
        res.update({"n_kept": nk, "n_exceptions": n_exc, "frac_failed": 1.0 - nk / N_SAMPLES})
        if nk >= 1:
            X = np.stack(xs_kept)
            mean, var = X.mean(axis=0), X.var(axis=0)
            std = np.sqrt(var)
            arrays.update({"mean": mean, "var": var})
            res.update({"H_q": float(np.mean(ens_lib.envelope(t, std[:, 0])[1])), "H_p": float(np.mean(ens_lib.envelope(t, std[:, 1])[1]))})
        else:
            res.update({"H_q": float("nan"), "H_p": float("nan")})
    except Exception as exc:
        res.update({"error": type(exc).__name__, "n_kept": 0, "frac_failed": 1.0, "H_q": float("nan"), "H_p": float("nan")})
    CACHE.mkdir(parents=True, exist_ok=True)
    if arrays:
        np.savez_compressed(path_npz, **arrays)
    path_json.write_text(json.dumps(res, indent=1) + "\n")
    log(f"{name}: kept {res['n_kept']}/{N_SAMPLES}, H_q {res['H_q']:.4g}, H_p {res['H_p']:.4g}" + (f"  ERROR {res['error']}" if "error" in res else ""))
    return res
