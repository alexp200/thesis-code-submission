"""Short-rollout data of the harmonic-oscillator figures (Figures 4.3, 4.4, B.4, B.5 and the vector-field panels of Figure 4.5) for one (model class, data set).

Models: the K = 10 ensembles of ../ensembles (checkpoints; convergence filter recorded in ../long_horizon/results/methods_*.json); the sPHNN is conservative (R and G absent).
Methods (no sampling from any Laplace posterior):
  LA         linearized Laplace of member 0
  MoLA       uniform mixture of the linearized Laplace components of the retained members
  DE         the retained MAP members
  SWAG       SWAG around member 0 (ens_lib.SWAG_KW: mini-batches of 4 trajectories, continuation learning rate ens_lib.SWAG_LR), 64 samples
  MultiSWAG  pooled SWAG samples of the retained members

Stored for every method: mean and std of (q, p) for the representative initial condition on t in [0, 10], and mean and norm of the componentwise std of the vector field
f_theta(q, p) on the grid q in [-10, 10] (51 points) x p in [-8, 8] (31 points).
Mixture rule (all methods, uniform weights): mean = mean_k mean_k, var = mean_k var_k + mean_k (mean_k - mean)^2 with var_k = 0 for point estimates (population statistics, ddof = 0).

Usage: python -u compute_short_rollout.py --model-class NODE --dataset lower_half
Output: data/{model}_{dataset}.npz (per-member caches in data/cache/)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ensembles"))

import ensemble_lib as ens_lib  # (enables float64 before any model construction)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np

DATA_DIR = Path(os.environ["HO_SHORT_DATA_DIR"]) if os.environ.get("HO_SHORT_DATA_DIR") else HERE / "data"      # a Level-2 recomputation can write to another directory
CACHE_DIR = DATA_DIR / "cache"
METHODS = ("LA", "MoLA", "DE", "SWAG", "MultiSWAG")
Q_LIM, P_LIM, NQ, NP = 10.0, 8.0, 51, 31


def log(msg: str):
    """Print a message with a time stamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def field_posterior(la, points):
    """Mean and per-output std of the linearized-Laplace vector field f_theta at `points`: Var[f_i(x)] = sum_k w_k (J_f(x) u_k)_i^2 (linearized predictive of Ott et al.)."""
    directions, weights = la._posterior_directions_and_weights()
    jacobians = jax.vmap(lambda state: jax.jacobian(la.f_field)(la.p_map, state))(points)
    projected = jnp.einsum("nod,dk->nok", jacobians, directions)
    variance = jnp.einsum("nok,k->no", projected**2, weights)
    mean = jax.vmap(lambda state: la.f_field(la.p_map, state))(points)
    return np.asarray(mean), np.asarray(jnp.sqrt(jnp.maximum(variance, 0.0)))


@eqx.filter_jit
def roll(model, ts, x0):
    """Rollout of one model on the grid ts."""
    return model(ts, x0)


@eqx.filter_jit
def field_at(model, points):
    """Vector field of one model at the given states."""
    return jax.vmap(lambda x: ens_lib.vector_field(model, x))(points)


def mixture(means, stds):
    """(K, ...) component means / stds -> mean, std of the uniform mixture."""
    means, stds = np.asarray(means), np.asarray(stds)
    mean = means.mean(axis=0)
    var = (stds**2).mean(axis=0) + ((means - mean) ** 2).mean(axis=0)
    return mean, np.sqrt(var)


def main():
    """Command-line entry: compute and store the short-rollout data of one model class and data set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-class", choices=ens_lib.MODEL_CLASSES, required=True)
    parser.add_argument("--dataset", choices=ens_lib.DATASETS, required=True)
    args = parser.parse_args()
    mc, dsname = args.model_class, args.dataset
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ds = ens_lib.build_dataset(dsname)
    x0 = ds["representative_x0"]
    t_eval, true_eval = ens_lib.base.representative_evaluation(ds)
    q, p = jnp.linspace(-Q_LIM, Q_LIM, NQ), jnp.linspace(-P_LIM, P_LIM, NP)
    Q, P = jnp.meshgrid(q, p)
    points = jnp.stack((Q.reshape(-1), P.reshape(-1)), axis=-1)
    members = ens_lib.load_members(mc, dsname)
    kept = json.loads((ens_lib.RESULT_DIR / f"methods_{mc}_{dsname}.json").read_text())["filter"]["kept"]
    swag_lr = ens_lib.SWAG_LR
    log(f"{mc}/{dsname}: kept members {kept}, SWAG learning rate {swag_lr:g}, x0={np.asarray(x0)}")
    if mc == "sPHNN":
        for k in kept:
            func = ens_lib.callable_member(mc, members[k]).func
            assert func.dissipation_matrix is None and func.input_matrix is None, "R and G must be structurally absent"

    # ---- LA / MoLA components and DE members ------------------------------------
    comp = {}
    for k in kept:
        path = CACHE_DIR / f"{mc}_{dsname}_la_k{k}.npz"
        if path.exists():
            comp[k] = dict(np.load(path))
            log(f"  [LA k={k}] cache")
            continue
        t0 = time.time()
        la = ens_lib.laplace_class(mc)(members[k], ds["t_train"], ds["x0s"], ds["ys_noisy"])
        la.fit_laplace_mll()
        f_mean, f_std = field_posterior(la, points)
        model = ens_lib.callable_member(mc, members[k])
        comp[k] = {
            "mu": np.asarray(la.mean_linearized(t_eval, x0)), "std": np.sqrt(np.maximum(np.asarray(ens_lib.lla_variance(la, t_eval, x0)), 0.0)),
            "field_mean": f_mean, "field_std": f_std, "direct_traj": np.asarray(roll(model, t_eval, x0)), "direct_field": np.asarray(field_at(model, points)),
        }
        np.savez(path, **comp[k])
        log(f"  [LA k={k}] {time.time() - t0:.0f}s")

    # ---- SWAG samples (fit / sampling keys 20000 + k / 30000 + k as in swag_dependence) ----
    swag = {}
    for k in kept:
        path = CACHE_DIR / f"{mc}_{dsname}_swag_k{k}_lr{swag_lr:g}.npz"
        if path.exists():
            swag[k] = dict(np.load(path))
            log(f"  [SWAG k={k}] cache")
            continue
        t0 = time.time()
        sw = ens_lib.swag_class(mc)(members[k], ds["t_train"], ds["x0s"], ds["ys_noisy"])
        kw = dict(ens_lib.SWAG_KW)
        kw["swag_lr"] = swag_lr
        sw.fit_swag(l2=ens_lib.SWAG_L2, key=jrd.key(20_000 + k), log_every=kw["swag_steps"], **kw)
        thetas = sw.sample_params(jrd.key(30_000 + k), n_samples=ens_lib.SWAG_MC_SAMPLES, eps=ens_lib.SWAG_SAMPLE_EPS)
        trajs, fields = [], []
        for i in range(thetas.shape[0]):
            model = sw.rebuild(thetas[i])
            x = np.asarray(roll(model, t_eval, x0))
            if np.all(np.isfinite(x)) and np.max(np.abs(x)) < ens_lib.SWAG_MAX_ABS_VALUE:
                trajs.append(x)
                fields.append(np.asarray(field_at(model, points)))
        swag[k] = {"traj": np.stack(trajs), "field": np.stack(fields)}
        np.savez(path, **swag[k])
        log(f"  [SWAG k={k}] {time.time() - t0:.0f}s kept {len(trajs)}/{thetas.shape[0]}")

    # ---- aggregate per method ------------------------------------------------------
    out = {"t": np.asarray(t_eval), "truth": np.asarray(true_eval), "obs_t": np.asarray(ds["t_train"]), "obs_y": np.asarray(ds["ys_noisy"]), "x0": np.asarray(x0),
           "train_t_end": np.asarray(ens_lib.base.training_t_end()), "grid_q": np.asarray(Q), "grid_p": np.asarray(P)}

    def put(name, traj_mean, traj_std, f_mean, f_std):
        out[f"{name}__mean"] = traj_mean
        out[f"{name}__std"] = traj_std
        out[f"{name}__field_mean"] = f_mean
        out[f"{name}__field_unc"] = np.linalg.norm(f_std, axis=-1).reshape(Q.shape)

    k0 = kept[0]
    put("LA", comp[k0]["mu"], comp[k0]["std"], comp[k0]["field_mean"], comp[k0]["field_std"])
    mu_m, std_m = mixture(np.stack([comp[k]["mu"] for k in kept]), np.stack([comp[k]["std"] for k in kept]))
    f_m, fs_m = mixture(np.stack([comp[k]["field_mean"] for k in kept]), np.stack([comp[k]["field_std"] for k in kept]))
    put("MoLA", mu_m, std_m, f_m, fs_m)
    trajs = np.stack([comp[k]["direct_traj"] for k in kept])
    fields = np.stack([comp[k]["direct_field"] for k in kept])
    put("DE", trajs.mean(axis=0), trajs.std(axis=0), fields.mean(axis=0), fields.std(axis=0))
    put("SWAG", swag[k0]["traj"].mean(axis=0), swag[k0]["traj"].std(axis=0), swag[k0]["field"].mean(axis=0), swag[k0]["field"].std(axis=0))
    pooled_t = np.concatenate([swag[k]["traj"] for k in kept])
    pooled_f = np.concatenate([swag[k]["field"] for k in kept])
    put("MultiSWAG", pooled_t.mean(axis=0), pooled_t.std(axis=0), pooled_f.mean(axis=0), pooled_f.std(axis=0))

    np.savez(DATA_DIR / f"{mc}_{dsname}.npz", **out)
    unc = {m: float(out[f"{m}__field_unc"].max()) for m in METHODS}
    log(f"saved data/{mc}_{dsname}.npz; max field uncertainty per method: {unc}")


if __name__ == "__main__":
    main()
