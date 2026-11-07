"""Long-horizon predictive moments of LA, MoLA and DE for one (model class, data set).

All three start from the same K = 10 MAP members (train_ensembles.py) after the convergence filter:
  LA    linearized Laplace of member 0 (fit_laplace_mll, linearized pushforward)
  MoLA  uniform mixture of the linearized-Laplace components of all retained members (law of total variance)
  DE    the retained MAP members
Mean and standard deviation of q and p on the grid t in [0, 15 T_true] (population statistics, ddof = 0). For the sPHNN the orbit frequency of every retained member is stored
as well: Omega = 2 pi / T with T the first return of the MAP orbit to the Poincare section (Table B.10). SWAG and MultiSWAG: swag_dependence/.
No sampling from any Laplace posterior is done anywhere.

Usage: python -u compute_methods.py --model-class sPHNN --dataset lower_half
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ensembles"))
import ensemble_lib as ens_lib  # (enables float64 before any model construction)

import jax.numpy as jnp
import numpy as np

from src import ensembles as ens


def log(msg: str):
    """Print a message with a time stamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    """Command-line entry: compute and store the moments of LA, MoLA and DE for one model class and data set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-class", choices=ens_lib.MODEL_CLASSES, required=True)
    parser.add_argument("--dataset", choices=ens_lib.DATASETS, required=True)
    args = parser.parse_args()
    mc, dsname = args.model_class, args.dataset

    # ---- data, evaluation grid, trained members and the convergence filter ----------
    ds = ens_lib.build_dataset(dsname)
    x0 = ds["representative_x0"]
    t_eval = jnp.linspace(0.0, ens_lib.horizon(), ens_lib.N_EVAL)
    all_members = ens_lib.load_members(mc, dsname)
    cache_dir = ens_lib.RESULT_DIR / f"cache_{mc}_{dsname}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Members whose training RMSE against the noise-free targets exceeds 5 times the median are dropped (src.ensembles.filter_members_by_rmse).
    tr = json.loads((ens_lib.TRAINING_DIR / f"training_{mc}_{dsname}.json").read_text())
    rmse_clean = np.array([m["train_rmse_clean"] for m in tr["per_seed"]])
    rmse_noisy = np.array([m["train_rmse_noisy"] for m in tr["per_seed"]])
    _, filt = ens.filter_members_by_rmse(all_members, rmse_clean)
    kept_idx = [i for i in range(len(all_members)) if bool(np.asarray(filt["keep_mask"])[i])]
    filter_report = {"train_rmse_clean": rmse_clean.tolist(), "train_rmse_noisy": rmse_noisy.tolist(), "cutoff": filt["cutoff"],
                     "kept": kept_idx, "dropped": filt["dropped_indices"], "dropped_rmse": filt["dropped_rmse"]}
    log(f"{mc}/{dsname}: K={len(all_members)} trained, kept {len(kept_idx)} {kept_idx}, dropped {filt['dropped_indices']}; "
        f"x0={np.asarray(x0)}, horizon={ens_lib.horizon():.2f}, N={ens_lib.N_EVAL}")

    # ---- LA components of the retained members (per-member cache) -----------------
    comp = {}
    for k in kept_idx:
        path, info_path = cache_dir / f"la_k{k}.npz", cache_dir / f"la_k{k}.json"
        if path.exists() and info_path.exists():
            z = np.load(path)
            comp[k] = {"mu": z["mu"], "var": z["var"], "info": json.loads(info_path.read_text())}
            log(f"  [LA k={k}] loaded from cache")
            continue
        t0 = time.time()
        la = ens_lib.laplace_class(mc)(all_members[k], ds["t_train"], ds["x0s"], ds["ys_noisy"])
        la.fit_laplace_mll()
        mu = np.asarray(la.mean_linearized(t_eval, x0))
        var = np.asarray(ens_lib.lla_variance(la, t_eval, x0))
        info = {"k": k, "prior_precision": float(la.prior_prec_opt), "sigma_noise": float(la.sigma_noise_opt), "D": int(la.D)}
        if mc == "sPHNN":
            info["Omega_MAP"] = float(2.0 * np.pi / ens_lib.return_time(la.rebuild(la.p_map), x0, ens_lib.true_period()))
        np.savez(path, mu=mu, var=var)
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        comp[k] = {"mu": mu, "var": var, "info": info}
        log(f"  [LA k={k}] done in {time.time() - t0:.0f}s (alpha={info['prior_precision']:.4g}, sigma={info['sigma_noise']:.4g})")

    # ---- moments of the three methods ------------------------------------------------
    mus = np.stack([comp[k]["mu"] for k in kept_idx])        # MAP rollouts == linearized means, (K, T, 2)
    variances = np.stack([comp[k]["var"] for k in kept_idx])
    mean = mus.mean(axis=0)
    between = np.einsum("kta,kta->ta", mus - mean, mus - mean) / mus.shape[0]
    arrays = {"t": np.asarray(t_eval),
              "LA__mean": mus[0], "LA__std": np.sqrt(np.maximum(variances[0], 0.0)),
              "MoLA__mean": mean, "MoLA__std": np.sqrt(np.maximum(variances.mean(axis=0) + between, 0.0)),
              "DE__mean": mean, "DE__std": mus.std(axis=0)}
    np.savez(ens_lib.RESULT_DIR / f"methods_{mc}_{dsname}.npz", **arrays)

    # ---- metadata ---------------------------------------------------------------------
    meta = {"model_class": mc, "dataset": dsname, "x0": np.asarray(x0).tolist(), "K": len(kept_idx), "K_trained": len(all_members), "filter": filter_report,
            "horizon": ens_lib.horizon(), "N_EVAL": ens_lib.N_EVAL, "T_true": ens_lib.true_period(),
            "la_components": [{key: val for key, val in comp[k]["info"].items() if key != "Omega_MAP"} for k in kept_idx]}
    if mc == "sPHNN":
        meta["Omega_MAP"] = [comp[k]["info"]["Omega_MAP"] for k in kept_idx]
    (ens_lib.RESULT_DIR / f"methods_{mc}_{dsname}.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"saved results/methods_{mc}_{dsname}.(npz|json)")


if __name__ == "__main__":
    main()
