"""One phase of one AL path per fresh process: train, evaluate or acquire. Finished phases are committed and never repeated, so a task resumes where it stopped.

    python run.py --task T      # runs the next phase of task T (control.py progress lists the tasks)
"""

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time

from design import ROOT, atomic_json, case, digest, load_plan, plan_path, verify
from store import commit, locked, read, save_npz, state, validate_ranking

sys.path.insert(0, str(ROOT))


def environment():
    """Check the CPU / float64 / single-thread setting and record the package versions."""
    import jax
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise ValueError("CPU/x64 required")
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    if any(os.environ.get(n) != "1" for n in names) or "--xla_cpu_parallel_codegen_split_count=1" not in os.environ.get("XLA_FLAGS", "").split():
        raise ValueError("Use the validated single-native-thread CPU launcher")
    return dict(python=platform.python_version(), native_threads={n: os.environ[n] for n in names},
                xla_flags=os.environ.get("XLA_FLAGS"), backend="cpu", x64=True,
                versions={n: importlib.metadata.version(n) for n in ("jax", "jaxlib", "equinox", "klax", "dynax", "diffrax", "optax", "numpy")},
                vcs_packages={n: json.loads(importlib.metadata.distribution(n).read_text("direct_url.json") or "null") for n in ("dynax", "klax")})


# ------------------------------------------------------------------------------------------------ phase: train
def train(folder, plan, task, i, ids, data):
    """Train the K = 3 members from scratch on the current training set and commit checkpoint, history and record."""
    import equinox as eqx
    import jax
    import numpy as np
    from numerical import helpers
    tc, _ = helpers()
    spec = plan["tasks"][task]
    seeds = plan["reporting_seeds"][str(spec["path"])][i]
    recipe = plan["recipes"][spec["model"]]
    start = time.perf_counter()
    ensemble, history = tc.train_ensemble(spec["model"], list(range(3)), data,
                                         steps=recipe["steps"], lr=recipe["lr"], key_offset=seeds[0],
                                         log_every=max(recipe["steps"]//10, 1), l2_regularization=recipe["l2"])
    jax.block_until_ready(ensemble)
    if not all(np.all(np.isfinite(np.asarray(a))) for a in jax.tree.leaves(ensemble) if eqx.is_array(a)):
        raise ValueError("Nonfinite trained parameters")
    seconds = time.perf_counter()-start
    norms = [float(tc.active_parameter_squared_norm(m, spec["model"])) for m in tc.split_ensemble(ensemble, 3)]
    model = folder / f"iter{i:02d}.eqx"
    temporary = model.with_suffix(".eqx.tmp")
    eqx.tree_serialise_leaves(temporary, ensemble)
    temporary.replace(model)
    hist = folder / f"iter{i:02d}_history.pkl"
    history.save(hist)
    verify(plan)
    commit(folder / f"iter{i:02d}_train.json",
           dict(iteration=i, dataset_size=len(ids), train_ids=ids, initialization_seeds=seeds,
                optimizer_key=seeds[0]+1, training_key=seeds[0], ensemble_size=3, recipe=recipe,
                normalization=dict(ambient_K=279.15, us_std=np.asarray(data["us_std"]).tolist(),
                                   ys_std=np.asarray(data["ys_std"]).tolist(), fitted_on_ids=ids),
                active_parameter_squared_norm=norms, train_seconds=seconds,
                hostname=platform.node(), slurm_job_id=os.environ.get("SLURM_JOB_ID")), [model, hist])


# ------------------------------------------------------------------------------------------------ phase: evaluate
def load_ensemble(folder, model, i, trained):
    """Load the committed ensemble of training-set size i."""
    import equinox as eqx
    from numerical import skeleton
    return eqx.tree_deserialise_leaves(folder / f"iter{i:02d}.eqx", skeleton(model, trained["initialization_seeds"]))


def batch_record(folder, signature, name, arrays, seconds):
    """Commit one block of results together with the signature of its inputs."""
    array_path = folder / f"{name}.npz"
    save_npz(array_path, **arrays)
    commit(folder / f"{name}.json", dict(input_signature=signature, seconds=seconds), [array_path])


def cached_batch(folder, signature, name):
    """Committed block, or None if it has to be computed; raises if it was computed from other inputs."""
    record = read(folder / f"{name}.json")
    if record is not None and record["input_signature"] != signature:
        raise ValueError("Inference cache uses different inputs")
    return record


def evaluate(folder, plan, task, i, ids, data, trained):
    """Training filter and member errors on the AP15 evaluation set (blockwise, resumable); commits the report of this training-set size."""
    import numpy as np
    from numerical import helpers, metrics, predict_batch, predictor
    from src.ensembles import member_train_rmse, filter_members_by_rmse
    spec = plan["tasks"][task]
    ensemble = load_ensemble(folder, spec["model"], i, trained)
    tc, _ = helpers()
    members = tc.split_ensemble(ensemble, 3)
    # training filter: members with training RMSE > 5 x median are excluded from the ensemble statistics
    losses = member_train_rmse(members, data["t"], data["x0s_train"], data["ys_train"], us_train=data["us_train"])
    if not np.all(np.isfinite(np.asarray(losses))):
        raise ValueError("Nonfinite training rollout; cannot filter silently")
    _, filtered = filter_members_by_rmse(members, losses)
    dropped = filtered["dropped_indices"]
    if len(dropped) == 3:
        raise ValueError("All training members failed convergence guard")
    signature = dict(train_commit_sha256=digest(folder / f"iter{i:02d}_train.json"), plan_sha256=digest(plan_path(plan["smoke"])))
    cache = folder / "evaluation" / f"iter{i:02d}"
    fn = predictor(data)
    batch = plan["acquisition"]["evaluation_batch"]
    block = plan["acquisition"]["cache_block"]
    indices = [data["all_ids"].index(sid) for sid in plan["evaluation_ids"]]
    eval_x, eval_u = data["x"][np.asarray(indices)], data["u"][np.asarray(indices)]
    targets = data["target_K"][indices]
    # errors of all members on the evaluation trajectories, in cached blocks
    mses, maxima, mean_mses = [], [], []
    seconds = 0.
    for start in range(0, len(indices), block):
        name = f"batch{start:04d}"
        record = cached_batch(cache, signature, name)
        if record is None:
            tick = time.perf_counter()
            end = min(start+block, len(indices))
            pieces = [predict_batch(fn, ensemble, eval_x[s:min(s+batch, end)], eval_u[s:min(s+batch, end)], batch)
                      for s in range(start, end, batch)]
            pred = np.concatenate(pieces, axis=1)
            errors = pred*np.asarray(data["ys_std"])+279.15-targets[None, start:end]
            if not np.all(np.isfinite(errors)):
                raise ValueError("Nonfinite evaluation member: no silent reduction of reporting K")
            kept = [k for k in range(3) if k not in dropped]
            mean_mse = np.stack((np.mean(errors.mean(axis=0)**2, axis=1),
                                 np.mean(errors[kept].mean(axis=0)**2, axis=1)))
            batch_record(cache, signature, name, dict(per_probe_mse_K2=np.mean(errors**2, axis=2),
                                                      max_error_K=np.max(np.abs(errors), axis=(2, 3)),
                                                      ensemble_mean_mse_K2=mean_mse), time.perf_counter()-tick)
            record = read(cache / f"{name}.json")
        with np.load(cache / f"{name}.npz", allow_pickle=False) as chunk:
            mses.append(chunk["per_probe_mse_K2"])
            maxima.append(chunk["max_error_K"])
            mean_mses.append(chunk["ensemble_mean_mse_K2"])
        seconds += record["seconds"]
    report = metrics(np.concatenate(mses, axis=1), np.concatenate(maxima, axis=1), plan, ids, dropped,
                     np.concatenate(mean_mses, axis=1))
    verify(plan)
    commit(folder / f"iter{i:02d}.json", dict(iteration=i, dataset_size=len(ids), train_ids=ids,
                                              initialization_seeds=trained["initialization_seeds"], evaluation=report,
                                              train_commit_sha256=signature["train_commit_sha256"],
                                              train_rmse_latent=np.asarray(losses).tolist(), dropped_indices=dropped,
                                              dropped_train_rmse=filtered["dropped_rmse"], training_filter_cutoff=filtered["cutoff"],
                                              effective_ensemble_size=3-len(dropped), evaluation_seconds=seconds))
    print(json.dumps(report["all_members"]), flush=True)


# ------------------------------------------------------------------------------------------------ phase: acquire
def acquire(folder, plan, task, i, ids, data, trained):
    """Choose the next trajectory: Random follows the predeclared order; DE / LA / MoLA score every candidate of the pool (both criteria) and pick the best of the task's criterion."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from numerical import acquisition_kernels
    from src.active_learning import score_sum_marginal, score_sum_variance
    spec = plan["tasks"][task]
    pool = [sid for sid in plan["pool_ids"] if sid not in ids]
    path = folder / "selections" / f"iter{i:02d}.json"
    report = read(folder / f"iter{i:02d}.json")
    signature = dict(report_sha256=digest(folder / f"iter{i:02d}.json"),
                     train_commit_sha256=digest(folder / f"iter{i:02d}_train.json"), plan_sha256=digest(plan_path(plan["smoke"])))
    started = time.perf_counter()
    if spec["method"] == "random":
        picked = plan["random_orders"][str(spec["path"])][i+1]
        verify(plan)
        commit(path, dict(**signature, pool_ids=pool, picked_id=picked, ranking=None,
                          random_seed=20262012+spec["path"], wall_seconds=time.perf_counter()-started))
        return
    # scores of all pool candidates, in cached blocks
    retained = [k for k in range(3) if k not in report["dropped_indices"]]
    cache = folder / "acquisition" / f"iter{i:02d}"
    batch = plan["acquisition"]["candidate_batch"]
    block = plan["acquisition"]["cache_block"]
    starts = list(range(0, len(pool), block))
    missing = [s for s in starts if cached_batch(cache, signature, f"batch{s:04d}") is None]
    fits = []
    kernel = None
    if missing:
        ensemble = load_ensemble(folder, spec["model"], i, trained)
        kernel, fits = acquisition_kernels(spec["model"], spec["method"], ensemble, data, retained, plan["acquisition"])
        # Persist calibration provenance for every fit attempt, including interrupted scoring.
        cache.mkdir(parents=True, exist_ok=True)
        atomic_json(cache / f"fit_attempt_{time.time_ns()}.json", dict(input_signature=signature, fits=fits))
    scored, seconds = [], 0.
    idx = [data["all_ids"].index(sid) for sid in pool]
    aggregate = jax.jit(jax.vmap(lambda var: jnp.stack((score_sum_marginal(var, 0.), score_sum_variance(var, 0.)))))
    for start in starts:
        name = f"batch{start:04d}"
        record = cached_batch(cache, signature, name)
        if record is None:
            tick = time.perf_counter()
            end = min(start+block, len(pool))
            pieces = []
            for s in range(start, end, batch):
                indices = idx[s:min(s+batch, end)]
                count = len(indices)
                padded = indices + [indices[-1]]*(batch-count)
                outputs = jax.block_until_ready(kernel(data["x"][jnp.asarray(padded)], data["u"][jnp.asarray(padded)]))
                pieces.append(tuple(np.asarray(a[:count]) for a in outputs))
            total, within, between = [np.concatenate([piece[k] for piece in pieces]) for k in range(3)]
            values = np.asarray(jax.block_until_ready(aggregate(jnp.asarray(total))))
            arrays = dict(candidate_ids=np.asarray(pool[start:end]), scores=values,
                          total_variance=total, within_variance=within, between_variance=between)
            if not all(np.all(np.isfinite(a)) for a in arrays.values()):
                raise ValueError("Nonfinite acquisition outputs")
            batch_record(cache, signature, name, arrays, time.perf_counter()-tick)
            record = read(cache / f"{name}.json")
        with np.load(cache / f"{name}.npz", allow_pickle=False) as chunk:
            np.testing.assert_array_equal(chunk["candidate_ids"], pool[start:start+block])
            scored.append(chunk["scores"])
        seconds += record["seconds"]
    # ranking by each criterion; ties are broken by the smaller trajectory id
    matrix = np.concatenate(scored)
    rankings = {}
    for column, name in enumerate(("sum_marginal", "sum_variance")):
        rankings[name] = sorted([dict(simulation_id=sid, score=float(v)) for sid, v in zip(pool, matrix[:, column])],
                                key=lambda r: (-r["score"], r["simulation_id"]))
    ranking = rankings[spec["criterion"]]
    picked = ranking[0]["simulation_id"]
    validate_ranking(ranking, pool, picked)
    verify(plan)
    commit(path, dict(**signature, pool_ids=pool, picked_id=picked, ranking=ranking, rankings=rankings,
                      criterion=spec["criterion"], acquisition_member_indices=[0] if spec["method"] == "LA" else retained,
                      scoring_seconds=seconds, fit_attempts_directory=str(cache.relative_to(folder)),
                      wall_seconds=time.perf_counter()-started))
    print(f"Picked {picked}, cached both criteria for {len(pool)} candidates", flush=True)


# ------------------------------------------------------------------------------------------------ driver
def run(task, smoke):
    """Run the next phase of one task under the path lock."""
    env = environment()
    plan = load_plan(smoke)
    spec = plan["tasks"][task]
    folder = case(plan, task)
    with locked(folder):
        expected = dict(plan_sha256=digest(plan_path(smoke)), task=spec, environment=env)
        manifest = read(folder / "manifest.json")
        if manifest is None:
            commit(folder / "manifest.json", expected)
        elif any(manifest[k] != v for k, v in expected.items()):
            raise ValueError("Source/recipe/environment changed; refuse mixed resume")
        i, phase, ids = state(folder, plan, task)
        print(f"task={task} {spec} N={len(ids)} phase={phase}", flush=True)
        if phase == "done":
            return
        from numerical import prepare_data
        data = prepare_data(ids)
        if phase == "train":
            train(folder, plan, task, i, ids, data)
        else:
            trained = read(folder / f"iter{i:02d}_train.json")
            if phase == "evaluate":
                evaluate(folder, plan, task, i, ids, data, trained)
            else:
                acquire(folder, plan, task, i, ids, data, trained)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True, choices=range(200))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.task, args.smoke)
