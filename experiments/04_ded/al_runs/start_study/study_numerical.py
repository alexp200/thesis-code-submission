"""The two kinds of task of the DED workflows: training of an ensemble (with its POD basis) and acquisition (LA, DE or MoLA scores of the pool); results are committed as immutable artifacts."""

import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from study_design import LOCAL, ROOT, PRIMARY, MOLA, STORED_SHARED, canonical, digest, recipe
from study_store import artifact, atomic, execution, ranking, read, training, validate_members, writer


# ------------------------------------------------------------------------------------------------ modules and data
def modules(request):
    """Adapter (training and reporting of the architecture of the request) and the kernel entry module."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "experiments/shared/al_path_repeats"))
    import repeat_adapter
    adapter = repeat_adapter.load(request["model_class"], request["ensemble_size"])
    if adapter.recipe != recipe(request["model_class"]):
        raise ValueError("Production training recipe differs from the frozen recipe")
    base = importlib.import_module("experiments.04_ded.kernels.la_de_runner.run_al")
    return adapter, base


def prepare(base, ids, pod_file, *, reference=None, pod_cache=None):
    """Encoded training and evaluation data for the training set `ids`; the POD basis is read from pod_file or the reference, else fitted (and cached) and written to pod_file."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from experiments.shared.cluster_recovery.pod_cache import load_or_fit
    from experiments.shared.cluster_recovery.runtime import release_phase_cache

    t, ds = base.load_dataset()
    try:
        if pod_file.exists():
            with np.load(pod_file, allow_pickle=False) as saved:
                coordinates = tuple(jnp.asarray(saved[k]) for k in ("modes_T", "modes_q", "x_eq"))
        elif reference is not None:
            with np.load(reference, allow_pickle=False) as saved:
                coordinates = tuple(jnp.asarray(saved[k]) for k in ("modes_T", "modes_q", "x_eq"))
        else:
            coordinates = load_or_fit(
                ds, ids, base.core.N_MODES, fitter=base.core.fit_pods, dataset_path=base.original.DATA_PATH,
                cache_dir=pod_cache if pod_cache is not None else pod_file.parent / "pod_fit", implementation_paths=(
                    ROOT / "src/preprocessing.py", PRIMARY / "core/ded_data.py"))
        if not all(bool(jnp.all(jnp.isfinite(a))) for a in coordinates):
            raise ValueError("Nonfinite POD coordinates")
        expected_shapes = [(ds.sizes["node"], base.core.N_MODES)] * 2 + [(base.core.N_MODES,)]
        if [tuple(a.shape) for a in coordinates] != expected_shapes:
            raise ValueError("Unexpected POD dimensions")
        for modes in coordinates[:2]:
            array = np.asarray(modes)
            c2 = np.sum(array[:, 0] ** 2)
            if c2 <= 0 or not np.allclose(array.T @ array / c2, np.eye(base.core.N_MODES), rtol=1e-8, atol=1e-8):
                raise ValueError("Invalid original POD normalization/orthogonality")
        if reference is not None:
            with np.load(reference, allow_pickle=False) as saved:
                for value, name in zip(coordinates, ("modes_T", "modes_q", "x_eq")):
                    if not np.array_equal(np.asarray(value), saved[name]):
                        raise ValueError("POD differs from the frozen reporting/reference coordinates")
        if not pod_file.exists():
            with pod_file.with_suffix(".npz.tmp").open("wb") as handle:
                np.savez(handle, **dict(zip(("modes_T", "modes_q", "x_eq"), map(np.asarray, coordinates))))
            pod_file.with_suffix(".npz.tmp").replace(pod_file)
        modes_T, modes_q, x_eq = coordinates
        train = base.core.encode_batch(ds, ids, *coordinates)
        evaluation = base.core.encode_batch(ds, list(range(25)), *coordinates)
        data = {"t": t, "train_ids": ids, "modes_T": modes_T, "modes_q": modes_q, "x_eq": x_eq,
                **dict(zip(("x0s_train", "us_train", "xs_train"), train)),
                **dict(zip(("x0s_evaluation", "us_evaluation", "xs_evaluation"), evaluation)),
                "T_true_evaluation": jnp.asarray(ds.temperature.isel(batch=list(range(25))).values)}
        jax.block_until_ready(data)
        if not all(bool(jnp.all(jnp.isfinite(value))) for value in data.values() if hasattr(value, "shape")):
            raise ValueError("Nonfinite encoded training/evaluation data")
        release_phase_cache("start study preprocessing", data)
        return data
    finally:
        ds.close()


# ------------------------------------------------------------------------------------------------ training task
def train(run, request, manifest):
    """Train (or reuse) the ensemble of a request in run/shared/<request id>/ and commit train.json."""
    folder = run / "shared" / request["id"]
    if training(run, request) is not None:
        return
    with writer(folder):
        if training(run, request) is not None:
            return
        adapter, base = modules(request)
        started = time.perf_counter()
        try:
            reference = request.get("reuse_source")
            if reference:
                old = read(Path(reference))
                # Old repeat artifacts have their own checksum fields rather than artifacts.
                checkpoint = Path(reference).with_suffix(".eqx")
                pod = Path(reference).with_name(Path(reference).stem + "_pod.npz")
                for path, key in ((checkpoint, "checkpoint_sha256"), (pod, "pod_sha256")):
                    if digest(path) != old[key]:
                        raise ValueError(f"Changed reuse source: {path}")
                if (canonical(old["train_ids"]) != request["train_ids"] or
                        old["training_key"] != request["training_key"] or old["ensemble_size"] != 3):
                    raise ValueError("Reuse source has different IDs/key/K")
                validate_members(old["evaluation"])
                import_history = Path(reference).with_name(Path(reference).stem + "_history.pkl")
                if not import_history.is_file():
                    raise ValueError("Reuse source history missing")
                for source, target in ((checkpoint, "model.eqx"), (pod, "pod.npz"), (import_history, "history.pkl")):
                    shutil.copyfile(source, folder / target)
                    if digest(source) != digest(folder / target):
                        raise ValueError("Reuse source changed during copy")
                metadata = {k: old[k] for k in ("evaluation", "dropped_indices", "effective_ensemble_size",
                                                "initialization_seeds", "training_key", "ensemble_size")}
                metadata.update(train_seconds=0., reused_training_seconds=old["train_seconds"],
                                reused_from=artifact(Path(reference), folder))
            else:
                reference_pod = request.get("reference_pod")
                stored_pod = next((d / request["id"] / "pod.npz" for d in STORED_SHARED if (d / request["id"] / "pod.npz").is_file()), None)
                if LOCAL and reference_pod is None and stored_pod is not None:
                    # The SVD sign convention of the POD basis depends on the platform; the stored basis (large_assets) keeps the latent coordinates of the thesis run.
                    reference_pod = {"path": str(stored_pod), "sha256": digest(stored_pod)}
                if request.get("reference_shared"):
                    reporting = run / "shared" / request["reference_shared"]
                    committed = read(reporting / "train.json")
                    if committed is None or committed["train_ids"] != request["train_ids"]:
                        raise ValueError("LA preparation requires the committed reporting POD")
                    reference_pod = {"path": str(reporting / "pod.npz"),
                                     "sha256": committed["artifacts"]["pod"]["sha256"]}
                if reference_pod:
                    if digest(reference_pod["path"]) != reference_pod["sha256"]:
                        raise ValueError("Frozen G1/reporting POD changed")
                data = prepare(base, request["train_ids"], folder / "pod.npz",
                               reference=Path(reference_pod["path"]) if reference_pod else None,
                               pod_cache=run / "pod_bases")
                ensemble, history, seconds = adapter.train(data, request["ensemble_size"], request["training_key"])
                import equinox as eqx
                eqx.tree_serialise_leaves(folder / "model.eqx.tmp", ensemble)
                (folder / "model.eqx.tmp").replace(folder / "model.eqx")
                history.save(folder / "history.pkl.tmp")
                (folder / "history.pkl.tmp").replace(folder / "history.pkl")
                metadata = {"training_key": request["training_key"], "ensemble_size": request["ensemble_size"],
                            "initialization_seeds": adapter.seeds(request["training_key"], request["ensemble_size"]),
                            "train_seconds": seconds}
                if request["role"] == "reporting":
                    evaluation, filtering = adapter.filter_and_report(ensemble, data)
                    validate_members(evaluation)
                    metadata.update(evaluation=evaluation, **filtering)
                else:
                    raw = base.core.split_ensemble(ensemble, 1)[0]
                    metadata["acquisition_train_rmse_latent"] = base.core.acquisition_train_rmse(
                        base.core.MODEL_CLASSES[request["model_class"]], data, raw)
            metadata.update(status="passed", request=request, model_class=request["model_class"],
                            train_ids=request["train_ids"], dataset_size=request["dataset_size"],
                            training_hyperparameters=adapter.recipe, manifest_sha256=digest(run / "manifest.json"),
                            execution=execution(), wall_seconds=time.perf_counter() - started,
                            artifacts={name: artifact(folder / filename, folder) for name, filename in
                                       (("checkpoint", "model.eqx"), ("pod", "pod.npz"), ("history", "history.pkl"))})
            metadata.update(checkpoint_sha256=metadata["artifacts"]["checkpoint"]["sha256"],
                            pod_sha256=metadata["artifacts"]["pod"]["sha256"],
                            history_sha256=metadata["artifacts"]["history"]["sha256"])
            atomic(folder / "train.json", metadata)
        finally:
            adapter.close()


# ------------------------------------------------------------------------------------------------ acquisition task
def acquire(run, request, manifest, allocation):
    """Score the remaining pool with LA, DE or MoLA from the committed ensemble and commit the ranking (selection.json)."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    folder = run / "acquisitions" / request["id"]
    if read(folder / "selection.json") is not None:
        return
    with writer(folder):
        if read(folder / "selection.json") is not None:
            return
        shared = run / "shared" / request["model_id"]
        metadata = read(shared / "train.json")
        adapter, base = modules(metadata["request"])
        started = time.perf_counter()
        try:
            pool = [i for i in range(25) if i not in request["train_ids"]]
            timings, artifacts = {}, {}
            # MoLA: Laplace components of the retained members 0, 1, 2 in fresh processes, uniform mixture of their means and variances
            if request["method"] == "MoLA":
                sys.path.insert(0, str(MOLA))
                executor = importlib.import_module("benchmark")
                moments = importlib.import_module("moments")
                members = [i for i in (0, 1, 2) if i not in metadata["dropped_indices"]]
                if not members:
                    raise ValueError("All fixed MoLA components training-filtered; no replacement")
                task = {"model_class": request["model_class"], "N": len(request["train_ids"]),
                        "train_ids": request["train_ids"], "pool_ids": pool, "chunk_size": 128,
                        "dataset_sha256": manifest["dataset_sha256"],
                        "checkpoint_versions": manifest["environment"]["versions"],
                        "production_hashes": manifest["implementation_sha256"],
                        "metadata": str(shared / "train.json"), "checkpoint": str(shared / "model.eqx"),
                        "pod": str(shared / "pod.npz"),
                        "input_hashes": {str(shared / name): digest(shared / name)
                                         for name in ("train.json", "model.eqx", "pod.npz")}}
                atomic(folder / "task.json", task)
                pinned = hasattr(os, "sched_setaffinity") and not LOCAL
                parent_affinity = os.sched_getaffinity(0) if pinned else None
                try:
                    if pinned:
                        os.sched_setaffinity(0, set(allocation))
                    group = executor.run_group(folder / "task.json", members, min(2, len(members)),
                                               16, folder / "components", resume=True)
                finally:
                    if pinned:
                        os.sched_setaffinity(0, parent_affinity)
                components = executor.load_outputs(folder / "components", members)
                mixture = moments.mixture([c["means"] for c in components], [c["variances"] for c in components])
                scores = moments.acquisition_scores(mixture["total"])
                with (folder / "pool_moments.npz").open("wb") as handle:
                    np.savez(handle, pool_ids=pool, member_indices=members,
                             component_means=np.stack([c["means"] for c in components]),
                             component_variances=np.stack([c["variances"] for c in components]), **mixture)
                timings.update(group=group, effective_components=len(members), used_indices=members,
                               degraded_mixture=len(members) < 3)
                artifacts["moments"] = artifact(folder / "pool_moments.npz", folder)
                for i in members:
                    for suffix in (".json", ".npz"):
                        path = folder / "components" / f"component{i:02d}{suffix}"
                        artifacts[path.stem + suffix] = artifact(path, folder)
            # LA: Laplace approximation of the separate single-member acquisition model; DE: empirical variance of the retained reporting members
            else:
                data = prepare(base, request["train_ids"], shared / "pod.npz")
                ensemble = eqx.tree_deserialise_leaves(shared / "model.eqx", adapter.skeleton(metadata["ensemble_size"]))
                t, ds = base.load_dataset()
                try:
                    initial, controls = base.core.pool_arrays(ds, pool, data["modes_T"], data["modes_q"], data["x_eq"])
                    cfg = base.core.MODEL_CLASSES[request["model_class"]]
                    if request["method"] == "LA":
                        raw = base.core.split_ensemble(ensemble, 1)[0]
                        base.core.acquisition_train_rmse(cfg, data, raw)
                        component, seconds = base.core.fit_single_seed_laplace(cfg, data, raw)
                        scores = np.asarray(base.core.score_pool_sum_marginal(component, t, initial, controls, chunk_size=128))
                        timings["laplace_fit_seconds"] = seconds
                        np.savez(folder / "calibration.npz", eigvals=np.asarray(component.eigvals_ggn),
                                 prior_prec=np.asarray(component.prior_prec_opt), sigma_noise=np.asarray(component.sigma_noise_opt))
                        artifacts["calibration"] = artifact(folder / "calibration.npz", folder)
                    else:
                        _, members, *_ = base.core.filter_converged_members(cfg, data, ensemble, 3, "start study DE")
                        scores = np.asarray(base.score_pool_de(members, t, initial, controls))
                finally:
                    ds.close()
            rows = ranking(scores, pool)
            result = {"status": "passed", "request": request, "train_ids": request["train_ids"], "pool_ids": pool,
                      "picked_id": rows[0]["batch_id"], "ranking": rows, "sigma_noise": 0.,
                      "model_commit": artifact(shared / "train.json", folder),
                      "artifacts": artifacts, "timings": timings, "execution": execution(),
                      "wall_seconds": time.perf_counter() - started, "manifest_sha256": digest(run / "manifest.json")}
            atomic(folder / "selection.json", result)
        finally:
            adapter.close()
