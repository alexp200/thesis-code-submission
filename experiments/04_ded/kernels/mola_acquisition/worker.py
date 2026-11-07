"""One fresh CPU process of the MoLA acquisition: load one trained member (one component of the mixture), exact Laplace fit, mean and marginal variance of every pool candidate."""

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

from common import ROOT, PRIMARY, atomic_json, digest, read_pod


def run(args):
    """Compute the Laplace component of one member and write its arrays (npz) and record (json)."""
    started = time.perf_counter()
    cpu_started = time.process_time()
    if args.cpu_ids and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(cpu) for cpu in args.cpu_ids.split(",")})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1 before Python starts")
    # Affinity is set before JAX creates its CPU thread pools.
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu":
        raise ValueError("CPU only")
    from importlib.metadata import version
    task = json.loads(args.task.read_text())
    result = {"status": "running", "member_index": args.member_index,
              "task_sha256": digest(args.task), "phase": "load", "timings": {},
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "versions": {name: version(name) for name in
                           ("jax", "jaxlib", "diffrax", "dynax", "klax", "optax", "equinox")}}
    ds = None
    try:
        # ---- checks, then the trained member, the POD and the data ----
        for name, expected in task["checkpoint_versions"].items():
            if result["versions"].get(name) != expected:
                raise ValueError(f"Dependency differs from checkpoint: {name}")
        for relative, expected in task["production_hashes"].items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f"Production source changed: {relative}")
        sys.path.insert(0, str(ROOT / PRIMARY / "core"))
        import _ded_al_core as core
        import ded_data
        from experiments.shared.cluster_recovery.runtime import release_phase_cache

        for path, expected in task["input_hashes"].items():
            if digest(path) != expected:
                raise ValueError(f"Frozen input changed: {path}")
        if digest(ded_data.DATA_PATH) != task["dataset_sha256"]:
            raise ValueError("Dataset changed")
        cfg = core.MODEL_CLASSES[task["model_class"]]
        metadata = json.loads(Path(task["metadata"]).read_text())
        skeleton = core.build_ensemble(cfg.member_builder, list(range(metadata["ensemble_size"])))
        ensemble = eqx.tree_deserialise_leaves(task["checkpoint"], skeleton)
        raw = core.split_ensemble(ensemble, metadata["ensemble_size"])[args.member_index]
        del ensemble, skeleton
        t, ds = ded_data.load_dataset()
        modes_T, modes_q, x_eq = map(jnp.asarray, read_pod(task["pod"]))
        x0, us, ys = ded_data.encode_batch(ds, metadata["train_ids"], modes_T, modes_q, x_eq)
        x0_pool, us_pool = core.pool_arrays(ds, task["pool_ids"], modes_T, modes_q, x_eq)
        jax.block_until_ready((x0, us, ys, x0_pool, us_pool))
        result["timings"]["load_seconds"] = time.perf_counter() - started
        result["phase"] = "ggn"
        atomic_json(args.output.with_suffix(".json"), result)

        # ---- GGN and marginal-likelihood calibration of the Laplace approximation ----
        class TimedLaplace(cfg.laplace_cls):
            eigen_seconds = 0.

            def _prepare_eigenbasis(self):
                start = time.perf_counter()
                super()._prepare_eigenbasis()
                jax.block_until_ready((self.eigvals_ggn, self.eigvecs_ggn))
                self.eigen_seconds += time.perf_counter() - start

        start = time.perf_counter()
        component = TimedLaplace(raw, t, x0, ys, us_train=us)
        jax.block_until_ready(component.GGN)
        result["timings"]["initialize_and_ggn_seconds"] = time.perf_counter() - start
        result["phase"] = "calibrate"
        atomic_json(args.output.with_suffix(".json"), result)
        start = time.perf_counter()
        component.fit_laplace_mll()
        jax.block_until_ready((component.eigvals_ggn, component.eigvecs_ggn,
                               component.prior_prec_opt, component.sigma_noise_opt))
        calibration_seconds = time.perf_counter() - start
        result["timings"].update(eigen_seconds=component.eigen_seconds,
                                eigen_and_mll_seconds=calibration_seconds,
                                mll_excluding_eigen_seconds=calibration_seconds - component.eigen_seconds)
        result["parameter_dimension"] = int(component.D)
        hyperparameters = np.asarray([component.prior_prec_opt, component.sigma_noise_opt])
        if not np.all(np.isfinite(hyperparameters)) or np.any(hyperparameters <= 0):
            raise ValueError("Invalid MLL calibration")
        release_phase_cache("MoLA component after fit", hyperparameters)
        result["phase"] = "jvp_pool"
        atomic_json(args.output.with_suffix(".json"), result)
        start = time.perf_counter()
        # ---- mean and marginal variance of every pool candidate ----
        variance_at = component.make_marginal_variance_eigen_jvp(t, chunk_size=task["chunk_size"])
        mean_at = eqx.filter_jit(lambda initial, controls: component.mean_linearized(t, initial, controls))
        means, variances = [], []
        for initial, controls in zip(x0_pool, us_pool):
            means.append(np.asarray(mean_at(initial, controls)))
            variances.append(np.asarray(variance_at(initial, controls)))
        result["timings"]["pool_mean_and_jvp_seconds"] = time.perf_counter() - start
        arrays = {"means": np.stack(means), "variances": np.stack(variances),
                  "eigvals": np.asarray(component.eigvals_ggn), "hyperparameters": hyperparameters,
                  "ggn_diagonal": np.asarray(jnp.diag(component.GGN))}
        if any(not np.all(np.isfinite(array)) for array in arrays.values()):
            raise ValueError("Nonfinite worker output")
        if np.any(arrays["variances"] < 0):
            raise ValueError("Negative epistemic variance")
        temporary = args.output.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        temporary.replace(args.output.with_suffix(".npz"))
        result.update(status="passed", phase="complete", artifact_sha256=digest(args.output.with_suffix(".npz")))
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        if ds is not None:
            ds.close()
        result.update(wall_seconds=time.perf_counter() - started,
                      cpu_seconds=time.process_time() - cpu_started,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) *
                      (1 if sys.platform == "darwin" else 1024))
        atomic_json(args.output.with_suffix(".json"), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--member-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-ids", default="")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    run(args)
