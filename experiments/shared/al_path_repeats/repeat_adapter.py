"""Training and reporting of DED ensembles for the study workflows (`04_ded/al_runs`), on top of the kernels in `04_ded/kernels` with explicit per-path seed keys."""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load(model, size):
    """Adapter for one architecture ('NODE' or 'sPHNN') and a reporting ensemble of `size` members; holds the opened data set (close it when done)."""
    sys.path.insert(0, str(ROOT))
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrd
    from experiments.shared.cluster_recovery.runtime import release_phase_cache

    base = importlib.import_module("experiments.04_ded.kernels.la_de_runner.run_al")
    base.check_environment()
    core, original = base.core, base.original
    t, ds = base.load_dataset()

    class Adapter:
        def __init__(self):
            self.recipe = original.training_hyperparams_for_meta(core.MODEL_CLASSES[model], original.FINAL_TRAJECTORY_L2[model])
            self.evaluation_ids = list(base.EVALUATION_IDS)

        def skeleton(self, size):
            """Untrained ensemble with the structure of a stored checkpoint."""
            return core.build_ensemble(core.MODEL_CLASSES[model].member_builder, list(range(size)))

        def seeds(self, key_value, size):
            """Initialisation seeds of the members for a training key."""
            return base.member_seeds(key_value, size)

        def train(self, data, size, key_value):
            """Two-stage training of `size` members; returns (ensemble, history, seconds)."""
            result = original.train_ensemble(core.MODEL_CLASSES[model], data, size,
                                             original.FINAL_TRAJECTORY_L2[model], jrd.key(key_value))
            jax.block_until_ready(result[0])
            if not all(bool(jnp.all(jnp.isfinite(a))) for a in jax.tree.leaves(result[0]) if eqx.is_array(a)):
                raise ValueError("Nonfinite trained parameters")
            release_phase_cache("AL repeat training", result[0])
            return result

        def filter_and_report(self, ensemble, data):
            """Training-RMSE filter and Kelvin errors of all members on the 25 simulations; returns (evaluation, filter record)."""
            _, _, dropped, rmse, cutoff, seconds = core.filter_converged_members(
                core.MODEL_CLASSES[model], data, ensemble, size, "AL repeat reporting")
            members = [core.callable_member(core.MODEL_CLASSES[model], m) for m in core.split_ensemble(ensemble, size)]
            metrics = base.evaluate_members(members, data, base.decode_state, dropped)
            metrics["evaluation_ids"] = self.evaluation_ids
            release_phase_cache("AL repeat reporting", metrics)
            mean_error = metrics["filtered_ensemble_mean_rmse_K"]
            metrics["distribution"] = f"K{size} training members conditional on this acquisition replicate"
            return metrics, {"dropped_indices": dropped, "dropped_train_rmse": rmse, "cutoff": cutoff,
                             "effective_ensemble_size": size - len(dropped), "filter_seconds": seconds,
                             "ensemble_mean_rmse_K": mean_error}

        def close(self):
            """Close the data set."""
            ds.close()

    return Adapter()
