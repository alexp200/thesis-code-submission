"""Reporting of an ensemble (Kelvin errors of every member on all 25 simulations) and the DE acquisition score (MGE of the empirical ensemble variance)."""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src import active_learning as al
from src.ensembles import ens_trajectory_flat_var


def summarize_errors(values):
    """Median and quartiles of the member errors; the statistic is only computed if all members are valid."""
    valid = [value for value in values if value is not None and np.isfinite(value)]
    result = {"total_count": len(values), "valid_count": len(valid),
              "status": "complete" if len(valid) == len(values) and values else "incomplete",
              "median_K": None, "q25_K": None, "q75_K": None, "iqr_K": None}
    # Do not silently change an all-K statistic into a converged-only statistic.
    if result["status"] == "complete":
        q25, median, q75 = np.quantile(valid, [.25, .5, .75], method="linear")
        result.update(median_K=float(median), q25_K=float(q25), q75_K=float(q75),
                      iqr_K=float(q75 - q25))
    return result


def evaluate_members(members, data, decode_state, dropped_indices=()):
    """Kelvin RMSE of every member (also the training-filtered ones) over all 25 simulations and of the mean of the retained members; every rollout is used for both. Only latent predictions are kept."""
    start = time.perf_counter()
    dropped = set(dropped_indices)
    records, retained = [], []
    retained_failed = False
    for index, member in enumerate(members):
        record = {"member_index": index, "training_filter_kept": index not in dropped,
                  "status": "passed", "rmse_K": None, "per_trajectory_rmse_K": None}
        try:
            prediction = jax.vmap(member, in_axes=(None, 0, 0))(
                data["t"], data["x0s_evaluation"], data["us_evaluation"])
            temperature = decode_state(prediction, data["modes_T"], data["x_eq"])
            if not bool(jnp.all(jnp.isfinite(temperature))):
                record["status"] = "nonfinite_rollout"
            else:
                mse = jnp.mean((temperature - data["T_true_evaluation"]) ** 2, axis=(1, 2))
                errors = np.asarray(jnp.sqrt(mse))
                value = float(jnp.sqrt(jnp.mean(mse)))
                if not np.isfinite(value) or not np.all(np.isfinite(errors)):
                    record["status"] = "nonfinite_error"
                else:
                    record.update(rmse_K=value, per_trajectory_rmse_K=errors.tolist())
                    if index not in dropped:
                        retained.append(prediction)
        except (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError) as error:
            expected_failures = ("The maximum number of solver steps was reached",
                                 "The minimum step size was reached in the differential equation solver",
                                 "Maximum number of rejected steps was reached")
            if not any(message in str(error) for message in expected_failures):
                raise
            record.update(status="solver_failed", error=str(error))
        if record["status"] != "passed" and index not in dropped:
            retained_failed = True
        records.append(record)
    mean_error = None
    if retained and not retained_failed:
        mean = jnp.mean(jnp.stack(retained), axis=0)
        temperature = decode_state(mean, data["modes_T"], data["x_eq"])
        value = float(jnp.sqrt(jnp.mean((temperature - data["T_true_evaluation"]) ** 2)))
        mean_error = value if np.isfinite(value) else None
    all_errors = [record["rmse_K"] for record in records]
    kept_errors = [record["rmse_K"] for record in records if record["training_filter_kept"]]
    return {"schema_version": 1, "units": "K", "quantile_method": "linear",
            "scope": "all evaluation trajectories, including selected training data",
            "distribution": "member training variability conditional on one acquisition path",
            "members": records, "all_members": summarize_errors(all_errors),
            "training_filtered_members": summarize_errors(kept_errors),
            "filtered_ensemble_mean_rmse_K": mean_error,
            "seconds": time.perf_counter() - start}


def stack_members(members):
    """Stack the members into one vmapped ensemble."""
    if len(members) < 2:
        raise ValueError("DE acquisition requires at least two training-converged members")
    return jax.tree.map(lambda *xs: jnp.stack(xs) if eqx.is_array(xs[0]) else xs[0], *members)


def score_pool_de(members, t, x0_pool, us_pool):
    """MGE score of every pool candidate from the empirical ensemble variance of the rollouts (ddof = 0); no dense covariance, GGN or Laplace fit."""
    ensemble = stack_members(members)

    @eqx.filter_jit
    def score_one(ensemble, x0, us):
        variance = ens_trajectory_flat_var(ensemble, t, x0, us)
        return al.score_sum_marginal(variance, sigma_noise=0.0)

    scores = jnp.stack([score_one(ensemble, x0, us) for x0, us in zip(x0_pool, us_pool)])
    if not bool(jnp.all(jnp.isfinite(scores))):
        raise RuntimeError("DE acquisition produced nonfinite pool scores")
    return scores
