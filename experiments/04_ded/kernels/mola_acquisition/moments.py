"""Uniform epistemic mixture of the Laplace components (MoLA) and its acquisition score."""

import numpy as np


def mixture(means, variances):
    """Mean, within- and between-component variance and total variance of the uniform mixture; arrays have shape (K, candidates, time, state)."""
    means, variances = np.asarray(means), np.asarray(variances)
    if means.shape != variances.shape or means.ndim != 4 or means.shape[0] < 1:
        raise ValueError("Expected matching (K, candidates, time, state) arrays")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(variances)):
        raise ValueError("Nonfinite component predictions")
    if np.any(variances < 0):
        raise ValueError("Negative component variance")
    within = np.mean(variances, axis=0)
    between = np.var(means, axis=0, ddof=0)
    return {"mean": np.mean(means, axis=0), "within": within,
            "between": between, "total": within + between}


def acquisition_scores(variance):
    """MGE score of every candidate from the total variance (src.active_learning with its floors and aggregation)."""
    import jax
    import jax.numpy as jnp
    from src import active_learning as al

    jax.config.update("jax_enable_x64", True)
    flat = jnp.asarray(variance).reshape(len(variance), -1)
    return np.asarray(jax.vmap(al.score_sum_marginal)(flat))
