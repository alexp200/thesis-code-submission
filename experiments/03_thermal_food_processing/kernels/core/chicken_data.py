"""Normalisation of the thermal food processing data.

All simulations start at the ambient temperature T_AMBIENT. The oven temperature and the two probe temperatures are shifted by it and divided by their standard deviation over the CURRENT
training trajectories (Roth et al. 2025, Appendix E), so the scales change when the training set grows.
"""

import jax.numpy as jnp

T_AMBIENT = 279.15  # K, shared initial temperature of all simulations


def normalization_stats(us_K_train, ys_K_train):
    """Scales from the training trajectories only: one standard deviation of the oven temperature and one per probe (T_A, T_B)."""
    us_std = jnp.std(us_K_train)
    ys_std = jnp.std(ys_K_train, axis=(0, 1))  # (2,), one std per probe
    return us_std, ys_std
