"""Data loading, POD fit and encoding of the DED data set (25 simulations on a 5 x 5 grid of source speed and power, 1836 nodes, 401 time points).

POD: separate bases for the temperature field (state) and the heat-source field (input), each fitted to the snapshots of the training trajectories only (src.preprocessing.fit_pod). The state coordinates
are shifted so that a uniform ambient temperature (ROOM_TEMPERATURE) maps to the latent origin. The state basis is fitted BEFORE this shift, the convention of the thesis runs, so that stored checkpoints stay valid.
"""

from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import xarray as xr

from src.preprocessing import fit_pod, encode_pod, decode_pod, shift_equilibrium_pod

_REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = _REPO_ROOT / "data" / "directed_energy_deposition" / "max_simulation" / "dataset.nc"

ROOM_TEMPERATURE = 293.15  # K, uniform initial/ambient temperature field, used as T_eq


# ------------------------------------------------------------------------------------------------ data and POD
def load_dataset():
    """Returns (t, ds): the common time grid and the xarray Dataset with all 25 simulations (batch order of the file)."""
    ds = xr.open_dataset(DATA_PATH, engine="h5netcdf")
    t = jnp.asarray(ds.time.values)
    return t, ds


def fit_pods(ds, train_ids, n_modes_state, n_modes_input):
    """Fit the temperature (state) and heat-source (input) POD bases from the training simulations only.

    Returns modes_T (n_node, n_modes_state), modes_q (n_node, n_modes_input) and x_eq (n_modes_state,), the latent equilibrium offset (ambient temperature encoded through modes_T).
    """
    T_train = ds.temperature.isel(batch=train_ids).values.reshape(-1, ds.sizes["node"])
    q_train = ds.heat_source.isel(batch=train_ids).values.reshape(-1, ds.sizes["node"])

    modes_T = fit_pod(jnp.asarray(T_train), n_modes_state)
    modes_q = fit_pod(jnp.asarray(q_train), n_modes_input)

    T_eq = jnp.full((ds.sizes["node"],), ROOM_TEMPERATURE)
    x_eq = shift_equilibrium_pod(T_eq, modes_T)

    return modes_T, modes_q, x_eq


def encode_batch(ds, batch_ids, modes_T, modes_q, x_eq):
    """Encode full-order temperature and heat-source fields into latent coordinates (state shifted by the equilibrium).

    Returns x0s (n_batch, n_modes_state), us (n_batch, n_time, n_modes_input) and xs (n_batch, n_time, n_modes_state), the trajectory-fitting targets.
    """
    T = jnp.asarray(ds.temperature.isel(batch=batch_ids).values)  # (n_batch, n_time, n_node)
    q = jnp.asarray(ds.heat_source.isel(batch=batch_ids).values)

    xs = encode_pod(T, modes_T) - x_eq
    us = encode_pod(q, modes_q)
    x0s = xs[:, 0, :]

    return x0s, us, xs


def decode_state(xs, modes_T, x_eq):
    """Latent state trajectories -> full-order temperature fields."""
    return decode_pod(xs + x_eq, modes_T)


# ------------------------------------------------------------------------------------------------ derivative targets
def differentiate_latent_trajectories(t, xs):
    """Time derivatives of POD trajectories (targets of the derivative-fitting stage).

    A backward-difference cubic Hermite interpolant is constructed separately for every trajectory and evaluated through Diffrax's derivative API at every recorded time, including both endpoints.
    ``xs`` has shape ``(..., n_time, n_state)``.
    """
    t = jnp.asarray(t)
    xs = jnp.asarray(xs)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be one-dimensional with at least two samples.")
    if xs.ndim < 2 or xs.shape[-2] != t.size:
        raise ValueError(
            f"xs must have shape (..., {t.size}, n_state); got {xs.shape}."
        )
    if not bool(jnp.all(jnp.diff(t) > 0)):
        raise ValueError("t must be strictly increasing.")

    leading_shape = xs.shape[:-2]
    flat_xs = xs.reshape((-1, xs.shape[-2], xs.shape[-1]))

    def differentiate_one(trajectory):
        coeffs = diffrax.backward_hermite_coefficients(t, trajectory)
        interpolation = diffrax.CubicInterpolation(t, coeffs)
        return jax.vmap(interpolation.derivative)(t)

    derivatives = jax.vmap(differentiate_one)(flat_xs)
    return derivatives.reshape((*leading_shape, xs.shape[-2], xs.shape[-1]))
