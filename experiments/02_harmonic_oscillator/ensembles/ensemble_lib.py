"""Shared setup of the harmonic-oscillator experiments (ensembles of NODE and conservative sPHNN under the UQ methods): data sets, training and SWAG settings, checkpoint and result paths,
the conservative sPHNN builder, the variance of the linearized Laplace pushforward, the first return to the Poincare section and the per-period envelope.

Data sets, the NODE builder and the loss come from oscillator_lib.py (imported as `base`).
The sPHNN built here is conservative: the dissipation matrix R and the input matrix G are structurally absent
(dynax.ISOPHS(dissipation_matrix=None, input_matrix=None)).

Everything runs in float64; the flag below must be set before any model is built (GGN and Laplace calibration are ill-conditioned in float32).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import dynax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrd
import klax
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import models as md

import oscillator_lib as base  # (same folder; scripts that import this module have it on sys.path)

# --- fixed experimental settings -------------------------------------------------------------------------------
DATASETS = ("lower_half", "left_half")
MODEL_CLASSES = ("NODE", "sPHNN")
DATASET_SEED = 17          # base.build_dataset(seed=17)
L2 = 1e-4                  # objective: MSE + 0.5 * L2 * ||theta||^2
STEPS = 5000
LEARNING_RATE = 1e-3
TRAINING_KEY = 0
MODEL_KEY_OFFSETS = {"NODE": 0, "sPHNN": 100}
DATASET_KEY_OFFSETS = {"lower_half": 0, "left_half": 1}
K_ENSEMBLE = 10            # ensemble members (seeds 0..9)
SEEDS = tuple(range(K_ENSEMBLE))
N_PERIODS = 15
N_EVAL = 3000              # evaluation grid density

# SWAG settings (Table B.8): SGD continuation with mini-batches of 4 of the 16 training trajectories at the learning rate SWAG_LR of the comparisons; swag_dependence/ varies the rate.
SWAG_LR = 0.1
SWAG_KW = dict(swag_steps=200, swag_lr=SWAG_LR, swag_freq=5, swag_optimizer="sgd", clip_grad_norm=1.0, batch_size=4)
SWAG_MC_SAMPLES = 64
SWAG_SAMPLE_EPS = 1e-14
# fit_swag's penalty is l2 * ||p||^2 (no 1/2); the MAP objective is
# 0.5 * L2 * ||theta||^2, so the SWAG continuation must use 0.5 * L2 to keep
# optimizing the SAME objective the MAP point is a mode of.
SWAG_L2 = 0.5 * L2
SWAG_MAX_ABS_VALUE = 1e6   # SWAG_ODESolver.predict's own default bounded-sample filter

CHECKPOINT_DIR = HERE / "checkpoints"
TRAINING_DIR = HERE / "training"
RESULT_DIR = Path(os.environ["HO_RESULT_DIR"]) if os.environ.get("HO_RESULT_DIR") else HERE.parent / "long_horizon" / "results"      # a Level-2 recomputation can write to (and cache in) another directory

SPRING_CONSTANT = base.SPRING_CONSTANT
MASS = base.MASS


def true_period() -> float:
    """Period of the true oscillator, 2 pi / sqrt(k / m)."""
    return float(2.0 * jnp.pi / jnp.sqrt(SPRING_CONSTANT / MASS))


def build_dataset(name: str) -> dict:
    """Noisy training data of `lower_half` or `left_half` (same generator and seed as in the Ott-style check)."""
    return base.build_dataset(name, seed=DATASET_SEED)


def horizon() -> float:
    """Length of the long-horizon rollouts: N_PERIODS true periods."""
    return N_PERIODS * true_period()


def envelope(t, y, T=None, n=N_PERIODS):
    """Per period k = 0..n-1 (window [kT, (k+1)T], T = true period): the maximum of y and the time at which it occurs."""
    T = true_period() if T is None else T
    tk, ek = [], []
    for k in range(n):
        m = (t >= k * T) & (t <= (k + 1) * T + 1e-12)
        i = np.argmax(np.where(m, y, -np.inf))
        tk.append(t[i])
        ek.append(y[i])
    return np.array(tk), np.array(ek)


# --- conservative sPHNN ------------------------------------------------------


def build_sphnn_conservative_member(key):
    """Conservative sPHNN: H = 0.5 x^T A x + FICNN residual with ConvexNormalization at x* = 0 and a constant learned skew-symmetric J;
    the dissipation matrix R and the input matrix G are structurally absent (None)."""
    ficnn_key, argmin_key, structure_key, _unused_dissipation_key, spd_key = jrd.split(key, 5)
    equilibrium = jnp.zeros(base.STATE_SIZE)
    potential = base.QuadraticPlusConvexResidual(
        spd_matrix=klax.nn.ConstantSPDMatrix(shape=base.STATE_SIZE, init=base.TINY_INIT, key=spd_key),
        residual=md.softplus_FICNN(
            klax.nn.FICNN(
                in_size=base.STATE_SIZE,
                out_size="scalar",
                width_sizes=base.SPHNN_FICNN_WIDTH_SIZES,
                weight_init=base.TINY_INIT,
                key=ficnn_key,
            )
        ),
    )
    hamiltonian = dynax.ConvexNormalization(
        dynax.NoArgsPotential(potential),
        state_size=base.STATE_SIZE,
        argmin_init=lambda key, shape, dtype: equilibrium.astype(dtype),
        argmin_learnable=False,
        epsilon=1e-6,
        key=argmin_key,
    )
    structure = klax.nn.ConstantSkewSymmetricMatrix(shape=base.STATE_SIZE, init=base.TINY_INIT, key=structure_key)
    vector_field = dynax.ISOPHS(
        hamiltonian=hamiltonian,
        structure_matrix=dynax.MatrixWrapper(structure),
        dissipation_matrix=None,
        input_matrix=None,
    )
    return dynax.ODESolver(vector_field, max_steps=base.MAX_STEPS)


def build_member(model_class: str, key):
    """One untrained ensemble member of the given model class."""
    if model_class == "NODE":
        return base.build_node_member(key)
    if model_class == "sPHNN":
        return build_sphnn_conservative_member(key)
    raise ValueError(model_class)


def build_ensemble(model_class: str, seeds):
    """Vmapped ensemble of untrained members, one per seed."""
    keys = jax.vmap(jrd.key)(jnp.asarray(list(seeds)))
    return eqx.filter_vmap(lambda key: build_member(model_class, key))(keys)


def split_ensemble(ensemble, n_members: int):
    """List of the individual members of a vmapped ensemble."""
    return base.split_ensemble(ensemble, n_members)


def callable_member(model_class: str, member):
    """Member that can be called as model(t, x0) (sPHNN: constraints applied)."""
    return base.callable_member(model_class, member)


def laplace_class(model_class: str):
    """Laplace wrapper class of the model class."""
    return md.Laplace_NeuralODESolver if model_class == "NODE" else md.Laplace_sPHNNSolver


def swag_class(model_class: str):
    """SWAG wrapper class of the model class."""
    return md.SWAG_NeuralODESolver if model_class == "NODE" else md.SWAG_sPHNNSolver


def checkpoint_path(model_class: str, dataset: str) -> Path:
    """Checkpoint file of the trained K = 10 ensemble."""
    return CHECKPOINT_DIR / f"{model_class}_{dataset}_K{K_ENSEMBLE}_l2_1em04.eqx"


def load_members(model_class: str, dataset: str) -> list:
    """The K trained members of a stored ensemble."""
    skeleton = build_ensemble(model_class, SEEDS)
    ensemble = eqx.tree_deserialise_leaves(checkpoint_path(model_class, dataset), skeleton)
    return split_ensemble(ensemble, K_ENSEMBLE)


# --- solver helpers ----------------------------------------------------------


def lla_variance(la, t_eval, x0):
    """Marginal variance (T, 2) of the linearized pushforward, diag(J Sigma J^T) with J = d x(t) / d theta and Sigma = U diag(w) U^T of the calibrated Laplace posterior."""
    J = jax.jacobian(lambda p: la.f_solve_flat_general(p, t_eval, x0))(la.p_map)  # (T*2, D)
    directions, weights = la._posterior_directions_and_weights()
    proj = (J @ directions).reshape(len(t_eval), 2, -1)
    return jnp.einsum("tak,k,tak->ta", proj, weights, proj)


def vector_field(model, x):
    """dx/dt of a callable dynax.ODESolver at state x (unforced)."""
    return model.func(0.0, x)


def return_time(model, x0, t_guess: float, n_grid: int = 4000, newton_iters: int = 8) -> float:
    """First return time of the orbit started at x0 to the half-line
    {s * x0 : s > 0} (a Poincare section through the origin, which lies
    inside every closed orbit around the minimum of a convex H).
    Linear-interpolated sign change on a dense grid, then Newton-refined with
    the vector field. For an exactly periodic orbit this is the period.
    """
    c = jnp.array([-x0[1], x0[0]])
    ts = jnp.linspace(0.0, 1.6 * t_guess, n_grid)
    xs = model(ts, x0)
    s = xs @ c
    side = xs @ x0
    cross = (s[:-1] * s[1:] < 0) & (side[:-1] > 0) & (ts[:-1] > 0.3 * t_guess)
    if not bool(jnp.any(cross)):
        raise RuntimeError("no return to the Poincare section found")
    i = int(jnp.argmax(cross))
    t = ts[i] - s[i] * (ts[i + 1] - ts[i]) / (s[i + 1] - s[i])
    for _ in range(newton_iters):
        x = model(jnp.array([0.0, t]), x0)[-1]
        sdot = c @ vector_field(model, x)
        t = t - (c @ x) / sdot
    return float(t)


