"""Training recipe of the thermal food processing AL experiments: model skeletons, trajectory loss and the vmapped ensemble training.

  - NODE: MLP vector field (two softplus layers of width 16). sPHNN: FICNN Hamiltonian (width 16, 16) with the equilibrium fixed at the origin and constant learned structure, dissipation and input
    matrices. Both start from small weights (NODE scale 1e-2, sPHNN scale 1e-4).
  - Loss: trajectory MSE on the normalised probe temperatures plus 0.5 * lambda * ||theta||^2 on the parameters that the corresponding Laplace class treats as trainable (for the sPHNN this
    excludes the fixed equilibrium); the same L2 convention as in the harmonic-oscillator experiments.
  - Training: constant Adam learning rate, full batch; all members of an ensemble are trained in ONE klax.fit call (vmap_ensemble=True), so they see the identical data at every step.
"""

import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrd
import klax
import optax

import dynax
from src.models import softplus_FICNN

STATE_SIZE, INPUT_SIZE = 2, 1
SMALL_INIT = jax.nn.initializers.variance_scaling(scale=0.01, mode="fan_in", distribution="truncated_normal")
TINY_INIT = jax.nn.initializers.variance_scaling(scale=1e-4, mode="fan_in", distribution="truncated_normal")


# ------------------------------------------------------------------------------------------------ model skeletons
def node_skeleton_from_key(key, state_size=STATE_SIZE):
    """Untrained NODE with the given key."""
    node_ode = dynax.NeuralODE(
        state_size=state_size, input_size=INPUT_SIZE, time_dependent=False,
        width_sizes=[16, 16], activation=jax.nn.softplus, weight_init=SMALL_INIT, key=key,
    )
    return dynax.ODESolver(node_ode, max_steps=262_144)


def sphnn_skeleton_from_key(key, state_size=STATE_SIZE):
    """Untrained sPHNN with the given key (equilibrium fixed at the origin)."""
    x_equilibrium = jnp.zeros(state_size)
    ficnn_key, argmin_key, j_key, r_key, g_key = jrd.split(key, 5)
    hamiltonian = dynax.ConvexNormalization(
        dynax.NoArgsPotential(softplus_FICNN(
            klax.nn.FICNN(in_size=state_size, out_size="scalar", width_sizes=[16, 16],
                          weight_init=TINY_INIT, key=ficnn_key)
        )),
        state_size=state_size,
        argmin_init=lambda key, shape, dtype: x_equilibrium.astype(dtype),
        argmin_learnable=False,
        epsilon=1e-6,
        key=argmin_key,
    )
    structure_matrix = klax.nn.ConstantSkewSymmetricMatrix(shape=state_size, init=TINY_INIT, key=j_key)
    dissipation_matrix = klax.nn.ConstantSPDMatrix(shape=state_size, init=TINY_INIT, key=r_key)
    input_matrix = klax.nn.ConstantMatrix(shape=(state_size, INPUT_SIZE), init=TINY_INIT, key=g_key)
    phs = dynax.ISOPHS(
        hamiltonian=hamiltonian,
        structure_matrix=dynax.MatrixWrapper(structure_matrix),
        dissipation_matrix=dynax.MatrixWrapper(dissipation_matrix),
        input_matrix=dynax.MatrixWrapper(input_matrix),
    )
    return dynax.ODESolver(phs, max_steps=262_144)


SKELETON_FROM_KEY = {"NODE": node_skeleton_from_key, "sPHNN": sphnn_skeleton_from_key}


# ------------------------------------------------------------------------------------------------ loss
def active_parameter_tree(model, model_class: str):
    """The parameter subtree penalised by L2 and used by the corresponding Laplace class (sPHNN: without the klax.NonTrainable / klax.Constraint wrapper leaves, i.e. the fixed equilibrium)."""
    if model_class == "NODE":
        return eqx.filter(model, eqx.is_array)

    wrapper_types = (klax.NonTrainable, klax.Constraint)
    is_wrapper = lambda leaf: isinstance(leaf, wrapper_types)
    filter_spec = jax.tree.map(
        lambda leaf: eqx.is_array(leaf) and not is_wrapper(leaf),
        model,
        is_leaf=is_wrapper,
    )
    params, _ = eqx.partition(model, filter_spec, is_leaf=is_wrapper)
    return params


def active_parameter_squared_norm(model, model_class: str) -> jax.Array:
    """Squared norm of the penalised parameters."""
    leaves = [
        leaf
        for leaf in jax.tree.leaves(active_parameter_tree(model, model_class))
        if eqx.is_array(leaf)
    ]
    return sum((jnp.sum(leaf**2) for leaf in leaves), start=jnp.asarray(0.0))


class TrajectoryMSEL2(klax.Loss):
    """Trajectory MSE over the probes plus 0.5 * lambda * ||theta||^2 on the penalised parameters."""

    def __init__(self, arch: str, l2_regularization: float, n_probes: int):
        self.arch = arch
        self.l2_regularization = float(l2_regularization)
        self.n_probes = n_probes

    def __call__(self, model, batch, run_state):
        del run_state
        ts, x0s, us, ys = batch
        callable_model = klax.unwrap(model)
        ys_pred = jax.vmap(callable_model, in_axes=(None, 0, 0))(ts, x0s, us)
        mse = jnp.mean((ys - ys_pred[..., :self.n_probes]) ** 2)
        penalty = active_parameter_squared_norm(model, self.arch)
        return mse + 0.5 * self.l2_regularization * penalty

    def value(self, model, batch, run_state):
        return self(model, batch, run_state)


# ------------------------------------------------------------------------------------------------ training
def train_ensemble(arch, seeds, data, steps, lr, key_offset, log_every, l2_regularization):
    """Train len(seeds) independent models of `arch` in one vmapped klax.fit call (full batch). Returns (ensemble, history); the ensemble has a leading (n_seeds,) axis on every array leaf."""
    keys = jnp.stack([jrd.key(s + key_offset) for s in seeds])
    ensemble = eqx.filter_vmap(lambda k: SKELETON_FROM_KEY[arch](k, state_size=STATE_SIZE))(keys)
    loss = TrajectoryMSEL2(arch, l2_regularization, data["n_probes"])
    train_data = (data["t"], data["x0s_train"], data["us_train"], data["ys_train"])
    ensemble, history = klax.fit(
        ensemble, train_data, batch_axes=(None, 0, 0, 0), batch_size=data["n_train"],
        steps=steps, loss=loss, log_every=log_every,
        vmap_ensemble=True, key=jrd.key(1 + key_offset), optimizer=optax.adam(lr),
    )
    return ensemble, history


def split_ensemble(ensemble, n_seeds):
    """Split a vmapped ensemble into a list of n_seeds individual models."""
    return [jax.tree_util.tree_map(lambda x: x[i] if eqx.is_array(x) else x, ensemble)
            for i in range(n_seeds)]
