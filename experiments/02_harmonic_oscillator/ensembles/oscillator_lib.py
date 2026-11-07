"""Data and models of the harmonic-oscillator experiments.

System: undamped oscillator (mass 1, spring constant 2, state (q, p)). The training data are 16 noisy trajectories over half a period for two data sets (`lower_half`,
`left_half`).

Models:
  - NODE: MLP vector field (tanh, one hidden layer of 256 units, orthogonal weights, PyTorch-style uniform biases); the same NODE as in the Ott-style check (../ott_check).
  - sPHNN: Hamiltonian H(x) = 0.5 x^T A x + FICNN_residual(x) with a learnable SPD matrix A and a FICNN of width [32, 32] (1259 active parameters against 1282 of the NODE),
    ConvexNormalization at x* = 0, constant learned skew-symmetric J. The quadratic term supplies the unbounded derivative dH/dq = k q of the true oscillator, which a softplus
    FICNN cannot represent; the sum stays convex. The conservative variant without R and G that the experiments use is built in ensemble_lib.py.

Also here: the L2-regularised trajectory loss, the parameter count on the parameters of the Laplace classes and the training RMSE of a member.
"""

from __future__ import annotations

from dataclasses import dataclass

import dynax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import klax


STATE_SIZE = 2
MASS = 1.0
SPRING_CONSTANT = 2.0
N_TRAJECTORIES = 16
N_TIME = 50
N_EVAL = 300
OBSERVATION_NOISE_VARIANCE = 0.3
EVALUATION_T_END = 10.0
MAX_STEPS = 65_536

# initialisation of the sPHNN weights (small values)
TINY_INIT = jax.nn.initializers.variance_scaling(
    scale=1e-4,
    mode="fan_in",
    distribution="truncated_normal",
)

# NODE: orthogonal weights and PyTorch-style uniform biases, as in ../ott_check/make_ott_figure.py
NODE_WIDTH_SIZES = [256]
NODE_WEIGHT_INIT = jax.nn.initializers.orthogonal(scale=1.0)
SPHNN_FICNN_WIDTH_SIZES = [32, 32]


def torch_linear_bias_initializer(key, shape, fan_in, dtype=jnp.float64):
    """Match torch.nn.Linear's default uniform bias initialization."""
    bound = 1.0 / jnp.sqrt(jnp.asarray(fan_in, dtype=dtype))
    return jrd.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    seed_offset: int
    mean_q: float
    mean_p_factor_sqrt_k: float

    @property
    def mean_p(self) -> float:
        return self.mean_p_factor_sqrt_k * SPRING_CONSTANT**0.5


DATASET_SPECS = {
    "lower_half": DatasetSpec(
        name="lower_half",
        seed_offset=0,
        mean_q=3.0,
        mean_p_factor_sqrt_k=0.0,
    ),
    "left_half": DatasetSpec(
        name="left_half",
        seed_offset=1,
        mean_q=0.0,
        mean_p_factor_sqrt_k=3.0,
    ),
}

MODEL_CLASSES = ("NODE", "sPHNN")


def oscillator_frequency() -> float:
    """Angular frequency sqrt(k / m) of the true oscillator."""
    return float(jnp.sqrt(SPRING_CONSTANT / MASS))


def training_t_end() -> float:
    """End of the training window: half a period."""
    return float(jnp.pi / oscillator_frequency())


def true_trajectory(x0: jax.Array, t: jax.Array) -> jax.Array:
    """Analytical solution (q, p) of the oscillator from x0 at the times t."""
    q0, p0 = x0
    frequency = jnp.sqrt(SPRING_CONSTANT / MASS)
    q = q0 * jnp.cos(frequency * t) + p0 / (MASS * frequency) * jnp.sin(
        frequency * t
    )
    p = p0 * jnp.cos(frequency * t) - MASS * frequency * q0 * jnp.sin(
        frequency * t
    )
    return jnp.stack((q, p), axis=-1)


def build_dataset(name: str, *, seed: int = 17) -> dict:
    """Reproduce the corresponding Ott verification training dataset."""
    spec = DATASET_SPECS[name]
    key = jrd.fold_in(jrd.key(seed), spec.seed_offset)
    ic_key, noise_key = jrd.split(key)
    q_key, p_key = jrd.split(ic_key)

    q0 = spec.mean_q + jnp.sqrt(0.2) * jrd.normal(q_key, (N_TRAJECTORIES,))
    p0 = spec.mean_p + jnp.sqrt(0.2) * jrd.normal(p_key, (N_TRAJECTORIES,))
    x0s = jnp.stack((q0, p0), axis=-1)

    t_train = jnp.linspace(0.0, training_t_end(), N_TIME)
    ys_clean = jax.vmap(lambda x0: true_trajectory(x0, t_train))(x0s)
    noise = jnp.sqrt(OBSERVATION_NOISE_VARIANCE) * jrd.normal(
        noise_key, ys_clean.shape
    )
    return {
        "name": name,
        "t_train": t_train,
        "x0s": x0s,
        "ys_clean": ys_clean,
        "ys_noisy": ys_clean + noise,
        "representative_x0": jnp.asarray((spec.mean_q, spec.mean_p)),
    }


def build_node_member(key: jax.Array):
    """One untrained NODE member (ODE solver around the MLP vector field)."""
    bias_key, node_key = jrd.split(key)
    vector_field = dynax.NeuralODE(
        state_size=STATE_SIZE,
        input_size=None,
        time_dependent=False,
        width_sizes=NODE_WIDTH_SIZES,
        activation=jax.nn.tanh,
        weight_init=NODE_WEIGHT_INIT,
        bias_init=torch_linear_bias_initializer,
        dtype=jnp.float64,
        key=node_key,
    )
    return dynax.ODESolver(vector_field, max_steps=MAX_STEPS)


class QuadraticPlusConvexResidual(eqx.Module):
    """H_raw(x) = 0.5*x^T A x + FICNN_residual(x) -- a plain x -> Scalar
    callable, exactly what dynax.NoArgsPotential expects to wrap. A (a
    learnable SPD matrix) supplies an everywhere-unbounded-derivative
    quadratic term that no stack of softplus-FICNN layers can produce on its
    own (see common.py's module docstring); the FICNN residual then only
    needs to learn a bounded correction. The sum stays convex since both
    terms are convex -- no changes to the libraries needed."""

    spd_matrix: klax.nn.ConstantSPDMatrix
    residual: eqx.Module

    def __call__(self, x):
        A = self.spd_matrix(x)
        quadratic = 0.5 * jnp.inner(x, A @ x)
        return quadratic + self.residual(x)


def split_ensemble(ensemble, n_members: int) -> list:
    """List of the individual members of a vmapped ensemble."""
    return [
        jax.tree.map(
            lambda leaf, index=index: leaf[index] if eqx.is_array(leaf) else leaf,
            ensemble,
        )
        for index in range(n_members)
    ]


def callable_member(model_class: str, member):
    """Member that can be called as model(t, x0); the sPHNN is finalized (constraints applied)."""
    return member if model_class == "NODE" else klax.finalize(member)


def active_parameter_tree(model, model_class: str):
    """Return the parameter tree used by the corresponding Laplace class."""
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
    """Squared norm of the parameters that the Laplace class treats as trainable (the L2 penalty acts on these)."""
    leaves = [
        leaf
        for leaf in jax.tree.leaves(active_parameter_tree(model, model_class))
        if eqx.is_array(leaf)
    ]
    return sum((jnp.sum(leaf**2) for leaf in leaves), start=jnp.asarray(0.0))


def active_parameter_count(model, model_class: str) -> int:
    """Number of parameters that the Laplace class treats as trainable."""
    return sum(
        leaf.size
        for leaf in jax.tree.leaves(active_parameter_tree(model, model_class))
        if eqx.is_array(leaf)
    )


class TrajectoryMSEL2(klax.Loss):
    """Trajectory MSE plus coupled L2 on the production-Laplace parameters."""

    def __init__(self, model_class: str, l2_regularization: float):
        self.model_class = model_class
        self.l2_regularization = float(l2_regularization)

    def __call__(self, model, data, run_state):
        del run_state
        t_train, x0s, targets = data
        callable_model = klax.unwrap(model)
        predictions = jax.vmap(callable_model, in_axes=(None, 0))(t_train, x0s)
        mse = jnp.mean((predictions - targets) ** 2)
        penalty = active_parameter_squared_norm(model, self.model_class)
        return mse + 0.5 * self.l2_regularization * penalty

    def value(self, model, data, run_state):
        return self(model, data, run_state)


def representative_evaluation(dataset: dict) -> tuple[jax.Array, jax.Array]:
    """Evaluation grid (to EVALUATION_T_END) and true trajectory from the representative initial state."""
    t_eval = jnp.linspace(0.0, EVALUATION_T_END, N_EVAL)
    return t_eval, true_trajectory(dataset["representative_x0"], t_eval)




def train_rmse(model_class: str, raw_member, dataset: dict) -> dict:
    """Training RMSE of one trained member against the noise-free and the noisy targets."""
    model = callable_member(model_class, raw_member)
    predictions = jax.vmap(model, in_axes=(None, 0))(dataset["t_train"], dataset["x0s"])
    return {
        "train_rmse_clean": float(jnp.sqrt(jnp.mean((predictions - dataset["ys_clean"]) ** 2))),
        "train_rmse_noisy": float(jnp.sqrt(jnp.mean((predictions - dataset["ys_noisy"]) ** 2))),
    }
