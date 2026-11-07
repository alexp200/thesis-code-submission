"""Computational building blocks of the DED active-learning runs: model builders, POD-latent data preparation, the two-stage training, the training-RMSE filter and the exact Laplace scoring of the pool.

  - State and input are POD coordinates (N_MODES = 40). The POD is fitted anew for every training set (prepare_iteration_data) and never changes inside a training run.
  - Models: NODE and sPHNN as dynax solvers (build_node_member, build_sphnn_member), collected in MODEL_CLASSES.
  - Training: derivative fitting on the latent trajectories, then trajectory fine-tuning with an L2 penalty on the Laplace parameters (train_hybrid_from_scratch).
  - Errors are reported in decoded Kelvin over all 25 simulations (see la_de_runner/reporting.py), a finite-design-space metric, not a held-out error.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import klax
import dynax

HERE = Path(__file__).resolve().parent
# core/ (HERE) -> kernels/ -> 04_ded/ ->
# experiments/ -> repo root. ded_data.py and ded_split.py are
# colocated in THIS folder.
REPO_ROOT = HERE.parents[3]
for _p in (REPO_ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import optax

from src import models as md
from src import active_learning as al
from src import regularization as reg
from src.ensembles import member_train_rmse, filter_members_by_rmse
from ded_data import fit_pods, encode_batch, differentiate_latent_trajectories
from ded_split import EVALUATION_IDS


# ------------------------------------------------------------------------------------------------ model builders

N_MODES = 40  # POD latent dimension of state and input
SMALL_INIT = jax.nn.initializers.variance_scaling(scale=0.01, mode="fan_in", distribution="truncated_normal")
TINY_INIT = jax.nn.initializers.variance_scaling(scale=1e-4, mode="fan_in", distribution="truncated_normal")


def build_node_member(key):
    """Untrained NODE (MLP vector field, two softplus layers of width 16)."""
    node_ode = dynax.NeuralODE(
        state_size=N_MODES, input_size=N_MODES, time_dependent=False,
        width_sizes=[16, 16], activation=jax.nn.softplus, weight_init=SMALL_INIT, key=key,
    )
    return dynax.ODESolver(node_ode, max_steps=262_144)


def build_sphnn_member(key):
    """Untrained sPHNN (FICNN Hamiltonian, equilibrium fixed at the latent origin)."""
    ficnn_key, argmin_key, j_key, r_key, g_key = jrd.split(key, 5)
    x_eq_latent = jnp.zeros(N_MODES)
    hamiltonian = dynax.ConvexNormalization(
        dynax.NoArgsPotential(md.softplus_FICNN(
            klax.nn.FICNN(in_size=N_MODES, out_size="scalar", width_sizes=[16, 16], weight_init=TINY_INIT, key=ficnn_key)
        )),
        state_size=N_MODES,
        argmin_init=lambda key, shape, dtype: x_eq_latent.astype(dtype),
        argmin_learnable=False, epsilon=1e-6, key=argmin_key,
    )
    structure_matrix = klax.nn.ConstantSkewSymmetricMatrix(shape=N_MODES, init=TINY_INIT, key=j_key)
    dissipation_matrix = klax.nn.ConstantSPDMatrix(shape=N_MODES, init=TINY_INIT, key=r_key)
    input_matrix = klax.nn.ConstantMatrix(shape=(N_MODES, N_MODES), init=TINY_INIT, key=g_key)
    phs = dynax.ISOPHS(
        hamiltonian=hamiltonian,
        structure_matrix=dynax.MatrixWrapper(structure_matrix),
        dissipation_matrix=dynax.MatrixWrapper(dissipation_matrix),
        input_matrix=dynax.MatrixWrapper(input_matrix),
    )
    return dynax.ODESolver(phs, max_steps=262_144)


@dataclass(frozen=True)
class ModelClassConfig:
    name: str  # "NODE" or "sPHNN"
    member_builder: object  # key -> untrained dynax.ODESolver
    laplace_cls: type  # md.Laplace_NeuralODESolver or md.Laplace_sPHNNSolver
    finalize_before_save: bool = True  # False for sPHNN (keeps the parameter wrappers, see Laplace_sPHNNSolver.check_model_wrapping)


MODEL_CLASSES = {
    "NODE": ModelClassConfig(name="NODE", member_builder=build_node_member, laplace_cls=md.Laplace_NeuralODESolver),
    "sPHNN": ModelClassConfig(name="sPHNN", member_builder=build_sphnn_member,
                              laplace_cls=md.Laplace_sPHNNSolver, finalize_before_save=False),
}


def callable_member(cfg: ModelClassConfig, member):
    """Callable form of a member: sPHNN members are finalized (wrappers resolved), NODE members are already callable. Never finalize before handing a member to cfg.laplace_cls, which needs the wrappers."""
    return klax.finalize(member) if not cfg.finalize_before_save else member


# ------------------------------------------------------------------------------------------------ data preparation


from experiments.shared.cluster_recovery.runtime import release_after_phase

@release_after_phase
def prepare_iteration_data(ds, t, train_ids: list[int]) -> dict:
    """POD bases from the training ids only, then training and evaluation (all 25 simulations) data encoded through them; returns everything a training, scoring or evaluation step needs."""
    from experiments.shared.cluster_recovery.pod_cache import load_or_fit
    from ded_data import DATA_PATH
    modes_T, modes_q, x_eq = load_or_fit(
        ds, train_ids, N_MODES, fitter=fit_pods, dataset_path=DATA_PATH,
        cache_dir=REPO_ROOT / "experiments/04_ded/kernels/core/pod_bases",
        implementation_paths=(
            REPO_ROOT / "src/preprocessing.py",
            Path(__file__).with_name("ded_data.py"),
        ),
    )

    x0s_train, us_train, xs_train = encode_batch(ds, train_ids, modes_T, modes_q, x_eq)
    x0s_evaluation, us_evaluation, xs_evaluation = encode_batch(
        ds, EVALUATION_IDS, modes_T, modes_q, x_eq
    )
    T_true_evaluation = jnp.asarray(ds.temperature.isel(batch=EVALUATION_IDS).values)
    return {
        "t": t, "train_ids": list(train_ids),
        "modes_T": modes_T, "modes_q": modes_q, "x_eq": x_eq,
        "x0s_train": x0s_train, "us_train": us_train, "xs_train": xs_train,
        "x0s_evaluation": x0s_evaluation,
        "us_evaluation": us_evaluation,
        "xs_evaluation": xs_evaluation,
        "T_true_evaluation": T_true_evaluation,
    }


def pool_arrays(ds, pool_ids: list[int], modes_T, modes_q, x_eq) -> tuple[jax.Array, jax.Array]:
    """Initial states and inputs of the pool candidates, encoded through the POD bases of the current training set."""
    x0_pool, us_pool, _xs_pool = encode_batch(ds, pool_ids, modes_T, modes_q, x_eq)
    return x0_pool, us_pool


# ------------------------------------------------------------------------------------------------ ensembles and training


def build_ensemble(member_builder, seeds):
    """Untrained ensemble with a leading member axis (vmapped over the seeds)."""
    keys = jax.vmap(jrd.key)(jnp.asarray(seeds))
    return eqx.filter_vmap(member_builder)(keys)


def split_ensemble(ensemble, n_members: int) -> list:
    """List of the individual members of a vmapped ensemble."""
    return [
        jax.tree_util.tree_map(lambda x, i=i: x[i] if eqx.is_array(x) else x, ensemble)
        for i in range(n_members)
    ]


# ------------------------------------------------------------------------------------------------ two-stage training (derivative -> trajectory)
# NODE and sPHNN use the same two stages; the schedules differ and are passed in by the caller. The Laplace/GGN is built from the final trajectory objective at the MAP point;
# derivative fitting is an unregularised initialisation only.


def _derivative_mse_loss(model, batch_data, run_state):
    """Derivative-fitting loss: direct vector-field evaluation (no ODE
    solve) against Hermite-interpolated derivative targets."""
    del run_state
    states, controls, targets = batch_data
    predictions = jax.vmap(lambda state, control: model.func(None, state, u=control))(states, controls)
    return jnp.mean((predictions - targets) ** 2)


_derivative_mse_loss = klax.loss(_derivative_mse_loss)


@release_after_phase
def train_hybrid_stages(
        cfg: ModelClassConfig, data: dict, ensemble_size: int,
        derivative_steps: int, derivative_lr: float,
        finetune_steps: int, finetune_lr: float,
        derivative_batch_size: int, finetune_l2_regularization: float,
        key: jax.Array,
        ) -> tuple[object, object, object, float, float]:
    """Run both stages; returns the ensemble, the histories of both stages and their wall times. Stage 2 starts with a fresh Adam state because learning rate and objective differ from stage 1."""
    stage1_key, stage2_key = jrd.split(key)

    # --- Stage 1: derivative fitting ---
    ensemble_key, train_key = jrd.split(stage1_key)
    seeds = jrd.randint(ensemble_key, (ensemble_size,), 0, 2**31 - 1)
    ensemble_init = build_ensemble(cfg.member_builder, seeds)

    derivatives = differentiate_latent_trajectories(data["t"], data["xs_train"])
    states = data["xs_train"].reshape(-1, N_MODES)
    controls = data["us_train"].reshape(-1, N_MODES)
    targets = derivatives.reshape(-1, N_MODES)

    start1 = time.perf_counter()
    ensemble_stage1, _history1 = klax.fit(
        ensemble_init, (states, controls, targets), batch_axes=(0, 0, 0),
        batch_size=derivative_batch_size, steps=derivative_steps, loss=_derivative_mse_loss,
        log_every=max(derivative_steps // 10, 1), key=train_key,
        optimizer=optax.adam(derivative_lr), vmap_ensemble=True,
    )
    jax.block_until_ready(ensemble_stage1)
    stage1_seconds = time.perf_counter() - start1

    # --- Stage 2: trajectory-fitting fine-tune, fresh optimizer state ---
    @klax.loss
    def trajectory_mse_loss(model, batch_data, run_state):
        ts_, x0s_, us_, xs_ = batch_data
        xs_pred = jax.vmap(model, in_axes=(None, 0, 0))(ts_, x0s_, us_)
        mse = jnp.mean((xs_ - xs_pred) ** 2)
        penalty = reg.active_parameter_squared_norm(model, cfg.name)
        return mse + 0.5 * finetune_l2_regularization * penalty

    train_data = (data["t"], data["x0s_train"], data["us_train"], data["xs_train"])
    n_train = data["x0s_train"].shape[0]

    start2 = time.perf_counter()
    ensemble_stage2, history2 = klax.fit(
        ensemble_stage1, train_data, batch_axes=(None, 0, 0, 0), batch_size=n_train,
        steps=finetune_steps, loss=trajectory_mse_loss, log_every=max(finetune_steps // 10, 1),
        key=stage2_key, optimizer=optax.adam(finetune_lr), vmap_ensemble=True,
    )
    jax.block_until_ready(ensemble_stage2)
    stage2_seconds = time.perf_counter() - start2

    return ensemble_stage2, _history1, history2, stage1_seconds, stage2_seconds


def train_hybrid_from_scratch(
        cfg: ModelClassConfig, data: dict, ensemble_size: int,
        derivative_steps: int, derivative_lr: float,
        finetune_steps: int, finetune_lr: float,
        derivative_batch_size: int, finetune_l2_regularization: float,
        key: jax.Array,
        ) -> tuple[object, object, float]:
    """Both stages; returns the final ensemble, the history of the trajectory stage and the total training time."""
    ensemble, _history1, history2, stage1_seconds, stage2_seconds = (
        train_hybrid_stages(
            cfg,
            data,
            ensemble_size,
            derivative_steps,
            derivative_lr,
            finetune_steps,
            finetune_lr,
            derivative_batch_size,
            finetune_l2_regularization,
            key,
        )
    )
    return ensemble, history2, stage1_seconds + stage2_seconds


def train_ensemble_for_model_class(
        cfg: ModelClassConfig, data: dict, ensemble_size: int, key: jax.Array,
        *, node_derivative_steps: int, node_derivative_lr: float,
        node_finetune_steps: int, node_finetune_lr: float,
        sphnn_derivative_steps: int, sphnn_derivative_lr: float,
        sphnn_finetune_steps: int, sphnn_finetune_lr: float,
        trajectory_l2_regularization: float,
        derivative_batch_size: int = 128,
        ) -> tuple[object, object, float]:
    """Select the schedule of the architecture and train."""
    if cfg.name == "NODE":
        derivative_steps, derivative_lr = node_derivative_steps, node_derivative_lr
        finetune_steps, finetune_lr = node_finetune_steps, node_finetune_lr
    elif cfg.name == "sPHNN":
        derivative_steps, derivative_lr = sphnn_derivative_steps, sphnn_derivative_lr
        finetune_steps, finetune_lr = sphnn_finetune_steps, sphnn_finetune_lr
    else:
        raise ValueError(f"Unknown model class: {cfg.name!r}")

    return train_hybrid_from_scratch(
        cfg,
        data,
        ensemble_size,
        derivative_steps,
        derivative_lr,
        finetune_steps,
        finetune_lr,
        derivative_batch_size,
        trajectory_l2_regularization,
        key,
    )


# ------------------------------------------------------------------------------------------------ training-RMSE filter


@release_after_phase
def filter_converged_members(cfg: ModelClassConfig, data: dict, ensemble, ensemble_size: int,
                              context: str) -> tuple[list, list[int], list[float], float | None, float]:
    """Keep the members whose training RMSE (latent space, what the loss optimised) is not above 5 x the median (src.ensembles).
    Returns raw members, callable members, dropped indices, their RMSE, the cutoff (None if nothing was dropped) and the time."""
    start = time.perf_counter()
    raw_members = split_ensemble(ensemble, ensemble_size)
    callable_members = [callable_member(cfg, m) for m in raw_members]

    train_rmse = member_train_rmse(callable_members, data["t"], data["x0s_train"], data["xs_train"],
                                    us_train=data["us_train"])
    kept_callable, report = filter_members_by_rmse(callable_members, train_rmse)
    keep_mask = report["keep_mask"]
    kept_raw = [m for i, m in enumerate(raw_members) if bool(keep_mask[i])]

    if len(kept_callable) < ensemble_size:
        print(f"  [{context}] dropped {ensemble_size - len(kept_callable)} non-converged member(s): "
              f"{report['dropped_indices']} (train RMSE {report['dropped_rmse']}, cutoff={report['cutoff']:.4g})")
    if len(kept_callable) == 0:
        raise RuntimeError(f"{context}: every member was filtered out as non-converged; "
                            "check training steps/hyperparameters before continuing.")

    cutoff = report["cutoff"] if report["dropped_indices"] else None
    filter_seconds = time.perf_counter() - start
    return kept_raw, kept_callable, report["dropped_indices"], [float(r) for r in report["dropped_rmse"]], cutoff, filter_seconds


@release_after_phase
def acquisition_train_rmse(cfg: ModelClassConfig, data: dict, raw_member) -> float:
    """Training RMSE of the single acquisition member; raises if it is not finite."""
    member = callable_member(cfg, raw_member)
    rmse = member_train_rmse(
        [member],
        data["t"],
        data["x0s_train"],
        data["xs_train"],
        us_train=data["us_train"],
    )[0]
    value = float(rmse)
    if not bool(jnp.isfinite(rmse)):
        raise RuntimeError(
            f"{cfg.name} acquisition member has non-finite train RMSE."
        )
    return value


# ------------------------------------------------------------------------------------------------ Laplace approximation and pool scoring


def _block_component_ready(component) -> None:
    """Wait for the asynchronous Laplace quantities."""
    jax.block_until_ready((component.eigvals_ggn, component.eigvecs_ggn,
                            component.prior_prec_opt, component.sigma_noise_opt))


@release_after_phase
def fit_single_seed_laplace(cfg: ModelClassConfig, data: dict, raw_member) -> tuple[object, float]:
    """Fit one Laplace component (GGN and marginal-likelihood calibration) on a single trained member; returns it with the fit time."""
    start = time.perf_counter()
    component = cfg.laplace_cls(raw_member, data["t"], data["x0s_train"], data["xs_train"], us_train=data["us_train"])
    component.fit_laplace_mll()
    _block_component_ready(component)
    posterior_scalars = jnp.asarray(
        [component.prior_prec_opt, component.sigma_noise_opt]
    )
    if not bool(jnp.all(jnp.isfinite(posterior_scalars))):
        raise RuntimeError("Laplace calibration produced non-finite hyperparameters.")
    if not bool(jnp.all(posterior_scalars > 0.0)):
        raise RuntimeError("Laplace calibration produced non-positive hyperparameters.")
    laplace_fit_seconds = time.perf_counter() - start
    return component, laplace_fit_seconds


SCORE_DIRECTION_CHUNK_SIZE = 128


@release_after_phase
def score_pool_sum_marginal(
        component,
        t,
        x0_pool,
        us_pool,
        chunk_size: int = SCORE_DIRECTION_CHUNK_SIZE,
        ) -> jax.Array:
    """MGE score (sum of log marginal variances) of every pool candidate from the exact linearized Laplace variance.

    Candidates are scored one after another with one compiled scorer; the posterior eigendirections are processed in chunks, so neither the trajectory Jacobian nor ``J @ U`` is materialised.
    """
    if us_pool is None:
        raise ValueError("us_pool is required for the forced DED models.")

    variance_scorer = component.make_marginal_variance_eigen_jvp(
        t,
        chunk_size=chunk_size,
    )
    scores = []
    for x0, us in zip(x0_pool, us_pool):
        variance = variance_scorer(x0, us)
        scores.append(
            al.score_sum_marginal(
                variance.reshape(-1),
                sigma_noise=0.0,
            )
        )
    return jnp.stack(scores)


def ranking_from_scores(scores: jax.Array, pool_ids: list[int]) -> list[dict]:
    """Pool ids sorted by descending score."""
    pairs = sorted(zip(pool_ids, (float(s) for s in scores)), key=lambda kv: kv[1], reverse=True)
    return [{"batch_id": bid, "score": s} for bid, s in pairs]
