"""Tests for the exact matrix-free Laplace pushforward."""

import diffrax
import dynax
import jax
import jax.numpy as jnp
import jax.random as jrd
import klax

from src import active_learning as al
from src import laplace as lp
from src import models as production_models

jax.config.update("jax_enable_x64", True)


def test_chunked_jvp_matches_explicit_jacobian():
    key_matrix, key_directions = jrd.split(jrd.key(0))

    output_dimension = 7
    parameter_dimension = 11

    matrix = jrd.normal(
        key_matrix,
        (output_dimension, parameter_dimension),
    )

    directions, _ = jnp.linalg.qr(
        jrd.normal(
            key_directions,
            (parameter_dimension, parameter_dimension),
        )
    )

    weights = jnp.linspace(0.1, 1.1, parameter_dimension)
    p_map = jnp.linspace(-0.2, 0.3, parameter_dimension)

    def function(p):
        return matrix @ p

    explicit_jacobian = jax.jacobian(function)(p_map)

    expected_variance = (
        (explicit_jacobian @ directions) ** 2
    ) @ weights

    jvp_variance = lp.chunked_eigen_jvp_marginal_variance(
        function,
        p_map,
        directions,
        weights,
        chunk_size=4,
    )

    assert jvp_variance.shape == (output_dimension,)
    assert jnp.allclose(
        jvp_variance,
        expected_variance,
        rtol=1e-10,
        atol=1e-10,
    )


def test_ded_pool_scorer_uses_reusable_production_jvp_api(monkeypatch):
    from pathlib import Path

    primary_dir = (
        Path(__file__).resolve().parents[1]
        / "experiments/04_ded/kernels/core"
    )
    monkeypatch.syspath_prepend(str(primary_dir))
    import _ded_al_core as ded_core

    calls = []

    class FakeComponent:
        def make_marginal_variance_eigen_jvp(self, t_eval, *, chunk_size):
            calls.append((t_eval, chunk_size))

            def scorer(x0_eval, us_eval):
                return jnp.concatenate([
                    x0_eval**2 + 1.0,
                    us_eval.reshape(-1)[:2] ** 2 + 1.0,
                ])

            return scorer

    t = jnp.array([0.0, 1.0])
    x0_pool = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    us_pool = jnp.array([
        [[0.5], [1.5]],
        [[2.5], [3.5]],
    ])

    scores = ded_core.score_pool_sum_marginal(
        FakeComponent(),
        t,
        x0_pool,
        us_pool,
    )
    expected = jnp.stack([
        al.score_sum_marginal(
            jnp.concatenate([
                x0**2 + 1.0,
                us.reshape(-1)[:2] ** 2 + 1.0,
            ])
        )
        for x0, us in zip(x0_pool, us_pool)
    ])

    assert len(calls) == 1
    assert calls[0][0] is t
    assert calls[0][1] == ded_core.SCORE_DIRECTION_CHUNK_SIZE == 128
    assert jnp.allclose(scores, expected)


def test_node_jvp_matches_existing_full_jacobian_path():
    node = dynax.NeuralODE(
        state_size=2,
        input_size=0,
        time_dependent=False,
        width_sizes=[2],
        activation=jax.nn.tanh,
        key=jrd.key(10),
    )

    solver = dynax.ODESolver(
        node,
        max_steps=1024,
    )

    t = jnp.array([0.0, 0.01, 0.02])
    x0_train = jnp.array([
        [0.10, -0.05],
        [-0.08, 0.04],
    ])
    y_train = jnp.zeros((2, len(t), 2))

    component = production_models.Laplace_NeuralODESolver(
        solver,
        t,
        x0_train,
        y_train,
    )

    # Prepare a deterministic posterior without running MLL optimization.
    component._prepare_eigenbasis()
    component.prior_prec_opt = jnp.asarray(0.8)
    component.sigma_noise_opt = jnp.asarray(0.4)
    component.active_idx = jnp.arange(component.D)

    x0_eval = x0_train[0]

    assert isinstance(
        component.map_model.adjoint,
        diffrax.RecursiveCheckpointAdjoint,
    )

    reference_variance = (
        component.std_linearized(
            t,
            x0_eval,
        )
        ** 2
    )

    jvp_variance = component.marginal_variance_eigen_jvp(
        t,
        x0_eval,
        chunk_size=3,
    )

    # The JVP path must not mutate the component's production adjoint.
    assert isinstance(
        component.map_model.adjoint,
        diffrax.RecursiveCheckpointAdjoint,
    )

    assert jvp_variance.shape == reference_variance.shape
    assert jnp.allclose(
        jvp_variance,
        reference_variance,
        rtol=2e-4,
        atol=2e-6,
    )

    reusable_scorer = component.make_marginal_variance_eigen_jvp(
        t,
        chunk_size=3,
    )

    reusable_variance = reusable_scorer(x0_eval)

    assert jnp.allclose(
        reusable_variance,
        reference_variance,
        rtol=2e-4,
        atol=2e-6,
    )


def test_forced_node_jvp_matches_existing_full_jacobian_path():
    node = dynax.NeuralODE(
        state_size=2,
        input_size=1,
        time_dependent=False,
        width_sizes=[2],
        activation=jax.nn.tanh,
        key=jrd.key(20),
    )

    solver = dynax.ODESolver(
        node,
        max_steps=1024,
    )

    t = jnp.array([0.0, 0.01, 0.02])

    x0_train = jnp.array([
        [0.10, -0.05],
        [-0.08, 0.04],
    ])

    us_train = jnp.array([
        [[0.0], [0.1], [0.2]],
        [[0.1], [0.0], [-0.1]],
    ])

    y_train = jnp.zeros((2, len(t), 2))

    component = production_models.Laplace_NeuralODESolver(
        solver,
        t,
        x0_train,
        y_train,
        us_train=us_train,
    )

    component._prepare_eigenbasis()
    component.prior_prec_opt = jnp.asarray(0.8)
    component.sigma_noise_opt = jnp.asarray(0.4)
    component.active_idx = jnp.arange(component.D)

    x0_eval = x0_train[0]
    us_eval = us_train[0]

    assert component.is_forced

    reference_variance = (
        component.std_linearized(
            t,
            x0_eval,
            us_eval,
        )
        ** 2
    )

    direct_jvp_variance = component.marginal_variance_eigen_jvp(
        t,
        x0_eval,
        us_eval,
        chunk_size=3,
    )

    reusable_scorer = component.make_marginal_variance_eigen_jvp(
        t,
        chunk_size=3,
    )

    reusable_variance = reusable_scorer(
        x0_eval,
        us_eval,
    )

    assert jnp.allclose(
        direct_jvp_variance,
        reference_variance,
        rtol=2e-4,
        atol=2e-6,
    )

    assert jnp.allclose(
        reusable_variance,
        reference_variance,
        rtol=2e-4,
        atol=2e-6,
    )

    assert isinstance(
        component.map_model.adjoint,
        diffrax.RecursiveCheckpointAdjoint,
    )

    reference_scores = []
    jvp_scores = []

    for candidate_x0, candidate_us in zip(x0_train, us_train):
        candidate_reference_variance = (
            component.std_linearized(
                t,
                candidate_x0,
                candidate_us,
            )
            ** 2
        )

        candidate_jvp_variance = reusable_scorer(
            candidate_x0,
            candidate_us,
        )

        reference_scores.append(
            al.score_sum_marginal(
                candidate_reference_variance.reshape(-1),
                sigma_noise=0.0,
            )
        )

        jvp_scores.append(
            al.score_sum_marginal(
                candidate_jvp_variance.reshape(-1),
                sigma_noise=0.0,
            )
        )

    reference_scores = jnp.stack(reference_scores)
    jvp_scores = jnp.stack(jvp_scores)

    assert jnp.allclose(
        jvp_scores,
        reference_scores,
        rtol=2e-5,
        atol=2e-5,
    )

    assert jnp.argmax(jvp_scores) == jnp.argmax(reference_scores)

    pool_scores = jnp.stack([
        al.score_sum_marginal(
            reusable_scorer(candidate_x0, candidate_us).reshape(-1),
            sigma_noise=0.0,
        )
        for candidate_x0, candidate_us in zip(x0_train, us_train)
    ])

    assert pool_scores.shape == reference_scores.shape

    assert jnp.allclose(
        pool_scores,
        reference_scores,
        rtol=2e-5,
        atol=2e-5,
    )

    assert jnp.argmax(pool_scores) == jnp.argmax(reference_scores)


def test_forced_sphnn_jvp_matches_full_jacobian():
    state_size = 2

    ficnn_key, argmin_key, j_key, r_key, g_key = jrd.split(
        jrd.key(30),
        5,
    )

    equilibrium = jnp.zeros(state_size)

    hamiltonian = dynax.ConvexNormalization(
        dynax.NoArgsPotential(
            production_models.softplus_FICNN(
                klax.nn.FICNN(
                    in_size=state_size,
                    out_size="scalar",
                    width_sizes=[2],
                    key=ficnn_key,
                )
            )
        ),
        state_size=state_size,
        argmin_init=lambda key, shape, dtype: equilibrium.astype(dtype),
        argmin_learnable=False,
        epsilon=1e-6,
        key=argmin_key,
    )

    sphnn = dynax.ISOPHS(
        hamiltonian=hamiltonian,
        structure_matrix=dynax.MatrixWrapper(
            klax.nn.ConstantSkewSymmetricMatrix(
                shape=state_size,
                key=j_key,
            )
        ),
        dissipation_matrix=dynax.MatrixWrapper(
            klax.nn.ConstantSPDMatrix(
                shape=state_size,
                key=r_key,
            )
        ),
        input_matrix=dynax.MatrixWrapper(
            klax.nn.ConstantMatrix(
                shape=(state_size, 1),
                key=g_key,
            )
        ),
    )

    solver = dynax.ODESolver(
        sphnn,
        max_steps=1024,
    )

    t = jnp.array([0.0, 0.01, 0.02])
    x0_train = jnp.array([
        [0.10, -0.05],
        [-0.08, 0.04],
    ])
    us_train = jnp.array([
        [[0.0], [0.1], [0.2]],
        [[0.1], [0.0], [-0.1]],
    ])
    y_train = jnp.zeros((2, len(t), state_size))

    component = production_models.Laplace_sPHNNSolver(
        solver,
        t,
        x0_train,
        y_train,
        us_train=us_train,
    )

    component._prepare_eigenbasis()
    component.prior_prec_opt = jnp.asarray(0.8)
    component.sigma_noise_opt = jnp.asarray(0.4)
    component.active_idx = jnp.arange(component.D)


    x0_eval = x0_train[0]
    us_eval = us_train[0]

    reference_variance = (
        component.std_linearized(t, x0_eval, us_eval) ** 2
    )

    direct_jvp_variance = component.marginal_variance_eigen_jvp(
        t,
        x0_eval,
        us_eval,
        chunk_size=3,
    )

    reusable_scorer = component.make_marginal_variance_eigen_jvp(
        t,
        chunk_size=3,
    )
    reusable_variance = reusable_scorer(x0_eval, us_eval)

    assert jnp.allclose(
        direct_jvp_variance,
        reference_variance,
        rtol=2e-4,
        atol=2e-6,
    )
    assert jnp.allclose(
        reusable_variance,
        reference_variance,
        rtol=2e-4,
        atol=2e-6,
    )

    assert isinstance(
        component.map_model.adjoint,
        diffrax.RecursiveCheckpointAdjoint,
    )
