"""Tests for src/mola.py's dense-covariance path (`dense_cov`, `predictive_dense_cov`,
`_component_covs`). Unlike tests/test_mola.py (which fakes
Laplace_ODESolver with plain arrays -- adequate for the mixture ARITHMETIC,
but its fake dense covariance is diagonal-only, see that file's docstring),
this module exercises the REAL src.models.Laplace_ODESolver.std_linearized(
cov_type="dense") code path -- faking a genuinely non-diagonal dense
covariance would risk testing a reimplementation instead of the real math
`MoLA_ODESolver.dense_cov` is built on. Kept fast by a tiny (2-hidden-unit, few-training-point)
unforced dynax NeuralODE.
"""

import jax
import jax.numpy as jnp
import jax.random as jrd
import dynax

from src import models as md
from src.mola import MoLA_ODESolver

STATE_SIZE = 2
N_TRAIN = 3
N_TIMESTEPS = 5


def _build_map_model(seed):
    key = jrd.key(seed)
    node = dynax.NeuralODE(
        state_size=STATE_SIZE, time_dependent=False, width_sizes=[3],
        activation=jax.nn.tanh, key=key,
    )
    return dynax.ODESolver(node, max_steps=4096)


def _build_calibrated_components(K, seed0=0):
    """K independent, already-CALIBRATED tiny Laplace_NeuralODESolver
    components sharing the SAME synthetic training data. fit_laplace_mll()
    is required before std_linearized (marginal or dense) is callable."""
    t_train = jnp.linspace(0.0, 0.5, N_TIMESTEPS)
    key_data = jrd.key(1234)
    x0_train = jrd.normal(key_data, (N_TRAIN, STATE_SIZE)) * 0.1
    y_train = jrd.normal(jrd.fold_in(key_data, 1), (N_TRAIN, N_TIMESTEPS, STATE_SIZE)) * 0.1

    components = []
    for k in range(K):
        map_model = _build_map_model(seed0 + k)
        LA = md.Laplace_NeuralODESolver(map_model, t_train, x0_train, y_train)
        LA.fit_laplace_mll()
        components.append(LA)
    return components


def _eval_grid():
    """Short/cheap eval horizon, distinct from the training grid. Built
    fresh INSIDE each test (not as a module-level constant) so it always
    picks up whatever jax_enable_x64 state is ambient at TEST-RUN time --
    like every array in this file it is constructed inside a function body. A module-level constant here would
    be built at IMPORT/collection time, which can predate another test
    file (e.g. a test that loads an experiment script which
    sets jax.config.update("jax_enable_x64", True) as a side effect) later
    flipping that global JAX config mid-session -- mixing a stale-dtype
    module constant with runtime-constructed float64 arrays inside
    _build_calibrated_components is exactly what triggers a dtype-mismatch
    inside diffeqsolve. See this repo's test suite: JAX's enable_x64 flag
    is process-global and does not reset between test files.
    """
    t_eval = jnp.linspace(0.0, 0.3, 4)
    x0_eval = jnp.array([0.05, -0.02])
    return t_eval, x0_eval


## ============================================================ ##
## 1. dense_cov() diagonal matches var() -- now via the EXACT dense path
## ============================================================ ##


def test_dense_cov_diagonal_matches_var_with_real_components():
    t_eval, x0_eval = _eval_grid()
    components = _build_calibrated_components(K=3)
    mola = MoLA_ODESolver(components)

    Sigma = mola.dense_cov(t_eval, x0_eval)
    var_total = mola.var(t_eval, x0_eval).reshape(-1)

    n = len(t_eval) * STATE_SIZE
    assert Sigma.shape == (n, n)
    assert jnp.allclose(jnp.diag(Sigma), var_total, atol=1e-5)


## ============================================================ ##
## 2. Within-mode term is genuinely non-diagonal (the actual bug being fixed)
## ============================================================ ##


def test_cov_within_mode_term_has_nonzero_off_diagonal_entries():
    """Regression test for the pre-fix behavior: `dense_cov()` used to build its
    within-mode term as diag(mean_k(var_LA,k)) -- purely diagonal, silently
    dropping every component's own cross-timestep/cross-state correlations.
    A real linearized-ODE-solve Jacobian pushforward is NOT diagonal in
    general (adjacent timesteps of the same trajectory are correlated), so
    mean_k(Sigma_k) (via std_linearized(cov_type="dense")) must have
    nonzero off-diagonal entries here."""
    t_eval, x0_eval = _eval_grid()
    components = _build_calibrated_components(K=3)
    mola = MoLA_ODESolver(components)

    covs = mola._component_covs(t_eval, x0_eval)  # (K, n, n)
    within = jnp.mean(covs, axis=0)
    n = within.shape[0]
    off_diag_mask = ~jnp.eye(n, dtype=bool)
    off_diag_max_abs = jnp.max(jnp.abs(within[off_diag_mask]))

    assert off_diag_max_abs > 1e-8, (
        "Expected genuinely non-diagonal within-mode covariance from a real "
        "linearized ODE-solve Jacobian pushforward, got near-zero off-diagonal "
        f"entries (max |off-diag| = {off_diag_max_abs})."
    )


def test_dense_cov_total_equals_within_plus_between():
    t_eval, x0_eval = _eval_grid()
    components = _build_calibrated_components(K=3)
    mola = MoLA_ODESolver(components)

    covs = mola._component_covs(t_eval, x0_eval)
    mus = mola._component_means(t_eval, x0_eval)

    within = jnp.mean(covs, axis=0)
    mus_flat = mus.reshape(mola.K, -1)
    between = jnp.cov(mus_flat, rowvar=False, ddof=0)

    Sigma = mola.dense_cov(t_eval, x0_eval)
    assert jnp.allclose(Sigma, within + between, atol=1e-6)


## ============================================================ ##
## 3. K = 1: dense_cov() reduces exactly to the single component's own dense Sigma
## ============================================================ ##


def test_dense_cov_k_equals_one_reduces_to_single_component():
    t_eval, x0_eval = _eval_grid()
    components = _build_calibrated_components(K=1)
    mola = MoLA_ODESolver(components)

    Sigma_mola = mola.dense_cov(t_eval, x0_eval)
    Sigma_single = components[0].std_linearized(t_eval, x0_eval, cov_type="dense")

    assert jnp.allclose(Sigma_mola, Sigma_single, atol=1e-6)


## ============================================================ ##
## 4. predictive_dense_cov: epistemic + noise on the DIAGONAL only
## ============================================================ ##


def test_predictive_dense_cov_adds_noise_only_on_diagonal():
    t_eval, x0_eval = _eval_grid()
    components = _build_calibrated_components(K=3)
    mola = MoLA_ODESolver(components)

    Sigma_epi = mola.dense_cov(t_eval, x0_eval)
    Sigma_pred = mola.predictive_dense_cov(t_eval, x0_eval)
    n = Sigma_epi.shape[0]

    diff = Sigma_pred - Sigma_epi
    expected_diag_add = mola._sigma_noise2_mean()

    off_diag_mask = ~jnp.eye(n, dtype=bool)
    assert jnp.allclose(diff[off_diag_mask], 0.0, atol=1e-8), (
        "predictive_dense_cov must only change the DIAGONAL relative to dense_cov() -- "
        "observation noise is i.i.d. across time/state, never correlated."
    )
    assert jnp.allclose(jnp.diag(diff), expected_diag_add, atol=1e-6)


def test_predictive_dense_cov_diagonal_matches_predictive_var():
    t_eval, x0_eval = _eval_grid()
    components = _build_calibrated_components(K=3)
    mola = MoLA_ODESolver(components)

    Sigma_pred = mola.predictive_dense_cov(t_eval, x0_eval)
    predictive_var = mola.predictive_var(t_eval, x0_eval).reshape(-1)

    assert jnp.allclose(jnp.diag(Sigma_pred), predictive_var, atol=1e-5)


def test_dense_cov_and_predictive_dense_cov_stay_decoupled():
    """cov() must never be contaminated by observation noise, and calling
    predictive_dense_cov() must not mutate/cache anything that changes a
    subsequent cov() call -- the two are meant to be fully independent
    entry points (see src/mola.py's module docstring on why they are
    separate methods, not a flag)."""
    t_eval, x0_eval = _eval_grid()
    components = _build_calibrated_components(K=2)
    mola = MoLA_ODESolver(components)

    Sigma_epi_before = mola.dense_cov(t_eval, x0_eval)
    _ = mola.predictive_dense_cov(t_eval, x0_eval)
    Sigma_epi_after = mola.dense_cov(t_eval, x0_eval)

    assert jnp.allclose(Sigma_epi_before, Sigma_epi_after, atol=1e-10)
    # And predictive_dense_cov must actually differ from dense_cov (nonzero noise).
    assert not jnp.allclose(Sigma_epi_before, mola.predictive_dense_cov(t_eval, x0_eval))
