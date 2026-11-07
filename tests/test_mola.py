"""Tests for src/mola.py -- mathematical invariants of the MoLA mixture
moments: synthetic jnp arrays standing in for component predictions, no real Laplace_ODESolver/ODE solve
involved (that machinery is exercised by the chicken-dataset scripts
already; these tests are about the mixture ARITHMETIC being correct, not
about re-testing Laplace_ODESolver itself).

A minimal fake stands in for Laplace_ODESolver: MoLA_ODESolver only ever
calls `.mean_linearized(t, x0, us)`, `.std_linearized(t, x0, us[, cov_type])`,
and reads `.sigma_noise_opt` off each component (see src/mola.py), so a fake
implementing exactly that surface is sufficient and keeps these tests fast
and independent of the ODE-solver libraries. The fake's `cov_type="dense"` path returns a
DIAGONAL matrix only (good enough for shape/plumbing checks) -- the real,
generally non-diagonal dense-covariance math is exercised separately in
tests/test_mola_dense_cov.py against a real Laplace_ODESolver.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import jax.scipy.stats as jstats
import pytest

from src.mola import MoLA_ODESolver
from src.ensembles import ens_trajectory_mean, ens_trajectory_var


def gaussian_nll(y_true, mean, std, min_std=1e-6):
    """Mean negative log-density of y_true under N(mean, std^2) per element."""
    return -jnp.mean(jstats.norm.logpdf(y_true, loc=mean, scale=jnp.maximum(std, min_std)))


class FakeLaplaceComponent:
    """Stands in for Laplace_ODESolver: fixed mean/std trajectories and a
    fixed calibrated sigma_noise, ignoring (t_eval, x0_eval, us_eval) --
    sufficient since MoLA_ODESolver never inspects those beyond forwarding
    them to the component calls being faked here.

    std_linearized supports cov_type="marginal" (default, returns the fixed
    std) and cov_type="dense" (returns diag(std**2) flattened to (n, n),
    n = T*state_size) -- a DIAGONAL fake dense covariance, sufficient for
    shape/plumbing tests but NOT a substitute for exercising the real
    (generally non-diagonal) dense covariance math, which is covered
    separately in tests/test_mola_dense_cov.py against a real
    Laplace_ODESolver (see that file's module docstring for why)."""

    def __init__(self, mean, std, sigma_noise=0.0):
        self._mean = mean
        self._std = std
        self.sigma_noise_opt = jnp.asarray(sigma_noise)

    def mean_linearized(self, t_eval, x0_eval, us_eval=None):
        return self._mean

    def std_linearized(self, t_eval, x0_eval, us_eval=None, cov_type="marginal"):
        if cov_type == "dense":
            var_flat = (self._std ** 2).reshape(-1)
            return jnp.diag(var_flat)
        return self._std


T, STATE_SIZE = 5, 2


def make_components(key, K, mean_scale=1.0, std_scale=0.3, sigma_noise=0.0):
    mean_key, std_key = jrd.split(key)
    means = mean_scale * jrd.normal(mean_key, (K, T, STATE_SIZE))
    stds = std_scale * jnp.abs(jrd.normal(std_key, (K, T, STATE_SIZE)))
    return [FakeLaplaceComponent(means[k], stds[k], sigma_noise) for k in range(K)], means, stds


## ============================================================ ##
## 1. Mixture weights sum to one
## ============================================================ ##


def test_weights_sum_to_one():
    components, _, _ = make_components(jrd.key(0), K=4)
    mola = MoLA_ODESolver(components)
    assert jnp.isclose(jnp.sum(mola.weights), 1.0)
    assert mola.weights.shape == (4,)


## ============================================================ ##
## 2. K = 1 reduces to the single component exactly
## ============================================================ ##


def test_k_equals_one_reduces_to_single_laplace():
    components, means, stds = make_components(jrd.key(1), K=1)
    mola = MoLA_ODESolver(components)

    mean = mola.mean(None, None)
    var = mola.var(None, None)
    parts = mola.variance_components(None, None)

    assert jnp.allclose(mean, means[0])
    assert jnp.allclose(var, stds[0] ** 2)
    assert jnp.allclose(parts["between"], 0.0)


## ============================================================ ##
## 3. Identical component means => var_between = 0
## ============================================================ ##


def test_identical_means_zero_between_variance():
    key = jrd.key(2)
    shared_mean = jrd.normal(key, (T, STATE_SIZE))
    stds = 0.3 * jnp.abs(jrd.normal(jrd.key(3), (3, T, STATE_SIZE)))
    components = [FakeLaplaceComponent(shared_mean, stds[k]) for k in range(3)]
    mola = MoLA_ODESolver(components)

    parts = mola.variance_components(None, None)
    assert jnp.allclose(parts["between"], 0.0, atol=1e-6)
    assert jnp.allclose(parts["within"], jnp.mean(stds ** 2, axis=0))
    assert jnp.allclose(parts["total"], parts["within"])


## ============================================================ ##
## 4. All local Laplace variances zero => MoLA reduces to ensemble uncertainty
## ============================================================ ##


def test_zero_local_variance_reduces_to_ensemble_variance():
    components, means, _ = make_components(jrd.key(4), K=6, std_scale=0.0)
    mola = MoLA_ODESolver(components)

    parts = mola.variance_components(None, None)
    assert jnp.allclose(parts["within"], 0.0)
    assert jnp.allclose(parts["total"], parts["between"])
    # Directly the population variance of the component means.
    assert jnp.allclose(parts["between"], jnp.var(means, axis=0))


## ============================================================ ##
## 5. Uniform weights + same MAP models => mean_MoLA = mean_deep_ensemble
## ============================================================ ##


def test_mean_matches_deep_ensemble_mean():
    components, means, _ = make_components(jrd.key(5), K=5)
    mola = MoLA_ODESolver(components)

    mola_mean = mola.mean(None, None)
    ensemble_mean = jnp.mean(means, axis=0)
    assert jnp.allclose(mola_mean, ensemble_mean)


def test_between_variance_matches_ensemble_variance_convention():
    """var_between must use the SAME ddof=0 (population) convention as
    src.ensembles' jnp.var(..., axis=...) -- see src/mola.py's module
    docstring on why no ddof rescaling is needed here."""
    components, means, _ = make_components(jrd.key(6), K=7)
    mola = MoLA_ODESolver(components)

    parts = mola.variance_components(None, None)
    ensemble_var_of_means = jnp.var(means, axis=0)  # population variance, ddof=0
    assert jnp.allclose(parts["between"], ensemble_var_of_means)


class LinearEnsembleMember(eqx.Module):
    """Deterministic per-member trajectory w * ones(T, STATE_SIZE) + x0 --
    just enough of an eqx.Module to be eqx.filter_vmap-batchable into the
    same shape src.ensembles' own docstring describes, without needing a
    real dynax.ODESolver model."""

    w: jax.Array

    def __call__(self, t, x0):
        return self.w * jnp.ones((T, STATE_SIZE)) + x0


def test_mola_mean_and_between_variance_match_real_ensemble_helpers():
    """End-to-end check against the actual src.ensembles helpers (not just
    reimplementing jnp.mean/jnp.var by hand), using a batched eqx.Module
    ensemble built the same way ensembles.py's own docstrings describe."""
    ws = jrd.normal(jrd.key(7), (4,))
    ensemble_module = eqx.filter_vmap(LinearEnsembleMember)(ws)

    x0 = jnp.zeros(STATE_SIZE)
    ens_mean = ens_trajectory_mean(ensemble_module, None, x0)
    ens_var = ens_trajectory_var(ensemble_module, None, x0)

    members = [LinearEnsembleMember(w) for w in ws]
    fake_components = [
        FakeLaplaceComponent(m(None, x0), jnp.zeros((T, STATE_SIZE))) for m in members
    ]
    mola = MoLA_ODESolver(fake_components)

    assert jnp.allclose(mola.mean(None, x0), ens_mean)
    assert jnp.allclose(mola.variance_components(None, x0)["between"], ens_var)


## ============================================================ ##
## 6. var_total >= 0 (numerically, up to float tolerance)
## ============================================================ ##


def test_var_total_nonnegative():
    components, _, _ = make_components(jrd.key(8), K=5, mean_scale=3.0, std_scale=0.5)
    mola = MoLA_ODESolver(components)
    var_total = mola.var(None, None)
    assert jnp.all(var_total >= -1e-10)

    std_total = mola.std(None, None)
    assert jnp.all(jnp.isfinite(std_total))
    assert jnp.all(std_total >= 0.0)


## ============================================================ ##
## 7. Shapes: single IC, batched ICs (via a Python loop -- see src/mola.py's
## docstring on why Laplace_ODESolver is never vmapped), full trajectories,
## multiple state dimensions.
## ============================================================ ##


@pytest.mark.parametrize("state_size", [1, 2, 4])
def test_shapes_single_initial_condition(state_size):
    key = jrd.key(9)
    means = jrd.normal(key, (3, T, state_size))
    stds = jnp.abs(jrd.normal(jrd.key(10), (3, T, state_size)))
    components = [FakeLaplaceComponent(means[k], stds[k]) for k in range(3)]
    mola = MoLA_ODESolver(components)

    assert mola.mean(None, None).shape == (T, state_size)
    assert mola.var(None, None).shape == (T, state_size)
    assert mola.std(None, None).shape == (T, state_size)
    parts = mola.variance_components(None, None)
    for v in parts.values():
        assert v.shape == (T, state_size)


def test_shapes_batched_initial_conditions_via_loop():
    """MoLA_ODESolver (like Laplace_ODESolver) takes ONE (x0, us) pair per
    call -- batched initial conditions are handled by a Python loop
    over trajectories, not a vmap over MoLA itself."""
    n_trajectories = 3
    key = jrd.key(11)
    all_means, all_stds = [], []
    for i in range(n_trajectories):
        k1, k2 = jrd.split(jrd.fold_in(key, i))
        all_means.append(jrd.normal(k1, (4, T, STATE_SIZE)))
        all_stds.append(jnp.abs(jrd.normal(k2, (4, T, STATE_SIZE))))

    means_out = []
    for i in range(n_trajectories):
        components = [
            FakeLaplaceComponent(all_means[i][k], all_stds[i][k]) for k in range(4)
        ]
        mola = MoLA_ODESolver(components)
        means_out.append(mola.mean(None, None))
    means_out = jnp.stack(means_out)
    assert means_out.shape == (n_trajectories, T, STATE_SIZE)


def test_dense_cov_shape_and_diagonal_matches_var():
    components, means, stds = make_components(jrd.key(12), K=4)
    mola = MoLA_ODESolver(components)

    cov = mola.dense_cov(None, None)
    n = T * STATE_SIZE
    assert cov.shape == (n, n)

    var_total = mola.var(None, None).reshape(-1)
    assert jnp.allclose(jnp.diag(cov), var_total, atol=1e-5)


## ============================================================ ##
## Predictive (noise-inclusive) variance stays separate from epistemic
## ============================================================ ##


def test_predictive_var_adds_mean_sigma_noise_squared():
    components, _, _ = make_components(jrd.key(13), K=3, sigma_noise=0.0)
    # Overwrite sigma_noise_opt per component to distinct nonzero values.
    noise_stds = jnp.array([0.1, 0.2, 0.3])
    for c, s in zip(components, noise_stds):
        c.sigma_noise_opt = s
    mola = MoLA_ODESolver(components)

    epistemic = mola.var(None, None)
    predictive = mola.predictive_var(None, None)
    expected_extra = jnp.mean(noise_stds ** 2)
    assert jnp.allclose(predictive - epistemic, expected_extra)


## ============================================================ ##
## Mixture log-likelihood: NOT equal to a single moment-matched Gaussian NLL
## ============================================================ ##


def test_mixture_nll_differs_from_single_gaussian_nll_for_multimodal_means():
    """With well-separated component means and small per-component
    variance, the true mixture density is strongly bimodal -- collapsing it
    to N(mu_MoLA, var_MoLA) would over/under-estimate the density at a point
    near one mode, so the two NLLs must disagree here."""
    mean_a = jnp.full((T, STATE_SIZE), -5.0)
    mean_b = jnp.full((T, STATE_SIZE), 5.0)
    small_std = jnp.full((T, STATE_SIZE), 0.1)
    components = [
        FakeLaplaceComponent(mean_a, small_std, sigma_noise=0.0),
        FakeLaplaceComponent(mean_b, small_std, sigma_noise=0.0),
    ]
    mola = MoLA_ODESolver(components)

    y_true = mean_a  # sits exactly on one mode
    mixture_nll = mola.nll(y_true, None, None, predictive=False)

    single_gaussian_nll = gaussian_nll(y_true, mola.mean(None, None), mola.std(None, None))

    assert not jnp.allclose(mixture_nll, single_gaussian_nll, atol=1e-2)
    # The true mixture should assign much higher density (lower NLL) to a
    # point sitting exactly on one of its modes than the blurred single
    # Gaussian centered between the two modes does.
    assert mixture_nll < single_gaussian_nll


def test_mixture_nll_reduces_to_single_gaussian_nll_for_k_equals_one():
    components, means, stds = make_components(jrd.key(14), K=1)
    mola = MoLA_ODESolver(components)
    y_true = means[0] + 0.05

    mixture_nll = mola.nll(y_true, None, None, predictive=False)
    single_nll = gaussian_nll(y_true, means[0], stds[0])
    assert jnp.allclose(mixture_nll, single_nll, atol=1e-5)


def test_mixture_nll_finite_when_local_variance_is_exactly_zero():
    """Regression test: Laplace_ODESolver.std_linearized returns EXACTLY 0
    at points with no epistemic uncertainty (e.g. t=0, pinned to x0
    regardless of theta) -- log_predictive_density/nll must not divide by
    zero or take log(0) there (see the min_std guard in src/mola.py)."""
    zero_std = jnp.zeros((T, STATE_SIZE))
    means = jrd.normal(jrd.key(15), (3, T, STATE_SIZE))
    components = [FakeLaplaceComponent(means[k], zero_std) for k in range(3)]
    mola = MoLA_ODESolver(components)

    y_true = means[0]
    for predictive in (False, True):
        nll = mola.nll(y_true, None, None, predictive=predictive)
        assert jnp.isfinite(nll), f"nll not finite for predictive={predictive}: {nll}"
