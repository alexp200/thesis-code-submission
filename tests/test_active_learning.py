import jax.numpy as jnp
import pytest

from src import active_learning as al


def test_sum_marginal_variance_floor_prevents_negative_infinity():
    var = jnp.array([0.0, 1.0])

    score = al.score_sum_marginal(var, variance_floor=1e-6)

    assert jnp.isfinite(score)
    assert float(score) == pytest.approx(float(jnp.log(1e-6) + jnp.log(1.0)))


def test_sum_marginal_variance_floor_clips_small_negative_variance():
    var = jnp.array([-1e-14, 2.0])

    score = al.score_sum_marginal(var, variance_floor=1e-6)

    assert jnp.isfinite(score)
    assert float(score) == pytest.approx(float(jnp.log(1e-6) + jnp.log(2.0)))


# ---------------------------------------------------------------------
# score_sum_variance (Lundby-style sum of raw marginal variances)
# ---------------------------------------------------------------------


def test_sum_variance_is_exact_sum_of_variance_vector():
    var = jnp.array([0.5, 1.25, 3.0])

    score = al.score_sum_variance(var)

    assert float(score) == pytest.approx(0.5 + 1.25 + 3.0)


def test_sum_variance_adds_sigma_noise_squared_per_coordinate():
    var = jnp.array([1.0, 2.0])
    sigma_noise = 0.5

    score = al.score_sum_variance(var, sigma_noise=sigma_noise)

    assert float(score) == pytest.approx(1.0 + 2.0 + 2 * sigma_noise**2)


def test_sum_variance_matches_zero_sigma_noise_default_used_by_chicken_experiments():
    var = jnp.array([1.0, 2.0, 3.0])

    score_default = al.score_sum_variance(var)
    score_explicit_zero = al.score_sum_variance(var, sigma_noise=0.0)

    assert float(score_default) == pytest.approx(float(score_explicit_zero))
    assert float(score_default) == pytest.approx(6.0)


def test_sum_variance_clips_tiny_negative_variance_to_zero():
    var = jnp.array([-1e-14, 2.0])

    score = al.score_sum_variance(var, variance_floor=1e-6)

    assert jnp.isfinite(score)
    assert float(score) == pytest.approx(2.0)


def test_sum_variance_zero_variance_contributes_nothing():
    var = jnp.array([0.0, 0.0, 5.0])

    score = al.score_sum_variance(var)

    assert float(score) == pytest.approx(5.0)


def test_sum_variance_does_not_silently_clip_materially_negative_variance():
    # A coordinate far more negative than variance_floor signals a broken
    # upstream covariance estimate, not floating-point noise -- it must NOT
    # be silently floored to zero like score_sum_marginal's safe_var does.
    var = jnp.array([-1.0, 2.0])

    score = al.score_sum_variance(var, variance_floor=1e-6)

    assert float(score) == pytest.approx(-1.0 + 2.0)


def test_sum_variance_and_sum_marginal_can_rank_candidates_differently():
    # Candidate A: one dominant high-variance coordinate, rest near zero.
    # Candidate B: several moderately-uncertain coordinates.
    # sum(var) favors the concentrated outlier (A); sum(log(var)) favors
    # the more evenly spread candidate (B), since log is concave and
    # therefore penalizes a single large value less than it rewards many
    # coordinates simultaneously being non-negligible.
    var_a = jnp.array([100.0, 1e-6, 1e-6, 1e-6])
    var_b = jnp.array([0.5, 0.5, 0.5, 0.5])

    sum_variance_a = al.score_sum_variance(var_a)
    sum_variance_b = al.score_sum_variance(var_b)
    sum_marginal_a = al.score_sum_marginal(var_a)
    sum_marginal_b = al.score_sum_marginal(var_b)

    # sum_variance ranks A above B ...
    assert sum_variance_a > sum_variance_b
    # ... but sum_marginal (log-sum) ranks B above A: the criteria disagree.
    assert sum_marginal_b > sum_marginal_a
