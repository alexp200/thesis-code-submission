"""Tests for src/ensembles.py: the population-variance (ddof = 0) convention of the ensemble variances and the training-RMSE member filter
(member_train_rmse, filter_members_by_rmse). Synthetic-array / toy-callable style (as in tests/test_mola.py); no Laplace_ODESolver or dynax model involved.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd

from src.ensembles import (
    evaluate_ensemble_trajectory,
    ens_trajectory_flat_var,
    ens_trajectory_var,
    member_train_rmse,
    filter_members_by_rmse,
    RelativeThreshold,
)

T, STATE_SIZE = 4, 2


class LinearEnsembleMember(eqx.Module):
    """Deterministic per-member trajectory w * ones(T, STATE_SIZE) + x0 -- same toy member as in tests/test_mola.py, batchable by eqx.filter_vmap."""

    w: jax.Array

    def __call__(self, t, x0):
        return self.w * jnp.ones((T, STATE_SIZE)) + x0


## ============================================================ ##
## Variance convention
## ============================================================ ##


def test_ens_trajectory_flat_var_is_population_variance():
    """The M members are the atoms of a discrete mixture with weights 1/M, so the variance divides by M (ddof = 0), not by M - 1."""
    ws = jrd.normal(jrd.key(0), (5,))
    ensemble = eqx.filter_vmap(LinearEnsembleMember)(ws)
    x0 = jnp.zeros(STATE_SIZE)

    preds = evaluate_ensemble_trajectory(ensemble, None, x0)
    M = preds.shape[0]
    flat_var = ens_trajectory_flat_var(ensemble, None, x0)
    manual_population_var = jnp.var(preds.reshape(M, -1), axis=0, ddof=0)
    sample_var = jnp.var(preds.reshape(M, -1), axis=0, ddof=1)

    assert flat_var.shape == (T * STATE_SIZE,)
    assert jnp.allclose(flat_var, manual_population_var, atol=1e-5)
    assert not jnp.allclose(flat_var, sample_var, atol=1e-5)


def test_ens_trajectory_var_matches_between_variance_convention():
    """ens_trajectory_var (marginal, per time step) is ddof = 0 as well."""
    ws = jrd.normal(jrd.key(2), (6,))
    ensemble = eqx.filter_vmap(LinearEnsembleMember)(ws)
    x0 = jnp.zeros(STATE_SIZE)

    preds = evaluate_ensemble_trajectory(ensemble, None, x0)
    manual_population_var = jnp.var(preds, axis=0, ddof=0)

    assert jnp.allclose(ens_trajectory_var(ensemble, None, x0), manual_population_var)


## ============================================================ ##
## Member filtering
## ============================================================ ##


class ConstantMember:
    """Plain callable member returning a fixed trajectory; member_train_rmse only calls `member(t, x0)` or `member(t, x0, us=us)`."""

    def __init__(self, value):
        self.value = value

    def __call__(self, t, x0, us=None):
        return self.value


def test_member_train_rmse_zero_for_perfect_members():
    ys_train = jnp.zeros((3, T, STATE_SIZE))
    x0s_train = jnp.zeros((3, STATE_SIZE))
    members = [ConstantMember(jnp.zeros((T, STATE_SIZE))) for _ in range(4)]

    rmse = member_train_rmse(members, None, x0s_train, ys_train)
    assert rmse.shape == (4,)
    assert jnp.allclose(rmse, 0.0)


def test_member_train_rmse_flags_non_converged_member():
    """A member whose training loss never converged shows up as a clear training-RMSE outlier."""
    ys_train = jnp.zeros((3, T, STATE_SIZE))
    x0s_train = jnp.zeros((3, STATE_SIZE))
    converged = [ConstantMember(jnp.zeros((T, STATE_SIZE))) for _ in range(4)]
    non_converged = ConstantMember(jnp.full((T, STATE_SIZE), 50.0))
    members = converged + [non_converged]

    rmse = member_train_rmse(members, None, x0s_train, ys_train)
    assert float(rmse[-1]) > float(rmse[:-1].max()) * 10


def test_filter_members_by_rmse_relative_threshold_removes_only_pathological_member():
    """RelativeThreshold(5.0) removes a member ~100x worse than the rest."""
    members = [ConstantMember(i) for i in range(5)]
    rmse = jnp.array([1.0, 1.1, 0.9, 1.2, 100.0])  # last member is pathological

    kept, report = filter_members_by_rmse(members, rmse, threshold=RelativeThreshold(5.0))

    assert len(kept) == 4
    assert report["dropped_indices"] == [4]
    assert report["dropped_rmse"] == [100.0]
    assert bool(report["keep_mask"][4]) is False
    assert all(bool(report["keep_mask"][i]) for i in range(4))


def test_filter_members_does_not_remove_ordinary_spread():
    """Members that are merely somewhat worse than the median must not be removed: filtering does not homogenise the ensemble."""
    members = [ConstantMember(i) for i in range(5)]
    rmse = jnp.array([1.0, 1.5, 2.0, 2.5, 3.0])  # ordinary spread, no outlier

    kept, report = filter_members_by_rmse(members, rmse, threshold=RelativeThreshold(5.0))

    assert len(kept) == 5
    assert report["dropped_indices"] == []


def test_filter_members_absolute_threshold():
    members = [ConstantMember(i) for i in range(3)]
    rmse = jnp.array([1.0, 2.0, 10.0])

    kept, report = filter_members_by_rmse(members, rmse, threshold=5.0)
    assert len(kept) == 2
    assert report["cutoff"] == 5.0


def test_filter_members_default_threshold_is_conservative():
    """Without an explicit threshold the default is RelativeThreshold(5.0)."""
    members = [ConstantMember(i) for i in range(4)]
    rmse = jnp.array([1.0, 1.0, 1.0, 4.9])  # just under 5x median -> kept
    kept, report = filter_members_by_rmse(members, rmse)
    assert len(kept) == 4

    rmse_pathological = jnp.array([1.0, 1.0, 1.0, 5.1])  # just over 5x median -> dropped
    kept2, report2 = filter_members_by_rmse(members, rmse_pathological)
    assert len(kept2) == 3
