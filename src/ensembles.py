"""Deep-ensemble utilities: predictive mean/variance of batched ensembles of ODE solvers, and the training-RMSE filter that removes members with a diverged fit."""
import equinox as eqx
import jax
import jax.numpy as jnp
import klax


# ------------------------------------------------------------------------------------------------ batched ensembles (leading member axis)
def evaluate_ensemble_trajectory(ensemble: klax.nn._mlp.MLP,
                                 ts: jnp.ndarray,
                                 x0: jnp.ndarray,
                                 us: jnp.ndarray = None
                                 ) -> jnp.ndarray:
    """Trajectories of all members of a batched ensemble, shape (M, len(ts), state_size)."""
    if us is None:
        return eqx.filter_vmap(
            lambda model, ts, x0: model(ts, x0),
            in_axes=(eqx.if_array(0), None, None),
        )(ensemble, ts, x0)

    return eqx.filter_vmap(
        lambda model, ts, x0, us: model(ts, x0, us=us),
        in_axes=(eqx.if_array(0), None, None, None),
    )(ensemble, ts, x0, us)


def ens_trajectory_mean(ensemble, ts, x0, us= None):
    """Predictive mean trajectory of an ensemble of ODESolvers, shape (len(ts), state_size)."""
    preds = evaluate_ensemble_trajectory(ensemble, ts, x0, us)
    return jnp.mean(preds, axis=0)


def ens_trajectory_var(ensemble, ts, x0, us=None, noise=None):
    """Predictive variance (ddof = 0) of an ensemble of ODESolvers, shape (len(ts), state_size); `noise` is an observation-noise standard deviation added as noise**2."""
    preds = evaluate_ensemble_trajectory(ensemble, ts, x0, us)
    var = jnp.var(preds, axis=0)
    if noise is not None:
        var = var + noise**2
    return var


def ens_trajectory_flat_var(ensemble: klax.nn._mlp.MLP,
                            ts: jnp.ndarray,
                            x0: jnp.ndarray,
                            us: jnp.ndarray = None
                            ) -> jnp.ndarray:
    """Predictive variance (ddof = 0) of the flattened trajectory, shape (len(ts) * state_size,)."""
    preds = evaluate_ensemble_trajectory(ensemble, ts, x0, us=us)
    return jnp.var(preds.reshape(preds.shape[0], -1), axis=0)  # (n,)


def ensemble_from_flat_samples(theta_samples, unravel, static):
    """Batched pytree (an ensemble axis on every array leaf) from an (n, D) array of flat parameter vectors, e.g. weight samples drawn from a SWAG posterior.

    `unravel` is the function returned by `jax.flatten_util.ravel_pytree` for one member's parameters (`eqx.partition(model, eqx.is_array)[0]`), `static` the static part of the same partition.
    `static` may still contain array leaves (e.g. a fixed equilibrium point); they are broadcast to the member axis so that every vmap over members sees a fully batched pytree.
    """
    n = theta_samples.shape[0]
    params_trees = jax.vmap(unravel)(theta_samples)
    static_broadcast = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (n, *leaf.shape)) if eqx.is_array(leaf) else leaf,
        static,
    )
    return eqx.combine(params_trees, static_broadcast)


# ------------------------------------------------------------------------------------------------ list of members and the training-RMSE filter
class RelativeThreshold:
    """Cutoff as `factor * median(rmse)` for `filter_members_by_rmse`."""

    def __init__(self, factor):
        self.factor = factor


def _call_member(member, t, x0, us=None):
    """`member(t, x0)` or `member(t, x0, us=us)`, depending on whether there is forcing.

    The member is unwrapped first (a raw sPHNN straight from the optimiser is not callable before); this does not change the caller's member, so the same raw member can still be given to a Laplace class.
    """
    member = klax.unwrap(member)
    if us is None:
        return member(t, x0)
    return member(t, x0, us=us)


def _member_rmse(members, t, x0s, ys, us=None):
    """Per-member RMSE pooled over all trajectories, time steps and observed state dimensions, in the units of x0s / ys.

    `ys` (n, T, state_size_observed) may have fewer channels than a member's raw output (augmented states); only the leading channels are compared. `us` is an optional (n, T, input_size) forcing.
    """
    state_size_observed = ys.shape[-1]
    rmses = []
    for member in members:
        preds = []
        for i in range(x0s.shape[0]):
            us_i = None if us is None else us[i]
            traj = _call_member(member, t, x0s[i], us=us_i)
            preds.append(traj[..., :state_size_observed])
        preds = jnp.stack(preds)
        rmses.append(jnp.sqrt(jnp.mean((preds - ys) ** 2)))
    return jnp.stack(rmses)


def member_train_rmse(members, t_train, x0s_train, ys_train, us_train=None):
    """Per-member RMSE on the data the member was fitted on (the criterion for members that failed to converge); returns a (K,) array."""
    return _member_rmse(members, t_train, x0s_train, ys_train, us=us_train)


def filter_members_by_rmse(members, rmse, threshold=None):
    """Keep the members whose `rmse` is below `threshold`: a `RelativeThreshold(factor)` (cutoff = factor * median(rmse), default factor 5) or an absolute float.

    The default is meant to reject a failed or diverged training run (typically 10-100 times worse), not to trim the ordinary spread of an ensemble.
    Returns (kept_members, report); the report holds "keep_mask", "rmse", "cutoff", "dropped_indices" and "dropped_rmse".
    """
    if threshold is None:
        threshold = RelativeThreshold(5.0)

    rmse = jnp.asarray(rmse)
    if isinstance(threshold, RelativeThreshold):
        cutoff = float(jnp.median(rmse) * threshold.factor)
    else:
        cutoff = float(threshold)

    keep_mask = rmse < cutoff
    dropped_idx = [i for i in range(len(members)) if not bool(keep_mask[i])]
    kept_members = [m for i, m in enumerate(members) if bool(keep_mask[i])]

    report = {
        "keep_mask": keep_mask,
        "rmse": rmse,
        "cutoff": cutoff,
        "dropped_indices": dropped_idx,
        "dropped_rmse": [float(rmse[i]) for i in dropped_idx],
    }
    if dropped_idx:
        print(f"[filter_members_by_rmse] Dropped {len(dropped_idx)}/{len(members)} "
              f"member(s) at indices {dropped_idx} (RMSE {report['dropped_rmse']}, "
              f"cutoff={cutoff:.4g}, median={float(jnp.median(rmse)):.4g}).")
    else:
        print(f"[filter_members_by_rmse] Kept all {len(members)} members "
              f"(cutoff={cutoff:.4g}, median={float(jnp.median(rmse)):.4g}).")
    return kept_members, report
