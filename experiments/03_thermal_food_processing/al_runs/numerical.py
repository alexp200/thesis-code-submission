"""Data preparation, batched inference, error statistics and the acquisition variance kernels (DE, LA, MoLA) of the AL study."""

import importlib
import sys

import numpy as np

from design import DATA, PRIMARY, ROOT

sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------------------------------------ data and models
def helpers():
    """Import the training module and the data module from kernels/core."""
    sys.path.insert(0, str(PRIMARY))
    return importlib.import_module("train_common"), importlib.import_module("chicken_data")


def prepare_data(ids):
    """All 30 trajectories, normalised with the statistics of the training trajectories `ids` only."""
    import jax.numpy as jnp
    import xarray as xr
    _, cd = helpers()
    with xr.open_dataset(DATA / "dataset.nc", engine="h5netcdf") as ds:
        all_ids = ds.simulation_id.values.astype(int).tolist()
        t = jnp.asarray(ds.time.values)
        ys = jnp.asarray(ds.temperature.values)
        us = jnp.asarray(ds.oven_temperature.values)[..., None]
    idx = jnp.asarray([all_ids.index(sid) for sid in ids])
    us_std, ys_std = cd.normalization_stats(us[idx], ys[idx])
    if not np.all(np.isfinite(np.asarray(us_std))) or not np.all(np.isfinite(np.asarray(ys_std))) or np.any(np.asarray(ys_std) <= 0) or float(us_std) <= 0:
        raise ValueError("Invalid training-only normalization scales")
    x = (ys[:, 0] - cd.T_AMBIENT) / ys_std
    controls = (us - cd.T_AMBIENT) / us_std
    return dict(t=t, all_ids=all_ids, x=x, u=controls, target_K=np.asarray(ys),
                x0s_train=x[idx], us_train=controls[idx], ys_train=(ys[idx]-cd.T_AMBIENT)/ys_std,
                us_std=us_std, ys_std=ys_std, n_probes=2, n_train=len(ids))


def skeleton(model, seeds):
    """Untrained ensemble of `model` with the given seeds (template for loading checkpoints)."""
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jrd
    tc, _ = helpers()
    return eqx.filter_vmap(tc.SKELETON_FROM_KEY[model])(jnp.stack([jrd.key(k) for k in seeds]))


# ------------------------------------------------------------------------------------------------ inference and errors
def predictor(data):
    """Jitted rollout of all ensemble members for a batch of initial states and inputs."""
    import equinox as eqx
    import jax
    import klax

    def one(member, x, u):
        member = klax.finalize(member) if klax.contains_unwrappables(member) else member
        return jax.vmap(member, in_axes=(None, 0, 0))(data["t"], x, u)[..., :2]

    return eqx.filter_jit(eqx.filter_vmap(one, in_axes=(eqx.if_array(0), None, None)))


def predict_batch(fn, ensemble, x, u, batch):
    """Apply `fn` to a batch that is padded to the fixed size `batch` (avoids recompilation)."""
    import jax
    import jax.numpy as jnp
    n = len(x)
    if n < batch:
        x = jnp.concatenate((x, jnp.repeat(x[-1:], batch-n, axis=0)))
        u = jnp.concatenate((u, jnp.repeat(u[-1:], batch-n, axis=0)))
    return np.asarray(jax.block_until_ready(fn(ensemble, x, u)))[:, :n]


def statistics(errors):
    """Median and quartiles of the member errors."""
    errors = np.asarray(errors)
    if errors.ndim != 1 or not np.all(np.isfinite(errors)):
        raise ValueError("All-member errors must be finite")
    q25, med, q75 = np.quantile(errors, [.25, .5, .75], method="linear")
    return dict(status="complete", total_count=len(errors), valid_count=len(errors),
                median_K=float(med), q25_K=float(q25), q75_K=float(q75), iqr_K=float(q75-q25))


def metrics(per_probe_mse, max_error, plan, train_ids, dropped, mean_mse=None):
    """Report of one evaluation: member RMSEs (all, per probe, per trajectory), subset statistics and the ensemble-mean RMSE."""
    per_probe_mse = np.asarray(per_probe_mse)
    per_trajectory = np.sqrt(per_probe_mse.mean(axis=2))
    errors = np.sqrt(per_probe_mse.mean(axis=(1, 2)))
    ids = plan["evaluation_ids"]
    groups = dict(plan["subset_ids"])
    groups["selected_training"] = [sid for sid in train_ids if sid in ids]
    groups["remaining_evaluation"] = [sid for sid in ids if sid not in train_ids]
    subsets = {}
    for name, group in groups.items():
        if not group:
            continue
        idx = [ids.index(sid) for sid in group]
        values = np.sqrt(per_probe_mse[:, idx].mean(axis=(1, 2)))
        subsets[name] = dict(count=len(idx), member_rmse_K=values.tolist(), **statistics(values))
    result = dict(units="K", quantile_method="linear", evaluation_ids=ids,
                scope=plan["evaluation_scope"], all_members=statistics(errors),
                members=[dict(member_index=k, status="passed", training_filter_kept=k not in dropped,
                              rmse_K=float(errors[k]), per_probe_rmse_K=np.sqrt(per_probe_mse[k].mean(axis=0)).tolist(),
                              per_trajectory_rmse_K=per_trajectory[k].tolist(),
                              per_trajectory_max_error_K=max_error[k].tolist()) for k in range(len(errors))],
                subsets=subsets)
    if mean_mse is not None:
        result["ensemble_mean_rmse_K"] = float(np.sqrt(mean_mse[0].mean()))
        result["filtered_ensemble_mean_rmse_K"] = float(np.sqrt(mean_mse[1].mean()))
        result["ensemble_mean_per_trajectory_rmse_K"] = np.sqrt(mean_mse[0].mean(axis=1)).tolist()
        result["filtered_ensemble_mean_per_trajectory_rmse_K"] = np.sqrt(mean_mse[1].mean(axis=1)).tolist()
    return result


# ------------------------------------------------------------------------------------------------ acquisition
def acquisition_kernels(model, method, ensemble, data, retained, settings):
    """Fit the Laplace approximations of the method and return (kernel, fits); kernel(x, u) gives the total, within and between variance of the probe rollouts of one candidate."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import time
    from src.models import Laplace_NeuralODESolver, Laplace_sPHNNSolver
    tc, _ = helpers()
    members = tc.split_ensemble(ensemble, 3)
    chosen = [0] if method == "LA" else retained
    if method == "LA" and 0 not in retained:
        raise ValueError("Predeclared LA member 0 failed training filter; no silent seed replacement")
    # Laplace fits: LA uses member 0, MoLA all retained members
    components, fits = [], []
    if method != "DE":
        cls = Laplace_NeuralODESolver if model == "NODE" else Laplace_sPHNNSolver
        for k in chosen:
            tick = time.perf_counter()
            c = cls(members[k], data["t"], data["x0s_train"], data["ys_train"], us_train=data["us_train"])
            c.fit_laplace_mll()
            jax.block_until_ready((c.eigvals_ggn, c.eigvecs_ggn, c.prior_prec_opt, c.sigma_noise_opt))
            directions, _ = c._posterior_directions_and_weights()
            if directions.shape[1] != len(c.p_map):
                raise ValueError("Truncated posterior is not permitted")
            components.append(c)
            fits.append(dict(member_index=k, seconds=time.perf_counter()-tick,
                             prior_precision=float(c.prior_prec_opt), sigma_noise=float(c.sigma_noise_opt),
                             parameter_count=len(c.p_map)))
    # DE: variance over the retained members; LA / MoLA: within (Laplace) + between (means) variance
    if method == "DE":
        subset = jax.tree.map(lambda a: a[jnp.asarray(chosen)] if eqx.is_array(a) else a, ensemble)
        predict = predictor(data)

        def kernel(x, u):
            between = jnp.var(predict(subset, x, u), axis=0)
            return between, jnp.zeros_like(between), between
    else:
        jvps = [c.make_marginal_variance_eigen_jvp(data["t"], chunk_size=settings["direction_chunk"]) for c in components]

        def one(x, u):
            means = jnp.stack([c.mean_linearized(data["t"], x, u) for c in components])
            variances = jnp.stack([fn(x, u).reshape(means.shape[1:]) for fn in jvps])
            within = jnp.mean(variances, axis=0)
            between = jnp.var(means, axis=0)
            return within+between, within, between

        kernel = jax.jit(jax.vmap(one))
    return kernel, fits
