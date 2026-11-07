"""Ott-style check of the Laplace approximation for a NODE on the harmonic oscillator (Appendix B.2.1, Figure B.1).

Follows the naive-NODE experiment of Ott et al. (2023): a 2-256-2 tanh NODE (Euler solver, step 0.02) is trained on 16 noisy trajectories of the data sets `lower_half` and `left_half`
(seed 17, orthogonal weights, PyTorch-style uniform biases, Adam 1e-3, 5000 full-batch steps, trajectory MSE), a full-network Laplace approximation is fitted and calibrated by the
marginal likelihood, and the figure shows the linearized predictive mean with +-2 std bands and the MAP vector field with the norm of its componentwise std. The vector-field colour
scale is the cap of Ott et al. (linear 0..3, values >= 3 saturate, labelled on the colour bar).

The scalar results are compared with runs/primary_investigations/*/summary.json before the figure is drawn (written to thesis_figures/final_audit/ott_verification_reproduction.json).

Output: thesis_figures/final/00_Masterthesis/Knowledgebase/imgs/appendix/a11_verification_ott_style_v2.(png,pdf)   (path from common/export_paths.py)
Usage: python -u make_ott_figure.py   (trains both networks, about 1 min)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import diffrax
import dynax
import jax.numpy as jnp
import jax.random as jrd
import klax
import matplotlib.pyplot as plt
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(ROOT / "figures"))
sys.path.insert(0, str(ROOT / "common"))
import export_paths as EP
import plot_lib as pl

from src import laplace as lp
from src import models as md

RUN_DIR = HERE / "runs" / "primary_investigations" / "torch_uniform_bias_verification_2026-09-17"
KEY = "a11_verification_ott_style_v2"
SEED = 17
WIDTH, STEPS, LEARNING_RATE = 256, 5000, 1e-3
N_TRAJECTORIES, N_TIME, N_EVAL, GRID_SIZE = 16, 50, 300, 35
MASS, SPRING = 1.0, 2.0
NOISE_VARIANCE = 0.3
EULER_DT0, MAX_STEPS = 0.02, 65_536
FIELD_Q_LIMITS, FIELD_P_LIMITS = (-6.0, 6.0), (-8.0, 8.0)
# name: (mean q, mean p / sqrt(k / m), seed offset, name in the stored summary); the second data set is "dataset-left-half" of Ott et al., Appendix A.2
DATASETS = {"lower_half": (3.0, 0.0, 0, "lower_half"), "left_half": (0.0, 3.0, 1, "full_cycle")}
LABELS = {"lower_half": "lower half", "left_half": "left half"}


# ---------------------------------------------------------------- data and model
def true_trajectory(x0, t):
    """Analytical solution (q, p) of the oscillator from x0 at the times t."""
    w = jnp.sqrt(SPRING / MASS)
    q = x0[0] * jnp.cos(w * t) + x0[1] / (MASS * w) * jnp.sin(w * t)
    p = x0[1] * jnp.cos(w * t) - MASS * w * x0[0] * jnp.sin(w * t)
    return jnp.stack([q, p], axis=-1)


def build_dataset(name, key):
    """16 noisy trajectories over half a period; initial states ~ N(mean, 0.2 I)."""
    mean_q, mean_p_factor, _, _ = DATASETS[name]
    mean_p = mean_p_factor * float(jnp.sqrt(SPRING / MASS))
    ic_key, noise_key = jrd.split(key)
    key_q, key_p = jrd.split(ic_key)
    q0 = mean_q + jnp.sqrt(0.2) * jrd.normal(key_q, (N_TRAJECTORIES,))
    p0 = mean_p + jnp.sqrt(0.2) * jrd.normal(key_p, (N_TRAJECTORIES,))
    x0s = jnp.stack([q0, p0], axis=-1)
    t_train = jnp.linspace(0.0, float(jnp.pi / jnp.sqrt(SPRING / MASS)), N_TIME)
    ys_clean = jax.vmap(lambda x0: true_trajectory(x0, t_train))(x0s)
    ys_noisy = ys_clean + jnp.sqrt(NOISE_VARIANCE) * jrd.normal(noise_key, ys_clean.shape)
    return {"name": name, "t_train": t_train, "x0s": x0s, "ys_clean": ys_clean, "ys_noisy": ys_noisy, "x0_eval": jnp.array([mean_q, mean_p])}


def torch_linear_bias_initializer(key, shape, fan_in, dtype=jnp.float64):
    """Match torch.nn.Linear's default U(-1/sqrt(fan_in), 1/sqrt(fan_in)) bias."""
    bound = 1.0 / jnp.sqrt(jnp.asarray(fan_in, dtype=dtype))
    return jrd.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)


def make_node(key):
    """Untrained NODE with Euler integration."""
    node = dynax.NeuralODE(state_size=2, input_size=None, time_dependent=False, width_sizes=[WIDTH], activation=jax.nn.tanh, dtype=jnp.float64, key=key,
                           weight_init=jax.nn.initializers.orthogonal(scale=1.0), bias_init=torch_linear_bias_initializer)
    return dynax.ODESolver(node, solver=diffrax.Euler(), stepsize_controller=diffrax.ConstantStepSize(), dt0=EULER_DT0, max_steps=MAX_STEPS)


@klax.loss
def trajectory_mse(model, data, run_state):
    """Mean squared error over all trajectories, times and coordinates."""
    del run_state
    t_train, x0s, ys = data
    return jnp.mean((jax.vmap(model, in_axes=(None, 0))(t_train, x0s) - ys) ** 2)


# ---------------------------------------------------------------- one data set
def run_one_dataset(name):
    """Train the NODE, fit the Laplace approximation and evaluate mean, std and the vector-field uncertainty; returns the plot payload and the scalars for the reproduction check."""
    dataset_key = jrd.fold_in(jrd.key(SEED), DATASETS[name][2])
    dataset = build_dataset(name, dataset_key)
    model_key, train_key = jrd.split(jrd.fold_in(dataset_key, 10))
    model, _ = klax.fit(make_node(model_key), (dataset["t_train"], dataset["x0s"], dataset["ys_noisy"]), batch_axes=(None, 0, 0), batch_size=N_TRAJECTORIES, steps=STEPS,
                        loss=trajectory_mse, log_every=STEPS // 20, key=train_key, optimizer=optax.adam(LEARNING_RATE), verbose=1)
    preds = jax.vmap(model, in_axes=(None, 0))(dataset["t_train"], dataset["x0s"])

    la = md.Laplace_NeuralODESolver(model, dataset["t_train"], dataset["x0s"], dataset["ys_noisy"])
    la.fit_laplace_mll()
    alpha, sigma = float(la.prior_prec_opt), float(la.sigma_noise_opt)

    t_eval = jnp.linspace(0.0, 10.0, N_EVAL)
    x0_eval = dataset["x0_eval"]
    mean_eval = la.mean_linearized(t_eval, x0_eval)
    jacobian = jax.jacobian(lambda p: la.f_solve_flat_general(p, t_eval, x0_eval, None))(la.p_map)
    std_eval = lp.pushforward_linearized_std(jacobian @ la.eigvecs_ggn, la.eigvals_ggn, la.active_idx, alpha, la.prior, sigma_noise=sigma).reshape(len(t_eval), la.state_size)

    q, p = jnp.linspace(*FIELD_Q_LIMITS, GRID_SIZE), jnp.linspace(*FIELD_P_LIMITS, GRID_SIZE)
    Q, P = jnp.meshgrid(q, p)
    points = jnp.stack([Q.reshape(-1), P.reshape(-1)], axis=-1)
    vf_map = jax.vmap(lambda z: la.f_field(la.p_map, z))(points)
    proj = lp.project_jacobian_vector(la.f_field, la.p_map, la.eigvecs_ggn, points)
    stds = [lp.pushforward_linearized_std(proj[i], la.eigvals_ggn, la.active_idx, alpha, la.prior, sigma_noise=sigma) for i in range(2)]
    vf_unc = jnp.linalg.norm(jnp.stack(stds, axis=-1), axis=-1)

    payload = {"dataset": dataset, "t_eval": t_eval, "true_eval": true_trajectory(x0_eval, t_eval), "mean_eval": mean_eval, "std_eval": std_eval, "Q": Q, "P": P, "vf_map": vf_map, "vf_unc": vf_unc}
    scalars = {"prior_precision": alpha, "sigma_noise": sigma, "parameter_count": int(la.D),
               "map_train_rmse_against_clean_data": float(jnp.sqrt(jnp.mean((preds - dataset["ys_clean"]) ** 2))),
               "map_train_rmse_against_noisy_data": float(jnp.sqrt(jnp.mean((preds - dataset["ys_noisy"]) ** 2)))}
    return payload, scalars


def compare_with_saved(scalars):
    """Relative differences of the scalar results to the summary of the original run."""
    saved = {r["dataset"]: r for r in json.loads((RUN_DIR / "summary.json").read_text())["results"]}
    rep = {}
    for name, s in scalars.items():
        old = saved[DATASETS[name][3]]
        old = {**old, **old["map_fit"]}
        rep[name] = {k: {"new": v, "saved": old[k], "rel_diff": abs(v - old[k]) / max(abs(old[k]), 1e-300)} for k, v in s.items()}
    return {"max_relative_difference_to_saved_summary": max(v["rel_diff"] for d in rep.values() for v in d.values()), "per_dataset": rep}


# ---------------------------------------------------------------- figure
def draw(payloads, out_base):
    """One row per data set: function space (left) and vector-field panel with colour bar (right); typography at the final document size, data set names as row headings."""
    pl.tps.apply_style(KEY)
    n = len(payloads)
    W = pl.tps.sizes.width_in(KEY)
    Hh = 1.95 * n + 0.8
    fig = plt.figure(figsize=(W, Hh))
    gs = fig.add_gridspec(n, 2, width_ratios=[3.0, 1.3], wspace=0.34, hspace=0.2, left=0.07, right=0.88, top=1.0 - 0.3 / Hh, bottom=0.55 / Hh)
    spec = pl.color_spec("cap3")
    rows = []
    for r, p in enumerate(payloads):
        ds = p["dataset"]
        t, noisy, t_train = np.asarray(p["t_eval"]), np.asarray(ds["ys_noisy"]), np.asarray(ds["t_train"])
        ax = fig.add_subplot(gs[r, 0])
        pl.draw_function_space(ax, t, np.asarray(p["true_eval"]), np.asarray(p["mean_eval"]), np.asarray(p["std_eval"]), train_t_end=float(t_train[-1]), obs=(t_train, noisy))
        ax.set_ylabel(r"$q,\ p$")
        ax.set_xlabel(r"$t$" if r == n - 1 else "")
        if r < n - 1:
            ax.set_xticklabels([])
        axv = fig.add_subplot(gs[r, 1])
        Q, P = np.asarray(p["Q"]), np.asarray(p["P"])
        pl.draw_vector_field(axv, Q, P, np.asarray(p["vf_unc"]).reshape(Q.shape), np.asarray(p["vf_map"]), spec, obs_xy=noisy.reshape(-1, 2))
        axv.set_ylabel(r"$p$")
        axv.set_xlabel(r"$q$" if r == n - 1 else "")
        if r < n - 1:
            axv.set_xticklabels([])
        pl.colorbar(fig, spec, cax=axv.inset_axes([1.07, 0.0, 0.07, 1.0]), labelsize=None)
        rows.append(([ax, axv], LABELS[ds["name"]]))
    pl.tps.row_headings(rows)
    pl.tps.legend_above(fig, pl.function_space_handles(), 6)
    pl.tps.save_figure(fig, out_base, generator="02_harmonic_oscillator/ott_check/make_ott_figure.py")


def main():
    """Command-line entry: train, check against the saved summary and draw the figure."""
    payloads, scalars = [], {}
    for name in DATASETS:
        payload, scalars[name] = run_one_dataset(name)
        payloads.append(payload)
    rep = compare_with_saved(scalars)
    EP.AUDIT.mkdir(parents=True, exist_ok=True)
    (EP.AUDIT / "ott_verification_reproduction.json").write_text(json.dumps(rep, indent=1, default=float))
    print("max relative difference to saved summary:", rep["max_relative_difference_to_saved_summary"])
    EP.IMG_APP.mkdir(parents=True, exist_ok=True)
    draw(payloads, EP.IMG_APP / "a11_verification_ott_style_v2")


if __name__ == "__main__":
    main()
