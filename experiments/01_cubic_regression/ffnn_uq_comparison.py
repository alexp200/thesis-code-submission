"""Second part of the cubic-regression experiment: a small FFNN on the same data, treated probabilistically with LA / MoLA / DE / SWAG / MultiSWAG, next to the exact BLR of the cubic model.

Data: identical to verify_analytic_cubic_regression_mll.py (11 points, noise std 1.5, seed 2026). Model: tanh MLP 1 -> 16 -> 1 (49 parameters, flat vector theta).
MAP members (K = 10 initializations): Adam on  mean r^2 + (lambda/2)||theta||^2  with lambda = 1e-3 (the network still nearly interpolates the noisy data, which is accepted: this is an illustration, not a best-fit exercise; lambda = 1e-4, the harmonic-oscillator value, interpolates exactly, lambda = 1e-2 fits at noise level but collapses the ensemble into one mode);
members are filtered with the
5 x median rule on the clean-target RMSE.
LA:  GGN of the member (full), isotropic prior; prior precision alpha and noise std sigma calibrated by the Laplace marginal likelihood (repository workflow, Adam 200 steps, lr 0.1),
     independently per member; Sigma = (alpha I + GGN / sigma^2)^-1; linearized predictive, no sampling.   MoLA: uniform mixture of the K Laplace components.   DE: the K MAP members.
SWAG: SGD continuation of the member (2000 steps, mini-batches of ONE of the 11 points, gradient clipping 1, snapshot every 10 steps = 200 snapshots, learning rate 0.5, full covariance, ridge 1e-14, 64 samples).
     This is a visualization choice, not the reference rule of the harmonic-oscillator study: it gives the plotted parameter marginals a visible width (std 0.2-0.35),
     at the price of a function-space Std larger than LA's inside the data and a SWA mean that fits the data less closely.
     MultiSWAG: pooled samples of all members.
Parameter space: marginals of the four best-determined parameters under LA of member 0 (smallest marginal std), the same four indices in every method:
     LA / SWAG single Gaussians around member 0, DE Dirac deltas (arrows), MoLA / MultiSWAG uniform mixtures of the K component marginals (the marginal of a Gaussian mixture is the mixture of the 1D marginals).
Function space on x in [-1.6, 1.6] as in the verification section; shaded band = 2 sqrt(V[f(x)|D]).
Output: thesis_figures/final/.../exp01: fig03_ffnn_uq_comparison_narrow.(png,pdf) (Figure 4.2: columns = the five NN methods) and tableA2_ffnn_setup.tex (Table B.2); audit: ffnn_summary.json.
Usage: python -u ffnn_uq_comparison.py
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the verification script is loaded as the module "cubic": its data generator and the exact BLR formulas are reused here
spec = importlib.util.spec_from_file_location("cubic", HERE / "verify_analytic_cubic_regression_mll.py")
cubic = importlib.util.module_from_spec(spec)
sys.modules["cubic"] = cubic
spec.loader.exec_module(cubic)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import klax
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(HERE.parent / "shared"))
import thesis_plot_style as tps
from thesis_plot_style import style as S
from src import laplace as lp
from src import swag as sw
from src.priors import IsotropicGaussianPrior

from plot_common import NN_LS, LStyles, LEGEND_KW, tints
import export_paths as EP

from ffnn_settings import (H, K, L2, INIT, TRAIN_STEPS, TRAIN_LR, CAL_STEPS, CAL_LR, SWAG_BATCH, SWAG_STEPS, SWAG_FREQ, SWAG_LR, SWAG_CLIP, SWAG_RIDGE, N_SAMPLES, SAMPLE_KEY_OFFSET)  # (reported settings, common/ffnn_settings.py)
N_SELECT = 4                                # number of weights shown in the parameter panels

from dataclasses import replace

cfg0 = cubic.VerificationConfig()
X_EVAL = (cfg0.x_eval_min, cfg0.x_eval_max, cfg0.n_eval)
FFNN_LS = NN_LS


# ---------------------------------------------------------------------------------------------- network
# The network is an MLP (1 -> H -> 1, tanh). Laplace and SWAG work on the flat parameter vector, so f(p, x) rebuilds the network
# from a flat vector (as src/models.py does for the ODE models).

def npdf(x, m, s):
    """Density of N(m, s^2) at x (marginals of the parameters)."""
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def make_model(key):
    """MLP 1 -> H -> 1 (tanh) with Gaussian initial weights, standard deviations from ffnn_settings.INIT (output bias zero)."""
    k1, k2, k3 = jrd.split(key, 3)
    mlp = klax.nn.MLP("scalar", "scalar", [H], activation=jnp.tanh, key=key)
    where = lambda m: (m.layers[0].weight, m.layers[0].bias, m.layers[1].weight, m.layers[1].bias)
    return eqx.tree_at(where, mlp, (INIT["w1"] * jrd.normal(k1, (H,)).reshape(1, H), INIT["b1"] * jrd.normal(k2, (H,)),
                                    INIT["w2"] * jrd.normal(k3, (H,)).reshape(H, 1), jnp.zeros(1)))


_params, _static = eqx.partition(make_model(jrd.key(0)), eqx.is_array)
_flat, _unravel = ravel_pytree(_params)     # flat order: hidden weights, hidden biases, output weights, output bias
D = _flat.size                              # number of parameters (49)


def flatten(model):
    """Flat parameter vector of a trained model."""
    return ravel_pytree(eqx.filter(model, eqx.is_array))[0]


def f(p, x):
    """The network at a single input x for the flat parameter vector p."""
    return eqx.combine(_unravel(p), _static)(x)


f_vec = jax.vmap(f, in_axes=(None, 0))      # f over a vector of inputs


class MAPLoss(klax.Loss):
    """MAP objective of one member: mean squared residual + (L2 / 2) * squared norm of all parameters."""

    def __init__(self, l2):
        self.l2 = float(l2)

    def value(self, model, data, run_state):
        x, y = data
        penalty = sum(jnp.sum(a ** 2) for a in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
        return jnp.mean((jax.vmap(model)(x) - y) ** 2) + 0.5 * self.l2 * penalty


def analyse(sigma_gen):
    """Everything that depends on the data set: MAP members, LA / MoLA / DE / SWAG / MultiSWAG moments and parameter marginals, exact BLR (empirical-Bayes hyper-parameters)."""
    # --- data of the cubic experiment (same generator as the verification script)
    cfg = replace(cfg0, sigma_generation=sigma_gen)
    x_train, y_train, theta_true = cubic.synthetic_dataset(cfg)
    x_eval = jnp.linspace(*X_EVAL)
    N = x_train.shape[0]
    y_clean = cubic.phi(x_train) @ theta_true

    # --- MAP objective on the flat vector (used by the SWAG continuation below)
    def loss(p, idx=None):
        x, y = (x_train, y_train) if idx is None else (x_train[idx], y_train[idx])
        return jnp.mean((f_vec(p, x) - y) ** 2) + 0.5 * L2 * jnp.sum(p**2)

    # --- SWAG: SGD continuation from a MAP member with mini-batches; collects the snapshot statistics of src/swag.py
    def swag_state(p0, lr, key):
        opt = optax.chain(optax.clip_by_global_norm(SWAG_CLIP), optax.sgd(lr), sw.swag_full(freq=SWAG_FREQ))
        def step(c, k):
            p, st = c
            idx = jrd.choice(k, N, (SWAG_BATCH,), replace=False)
            u, st = opt.update(jax.grad(loss)(p, idx), st, p)
            return (optax.apply_updates(p, u), st), None
        return jax.lax.scan(step, (p0, opt.init(p0)), jrd.split(key, SWAG_STEPS))[0][1][-1]

    def pred(samples):
        return np.asarray(jax.vmap(lambda p: f_vec(p, x_eval))(samples))

    def swag_samples(s, key, n=N_SAMPLES):
        return jax.vmap(lambda k: sw.sample_swag_full(k, s, eps=SWAG_RIDGE))(jrd.split(key, n))

    # --- K MAP members, trained as one vmapped ensemble (full batch, Adam with cosine learning-rate decay);
    #     members whose clean RMSE exceeds 5 x the median are dropped
    swag_j = jax.jit(swag_state)
    ensemble = eqx.filter_vmap(make_model)(jnp.stack([jrd.key(k) for k in range(K)]))
    ensemble, _ = klax.fit(ensemble, (x_train, y_train), batch_size=N, batch_axes=(0, 0), steps=TRAIN_STEPS, loss=MAPLoss(L2),
                           optimizer=optax.adam(optax.cosine_decay_schedule(TRAIN_LR, TRAIN_STEPS)),
                           make_logger=False, vmap_ensemble=True, key=jrd.key(0))
    all_members = [flatten(jax.tree.map(lambda a, k=k: a[k] if eqx.is_array(a) else a, ensemble)) for k in range(K)]
    rmse = np.array([float(jnp.sqrt(jnp.mean((f_vec(p, x_train) - y_clean) ** 2))) for p in all_members])
    keep = rmse <= 5 * np.median(rmse)
    members = [p for p, k in zip(all_members, keep) if k]
    Kr = len(members)
    print(f"[noise {sigma_gen}] clean RMSE per member", np.round(rmse, 3), "| retained", Kr, "of", K, "| noise: sample std", round(float(jnp.std(y_train - y_clean)), 3), "| y range", round(float(y_train.max() - y_train.min()), 2))
    # --- Laplace approximation per member: GGN, eigenvalues, alpha and sigma from the marginal likelihood, posterior covariance
    la = []
    for p in members:
        G = lp.compute_ggn(f, p, x_train, hessian_scale=1.0)
        ev, _ = lp.GGN_eigendecompose(G, THRESHOLD=1e-12)
        sse = float(jnp.sum((f_vec(p, x_train) - y_train) ** 2))
        a, s = lp.calibrate_mll(sse, p, N, ev, IsotropicGaussianPrior(D=D), n_steps=CAL_STEPS, lr=CAL_LR, verbose=False)
        la.append({"mu": np.asarray(p), "cov": np.asarray(jnp.linalg.inv(float(a) * jnp.eye(D) + G / float(s) ** 2)), "alpha": float(a), "sigma": float(s), "rank": int((np.asarray(ev) > 1e-8 * float(ev.max())).sum()), "train_rms": float(jnp.sqrt(jnp.mean((f_vec(p, x_train) - y_train) ** 2)))})
    print("  calibrated alpha", np.round([l["alpha"] for l in la], 3), "| sigma", np.round([l["sigma"] for l in la], 3))
    # --- DE predictions and SWAG components (one per member)
    de_pred = pred(jnp.stack(members))
    swag = [swag_j(p, SWAG_LR, jrd.key(k)) for k, p in enumerate(members)]
    sw_mc = [(np.asarray(s_.mean), np.asarray(s_.cov / jnp.maximum(s_.n - 1, 1) + SWAG_RIDGE * jnp.eye(D))) for s_ in swag]
    sw_smp = [pred(swag_samples(s_, jrd.key(SAMPLE_KEY_OFFSET + k))) for k, s_ in enumerate(swag)]
    res_fit = [float(jnp.mean(jnp.abs(f_vec(s_.mean, x_train) - y_train))) for s_ in swag]
    print("  SWAG: lr", SWAG_LR, "| finite", all(np.isfinite(v).all() for v in sw_smp), "| mean|residual| of the SWA means", np.round(np.mean(res_fit), 3))

    # --- function-space moments of every method on x_eval (MoLA: mean of the variances + variance of the means)
    def la_pred(l):
        J = np.asarray(jax.vmap(jax.grad(f), in_axes=(None, 0))(jnp.asarray(l["mu"]), x_eval))
        return np.asarray(f_vec(jnp.asarray(l["mu"]), x_eval)), np.einsum("nd,de,ne->n", J, l["cov"], J)

    lp_ = [la_pred(l) for l in la]
    func = {"LA": (lp_[0][0], np.sqrt(lp_[0][1])),
            "MoLA": (np.mean([m for m, _ in lp_], 0), np.sqrt(np.mean([v for _, v in lp_], 0) + np.var([m for m, _ in lp_], 0))),
            "DE": (de_pred.mean(0), de_pred.std(0)), "SWAG": (sw_smp[0].mean(0), sw_smp[0].std(0)),
            "MultiSWAG": (np.concatenate(sw_smp).mean(0), np.concatenate(sw_smp).std(0))}
    xin = np.abs(np.asarray(x_eval)) <= 1
    print("  mean predictive Std inside / outside [-1,1]:", {k: (round(float(v[1][xin].mean()), 3), round(float(v[1][~xin].mean()), 3)) for k, v in func.items()})
    # --- parameter space: the four best-determined weights of member 0 (smallest LA std) and the marginals of every method
    sd0 = np.sqrt(np.diag(la[0]["cov"]))
    sel = [int(i) for i in np.argsort(sd0)[:N_SELECT]]
    print("  plotted theta_0..theta_3 = weight indices", sel, "| LA std", np.round(sd0[sel], 3), "| SWAG std (member 0)", np.round(np.sqrt(np.diag(sw_mc[0][1]))[sel], 3))
    MU = {"LA": [la[0]["mu"]], "MoLA": [l["mu"] for l in la], "DE": [l["mu"] for l in la], "SWAG": [sw_mc[0][0]], "MultiSWAG": [m for m, _ in sw_mc]}
    SD = {"LA": [np.sqrt(np.diag(la[0]["cov"]))], "MoLA": [np.sqrt(np.diag(l["cov"])) for l in la], "SWAG": [np.sqrt(np.diag(sw_mc[0][1]))], "MultiSWAG": [np.sqrt(np.diag(c)) for _, c in sw_mc]}
    # exact BLR at its empirical-Bayes optimum for THIS data set
    eb = cubic.optimize_objective_from_grid(lambda z: cubic.exact_blr_neg_log_mll_from_logparams(z, np.asarray(x_train), np.asarray(y_train)))
    print(f"  BLR empirical Bayes: alpha {eb['alpha']:.4f}, sigma {eb['sigma']:.4f}")
    # --- numbers for the audit file ffnn_summary.json, and everything the figures need
    info = {"sigma_gen": sigma_gen, "noise_sample_std": float(jnp.std(y_train - y_clean)), "y_range": float(y_train.max() - y_train.min()), "L2": L2,
            "clean_rmse_per_member": rmse.tolist(), "n_retained": Kr, "n_trained": K,
            "train_residual_rms_per_member": [l["train_rms"] for l in la], "alpha_per_member": [l["alpha"] for l in la], "sigma_per_member": [l["sigma"] for l in la], "ggn_rank": sorted({l["rank"] for l in la}), "D": D,
            "swag": {"lr": SWAG_LR, "batch": SWAG_BATCH, "steps": SWAG_STEPS, "freq": SWAG_FREQ, "n_snapshots": SWAG_STEPS // SWAG_FREQ, "n_samples": 64, "mean_abs_residual_of_swa_means": float(np.mean(res_fit))},
            "predictive_std_inside_outside": {k: [float(v[1][xin].mean()), float(v[1][~xin].mean())] for k, v in func.items()},
            "selected_weight_indices": sel, "la_std_selected": sd0[sel].tolist(), "swag_std_selected_member0": np.sqrt(np.diag(sw_mc[0][1]))[sel].tolist(),
            "de_values_selected": [[float(l["mu"][i]) for l in la] for i in sel], "blr_empirical_bayes": {"alpha": eb["alpha"], "sigma": eb["sigma"], "log_mll": eb["log_mll"]}}
    return dict(info=info, x_train=np.asarray(x_train), y_train=np.asarray(y_train), x_eval=np.asarray(x_eval), truth=np.asarray(cubic.phi(x_eval) @ theta_true), theta_true=np.asarray(theta_true),
                func=func, sel=sel, MU=MU, SD=SD, la=la, sigma_gen=sigma_gen, eb=eb)


# ------------------------------------------------------------------------------------------------ figures
def panel_range(R, name, q=0.005, pad=0.08):
    """Horizontal range of one parameter panel (free per panel: the raw parameter values of the different models / methods have no common meaning, only the shapes are compared).
    Union over the four plotted parameters of the [q, 1-q] quantile range of the shown density (DE: min / max of the member values), padded by `pad`."""
    sel = R["sel"]
    if name == "DE":
        v = [l["mu"][i] for l in R["la"] for i in sel]
        lo, hi = min(v), max(v)
    else:
        grid = np.linspace(-60, 60, 24001)
        lo, hi = np.inf, -np.inf
        for i in sel:
            dens = np.mean([npdf(grid, mu[i], sd[i]) for mu, sd in zip(R["MU"][name], R["SD"][name])], axis=0)
            cdf = np.cumsum(dens) / dens.sum()
            lo, hi = min(lo, grid[np.searchsorted(cdf, q)]), max(hi, grid[np.searchsorted(cdf, 1 - q)])
    w = hi - lo
    return lo - pad * w, hi + pad * w


def draw_param(ax, R, name):
    """Parameter panel of one method: marginal densities of the four selected weights (DE: Dirac deltas)."""
    sel, MU, SD, la = R["sel"], R["MU"], R["SD"], R["la"]
    cs = tints(S.METHOD_COLOR[name], N_SELECT)
    lo, hi = panel_range(R, name)
    if name == "DE":
        for j, i in enumerate(sel):                  # Dirac deltas: shaft + head, all of the same height (each delta carries the mass 1/K); the four parameters differ by tint and line style only
            for v in [l["mu"][i] for l in la]:
                ax.vlines(v, 0, 1.0, color=cs[j], ls=FFNN_LS[j], lw=1.0, alpha=0.9, zorder=3 + j)
                ax.plot([v], [1.0], marker="^", ms=3.2, color=cs[j], mec="none", ls="", alpha=0.9, zorder=3 + j)
        ax.set_ylim(0, 1.25)
    else:
        xs_f = np.linspace(lo, hi, 3000)
        for j, i in enumerate(sel):
            dens = np.mean([npdf(xs_f, mu[i], sd[i]) for mu, sd in zip(MU[name], SD[name])], axis=0)
            ax.plot(xs_f, dens, color=cs[j], ls=FFNN_LS[j], lw=S.LW["main"])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)      # no numbers: the scale of a parameter axis carries no meaning across models / methods
    ax.grid(False, axis="x")


def draw_func(ax, R, name):
    """Function-space panel of one method: predictive mean, band of 2 standard deviations, data points and the true cubic."""
    col = S.METHOD_COLOR[name]
    m, s_ = R["func"][name]
    xe = R["x_eval"]
    ax.scatter(R["x_train"], R["y_train"], s=6, color=S.DATA_COLOR, lw=0, zorder=5)
    ax.plot(xe, R["truth"], **S.TRUTH)
    ax.plot(xe, m, color=col, lw=S.LW["mean"])
    ax.fill_between(xe, m - 2 * s_, m + 2 * s_, color=col, alpha=0.2, lw=0)
    ax.set_ylim(-14, 30)


def legend_handles():
    """Handles and labels of the common legend above the figure."""
    hs = [LStyles("0.35", FFNN_LS), Line2D([], [], color="0.4", lw=S.LW["mean"]), Patch(facecolor="0.4", alpha=0.25),
          Line2D([], [], ls="", marker="o", ms=2.5, color=S.DATA_COLOR), Line2D([], [], **S.TRUTH)]
    labels = [r"$\theta_{\mathrm{NN},i}$", r"$\mathbb{E}[f(x)\mid\mathcal{D}]$", r"$2\sqrt{\mathbb{V}[f(x)\mid\mathcal{D}]}$", "noisy observations", "true cubic"]
    return hs, labels


KEY_NARROW = "fig03_ffnn_uq_comparison_narrow"
ROWS = ["LA", "MoLA", "DE", "SWAG", "MultiSWAG"]


def make_figure(R, base):
    """Figure 4.2: columns = the five NN methods; top row the parameter marginals, bottom row the predictive mean and 2 standard deviations; one text width."""
    tps.apply_style(KEY_NARROW)
    fig, axes = plt.subplots(2, len(ROWS), figsize=tps.figsize_for(KEY_NARROW, 3.15), gridspec_kw={"hspace": 0.4, "wspace": 0.17, "top": 0.86})
    axp, axf = list(axes[0]), list(axes[1])
    for a_ in axf[1:]:
        a_.sharey(axf[0])
        a_.tick_params(labelleft=False)
    for r, name in enumerate(ROWS):
        draw_param(axp[r], R, name)
        draw_func(axf[r], R, name)
        tps.col_heading(axp[r], name)                                            # method identifiers: normal-weight column headings outside the axes
        if r == 0:
            axp[r].set_ylabel(r"$q(\theta\mid\mathcal{D})$")
            axf[r].set_ylabel(r"$f(x)$")
        axp[r].set_xlabel(r"$\theta_{\mathrm{NN}}$")
        axf[r].set_xlabel(r"$x$")
        if r > 0:
            axp[r].set_yticks([])
    axf[0].set_yticks([-10, 0, 10, 20, 30])                                      # shared y axis; the ticks of the full range -14..30 stay visible at the larger font size
    fig.align_ylabels([axp[0], axf[0]])          # y labels of the first column on one vertical line (the function panel has tick numbers, the parameter panel none)
    hs, labels = legend_handles()
    tps.legend_above(fig, hs, 3, labels=labels, **LEGEND_KW)
    tps.save_figure(fig, base, generator="01_cubic_regression/ffnn_uq_comparison.py")


def table_a2_lines():
    """tableA2_ffnn_setup.tex: compact inference / numerics report (Table B.2). Settings: common/ffnn_settings.py."""
    from export_tables import ffnn_inference_table
    return ffnn_inference_table(cfg0)


def export(R):
    """Thesis export: Figure 4.2, Table B.2 and the numbers behind them (ffnn_summary.json in the audit folder)."""
    EP.ensure()
    make_figure(R, EP.IMG_MAIN / "fig03_ffnn_uq_comparison_narrow")
    (EP.TAB_APP / "tableA2_ffnn_setup.tex").write_text("\n".join(table_a2_lines()) + "\n")
    (EP.AUDIT / "ffnn_summary.json").write_text(json.dumps(R["info"], indent=1))
    print("exported:", EP.IMG_MAIN / "fig03_ffnn_uq_comparison_narrow", "+ table B.2 + ffnn_summary.json")


if __name__ == "__main__":
    export(analyse(1.5))
