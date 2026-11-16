"""Example 1: uncertainty quantification of a neural ODE with the three methods of the thesis, on the harmonic oscillator.

    DE    the K trained members themselves: mean and spread of their rollouts
    LA    one member plus a linearised Laplace approximation of its parameters (src.models.Laplace_*Solver)
    MoLA  K members, each with its own Laplace approximation, averaged as a mixture (src.mola.MoLA_ODESolver)

Steps: 1 data, 2 train K members, 3 DE, 4 LA, 5 MoLA, 6 compare and plot. All methods start from the same trained members; the Laplace methods fit after training, nothing is retrained.
The experiments do exactly this for the same model classes (experiments/02_harmonic_oscillator/long_horizon/compute_methods.py); see docs/DESIGN.md for how the library is organised.

What to expect: with 3 members and 2000 steps the means drift in phase from the true orbit after the training window (RMSE about 1.6 at an amplitude of 3), and the LA and MoLA bands grow far beyond the amplitude
because the linearised pushforward accumulates phase uncertainty (the plot is cut at +-8). The thesis studies this behaviour in Section 4.1.2 (Figures 4.5 and 4.6).

Run from the repository root:   python examples/01_uq_workflow.py [--model sPHNN|NODE] [--steps 2000]      (a few minutes on a laptop CPU)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _oscillator import MODEL_CLASSES, base, ens_lib, fit_laplace, rollout, train_ensemble

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.mola import MoLA_ODESolver

K = 3                                                    # number of members


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_CLASSES, default="sPHNN")
    ap.add_argument("--steps", type=int, default=2000, help="training steps per member (the experiments use 5000)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output" / "uq_workflow.png")
    args = ap.parse_args(argv)

    # 1  Data: 16 noisy trajectories of the oscillator over half a period, started near (3, 0); prediction from (3, 0) up to t = 10, far beyond the training window
    data = ens_lib.build_dataset("lower_half")
    t_train, x0s, ys = data["t_train"], data["x0s"], data["ys_noisy"]
    x0 = data["representative_x0"]
    t_eval = jnp.linspace(0.0, base.EVALUATION_T_END, 100)
    truth = base.true_trajectory(x0, t_eval)

    # 2  Train K members from different random initialisations
    members = train_ensemble(args.model, t_train, x0s, ys, seeds=tuple(range(K)), steps=args.steps)
    print(f"trained {K} {args.model} members, {args.steps} steps each")

    # 3  DE: the members' own predictions
    runs = jnp.stack([rollout(args.model, m, t_eval, x0) for m in members])         # (K, n_time, 2)
    de_mean, de_var = runs.mean(axis=0), runs.var(axis=0)

    # 4  LA: a Laplace approximation of member 0. The wrapper class knows how to flatten the parameters of this model class (for the sPHNN: the raw coordinates, without the fixed equilibrium)
    la = fit_laplace(args.model, members[0], t_train, x0s, ys)       # curvature (GGN) and marginal-likelihood calibration of the prior precision and the noise: la.fit_laplace_mll()
    la_mean, la_var = la.mean_linearized(t_eval, x0), la.std_linearized(t_eval, x0) ** 2     # the mean is the MAP rollout; the variance is linearised through the ODE solve

    # 5  MoLA: one Laplace approximation per member, then the mixture: mean of the means, mean of the variances (within) plus variance of the means (between)
    mola = MoLA_ODESolver([la] + [fit_laplace(args.model, m, t_train, x0s, ys) for m in members[1:]])
    mola_mean, mola_var = mola.mean(t_eval, x0), mola.var(t_eval, x0)

    # 6  Compare: the same mean as DE, and a variance that adds the Laplace variance of every member to the spread between members
    print(f"\n{'':6s}{'RMSE of the mean':>18s}{'std q at t=2':>14s}{'std q at t=10':>15s}")
    i2, i10 = int(jnp.argmin(jnp.abs(t_eval - 2.0))), -1
    for name, mean, var in (("DE", de_mean, de_var), ("LA", la_mean, la_var), ("MoLA", mola_mean, mola_var)):
        rmse = float(jnp.sqrt(jnp.mean((mean - truth) ** 2)))
        print(f"{name:6s}{rmse:18.3f}{float(jnp.sqrt(var[i2, 0])):14.3f}{float(jnp.sqrt(var[i10, 0])):15.3f}")
    print(f"\nMoLA mean equals DE mean: max difference {float(jnp.abs(mola_mean - de_mean).max()):.1e}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for ax, (name, mean, var) in zip(axes, (("DE", de_mean, de_var), ("LA", la_mean, la_var), ("MoLA", mola_mean, mola_var))):
        ax.axvspan(0.0, float(t_train[-1]), color="0.9", label="training window")
        ax.plot(t_eval, truth[:, 0], "k--", lw=1, label="true q")
        ax.plot(t_eval, mean[:, 0], lw=1.5, label="mean")
        ax.fill_between(t_eval, mean[:, 0] - 2 * jnp.sqrt(var[:, 0]), mean[:, 0] + 2 * jnp.sqrt(var[:, 0]), alpha=0.3, label=r"$\pm 2$ std")
        ax.set_title(name)
        ax.set_xlabel("t")
    axes[0].set_ylim(-8, 8)                                  # the LA / MoLA bands are far wider (see the docstring)
    axes[0].set_ylabel("q")
    axes[0].legend(loc="lower left", fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure: {args.out}")


if __name__ == "__main__":
    main()
