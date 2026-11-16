"""Example 2: trajectory-pool active learning (Figure 3.2 of the thesis) on the harmonic oscillator.

A pool of candidate initial states is given. Each candidate is a complete trajectory whose observation is released only when the candidate is acquired. Starting from one trajectory, the loop

    train K models -> evaluate -> stop at the budget -> fit the UQ method -> score the pool -> acquire the highest-scored candidate -> (repeat)

runs for the UQ methods DE, LA and MoLA with the acquisition score MGE (sum of the log marginal variances) or VAR (sum of the marginal variances), and for a Random baseline.
Every box of Figure 3.2 is one line of `run_path`. The thermal-food-processing and DED experiments run this same loop; they only organise it for long cluster runs
(one phase per process, state on disk, frozen plan), see docs/AL_WORKFLOW.md.

To keep the loop short, the scores and the test error are both computed on the training window of half a period; the test error therefore measures the fit inside the window, not the extrapolation.
A single path of this toy problem is noisy, and the paths of different methods can coincide when they pick the same candidates; the table shows how the loop works, not which method is better.
The statements of the thesis rest on many starts and paths.

Run from the repository root:   python examples/02_active_learning.py [--model sPHNN|NODE] [--methods DE LA MoLA] [--steps 1500]      (about 8 minutes on a laptop CPU; --n-max 3 is faster)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _oscillator import MODEL_CLASSES, base, fit_laplace, rollout, train_ensemble

import jax
import jax.numpy as jnp
import jax.random as jrd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.active_learning import score_sum_marginal, score_sum_variance
from src.mola import MoLA_ODESolver

K = 3                                                                   # models per training set
OMEGA = base.oscillator_frequency()
T_GRID = jnp.linspace(0.0, base.training_t_end(), 30)                  # every trajectory is observed (and predicted) over half a period
NOISE_STD = 0.1
SCORES = {"MGE": score_sum_marginal, "VAR": score_sum_variance}


def state(amplitude, phase_deg):
    """Initial state (q, p) of the oscillator for an amplitude and a phase."""
    phase = jnp.deg2rad(phase_deg)
    return amplitude * jnp.array([jnp.cos(phase), OMEGA * jnp.sin(phase)])


# The pool: 3 amplitudes x 4 phases. The start is amplitude 3, phase 0. The test set (never trained on, never acquired) lies between and beyond the pool.
POOL = [(a, ph) for a in (1.5, 3.0, 4.5) for ph in (0, 90, 180, 270)]
START = POOL.index((3.0, 0))
TEST = jnp.stack([state(a, ph) for a in (1.0, 2.25, 3.75, 5.0) for ph in (45, 135, 225, 315)])


def oracle(i):
    """The (noisy) observation of pool candidate i: what an experiment or a simulation returns after the candidate is acquired."""
    clean = base.true_trajectory(state(*POOL[i]), T_GRID)
    return clean + NOISE_STD * jrd.normal(jrd.fold_in(jrd.key(123), i), clean.shape)


def test_error(model_class, members):
    """Node error of the thesis: the median over the K members of the RMSE on the test trajectories."""
    truth = jax.vmap(lambda x0: base.true_trajectory(x0, T_GRID))(TEST)
    rmse = [float(jnp.sqrt(jnp.mean((jax.vmap(lambda x0: rollout(model_class, m, T_GRID, x0))(TEST) - truth) ** 2))) for m in members]
    return float(np.median(rmse))


def fit_uq(method, model_class, members, x0s, ys):
    """Fit the UQ method on the current training set; returns a function x0 -> predictive variance of the rollout, shape (n_time, 2)."""
    if method == "DE":                                                  # spread of the K members' rollouts
        return lambda x0: jnp.stack([rollout(model_class, m, T_GRID, x0) for m in members]).var(axis=0)
    if method == "LA":                                                  # linearised Laplace approximation around member 0
        la = fit_laplace(model_class, members[0], T_GRID, x0s, ys)
        return lambda x0: la.std_linearized(T_GRID, x0) ** 2
    mola = MoLA_ODESolver([fit_laplace(model_class, m, T_GRID, x0s, ys) for m in members])          # MoLA: one Laplace approximation per member, mixture
    return lambda x0: mola.var(T_GRID, x0)


def run_path(model_class, method, criterion, n_max, steps, train, rng):
    """One AL path from the start trajectory up to n_max trajectories. `method` is DE, LA, MoLA or Random. Returns the picks and the test error after every training."""
    train_ids, errors = [START], []
    while True:
        n_D = len(train_ids)
        x0s = jnp.stack([state(*POOL[i]) for i in train_ids])
        ys = jnp.stack([oracle(i) for i in train_ids])                       # released observations of the acquired candidates
        members = train(tuple(train_ids), x0s, ys, steps)                    # Train K models on the training set
        errors.append(test_error(model_class, members))                      # Evaluate models (only reported, never used for the selection)
        if n_D >= n_max:                                                     # n_D >= n_D,max?  -> stop
            return train_ids, errors
        pool = [i for i in range(len(POOL)) if i not in train_ids]           # remaining candidates
        if method == "Random":
            pick = int(rng.choice(pool))                                     # Random baseline: uniform from the same pool
        else:
            variance = fit_uq(method, model_class, members, x0s, ys)         # Score pool, part 1: predictive variance of every candidate's rollout
            scores = [float(SCORES[criterion](variance(state(*POOL[i])))) for i in pool]            # part 2: acquisition score
            pick = pool[int(np.argmax(scores))]                              # Select candidate: the highest score
        train_ids.append(pick)                                               # Acquire and add: the observation of the pick joins the training set


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_CLASSES, default="sPHNN")
    ap.add_argument("--methods", nargs="+", default=["DE", "LA", "MoLA"], choices=["DE", "LA", "MoLA"])
    ap.add_argument("--criterion", choices=list(SCORES), default="MGE")
    ap.add_argument("--n-max", type=int, default=4, help="training-set size at which the loop stops")
    ap.add_argument("--steps", type=int, default=1500, help="training steps per model (the experiments use 5000 to 30000)")
    ap.add_argument("--random-paths", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output" / "active_learning.png")
    args = ap.parse_args(argv)

    trained = {}                                                             # paths with the same training set share its models (the seeds depend on the size only)

    def train(train_ids, x0s, ys, steps):
        if train_ids not in trained:
            seeds = tuple(1000 * (len(train_ids) - 1) + k for k in range(K))
            trained[train_ids] = train_ensemble(args.model, T_GRID, x0s, ys, seeds=seeds, steps=steps)
        return trained[train_ids]

    paths = {m: run_path(args.model, m, args.criterion, args.n_max, args.steps, train, None) for m in args.methods}
    for r in range(args.random_paths):
        paths[f"Random {r + 1}"] = run_path(args.model, "Random", args.criterion, args.n_max, args.steps, train, np.random.default_rng(r))

    print(f"\ntest RMSE (median over {K} models) after training on n_D trajectories; picks as (amplitude, phase)")
    print(f"{'':10s}" + "".join(f"{'n_D=' + str(n):>9s}" for n in range(1, args.n_max + 1)) + "   picks")
    for name, (ids, errors) in paths.items():
        print(f"{name:10s}" + "".join(f"{e:9.3f}" for e in errors) + "   " + " ".join(str(POOL[i]) for i in ids[1:]))

    fig, ax = plt.subplots(figsize=(5, 3.2))
    styles = {"DE": "o-", "LA": "s--", "MoLA": "^:"}
    for name, (ids, errors) in paths.items():
        ax.plot(range(1, args.n_max + 1), errors, styles.get(name, "o-"), color="0.7" if name.startswith("Random") else None, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("training trajectories $n_D$")
    ax.set_ylabel("test RMSE")
    ax.legend(fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure: {args.out}")


if __name__ == "__main__":
    main()
