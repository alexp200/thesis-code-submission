"""Shared setup of the two examples: the harmonic oscillator of the experiments (undamped, state (q, p)), its two model classes and the training of an ensemble.

Data, model builders and the loss come from experiments/02_harmonic_oscillator; nothing is re-implemented here. `ens_lib` enables float64, which the Laplace approximation needs."""
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "02_harmonic_oscillator" / "ensembles"))
import ensemble_lib as ens_lib                           # also puts the repository root on sys.path, so that `import src` works

import jax.random as jrd
import klax
import optax

base = ens_lib.base                                      # oscillator_lib: true solution, NODE builder, trajectory loss
MODEL_CLASSES = ens_lib.MODEL_CLASSES                    # ("NODE", "sPHNN"); the sPHNN is the conservative one of the experiments


def train_ensemble(model_class, t, x0s, ys, seeds, steps):
    """Train one member per seed from scratch on the trajectories ys (n_traj, n_time, 2) observed from the initial states x0s; returns the list of members.

    Objective: trajectory MSE + 0.5 * lambda * ||theta||^2 (the training of the experiments, full batch Adam)."""
    ensemble = ens_lib.build_ensemble(model_class, seeds)             # all members at once (vmapped), different seeds = different initialisations
    ensemble, _ = klax.fit(ensemble, (t, x0s, ys), batch_axes=(None, 0, 0), batch_size=len(x0s), steps=steps,
                           loss=base.TrajectoryMSEL2(model_class, ens_lib.L2), optimizer=optax.adam(ens_lib.LEARNING_RATE),
                           make_logger=False, key=jrd.key(seeds[0]), vmap_ensemble=True)
    return ens_lib.split_ensemble(ensemble, len(seeds))


def rollout(model_class, member, t, x0):
    """The trajectory (n_time, 2) of one trained member from the initial state x0 (the MAP prediction)."""
    return ens_lib.callable_member(model_class, member)(t, x0)


def fit_laplace(model_class, member, t, x0s, ys):
    """Laplace approximation around one trained member: curvature (GGN) of the trajectory loss, then calibration of the prior precision and the observation noise by the marginal likelihood.

    The wrapper class of the model class (src.models.Laplace_NeuralODESolver or Laplace_sPHNNSolver) is chosen by `ens_lib.laplace_class`; the library's progress output is not shown."""
    with contextlib.redirect_stdout(io.StringIO()):
        la = ens_lib.laplace_class(model_class)(member, t, x0s, ys)
        la.fit_laplace_mll()
    return la
