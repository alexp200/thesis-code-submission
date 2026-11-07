"""Training schedule of the DED runs: the hyperparameters of the two training stages per architecture and the entry point `train_ensemble`.

Stage 1 fits the vector field to the time derivatives of the latent trajectories (20k steps, learning rate 1e-3, no L2). Stage 2 fine-tunes on the trajectories with the loss
MSE + 0.5 * lambda * ||theta_laplace||^2: NODE 15k steps at 1e-4 with lambda = 1e-5, sPHNN 10k steps at 1e-5 with lambda = 1e-6.
"""

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)  # before any array or model is built; the GGN / marginal-likelihood calibration is numerically sensitive to it

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ded_data import DATA_PATH
import _ded_al_core as core

NODE_DERIVATIVE_STEPS = 20_000
NODE_DERIVATIVE_LR = 1e-3
NODE_FINETUNE_STEPS = 15_000
NODE_FINETUNE_LR = 1e-4

SPHNN_DERIVATIVE_STEPS = 20_000
SPHNN_DERIVATIVE_LR = 1e-3
SPHNN_FINETUNE_STEPS = 10_000
SPHNN_FINETUNE_LR = 1e-5

FINAL_TRAJECTORY_L2 = {
    "NODE": 1e-5,
    "sPHNN": 1e-6,
}


def training_hyperparams_for_meta(
        cfg: core.ModelClassConfig,
        trajectory_l2: float,
        ) -> dict:
    """The training hyperparameters of an architecture, as recorded with every trained ensemble."""
    if cfg.name == "NODE":
        return {
            "procedure": "derivative_then_trajectory_finetune",
            "derivative_steps": NODE_DERIVATIVE_STEPS,
            "derivative_lr": NODE_DERIVATIVE_LR,
            "finetune_steps": NODE_FINETUNE_STEPS,
            "finetune_lr": NODE_FINETUNE_LR,
            "derivative_l2": 0.0,
            "finetune_l2": trajectory_l2,
            "finetune_objective": "MSE + 0.5 * finetune_l2 * ||theta_laplace||^2",
        }
    elif cfg.name == "sPHNN":
        return {
            "procedure": "derivative_then_trajectory_finetune",
            "derivative_steps": SPHNN_DERIVATIVE_STEPS, "derivative_lr": SPHNN_DERIVATIVE_LR,
            "finetune_steps": SPHNN_FINETUNE_STEPS, "finetune_lr": SPHNN_FINETUNE_LR,
            "derivative_l2": 0.0,
            "finetune_l2": trajectory_l2,
            "finetune_objective": "MSE + 0.5 * finetune_l2 * ||theta_laplace||^2",
        }
    raise ValueError(f"Unknown model class: {cfg.name!r}")


def train_ensemble(
        cfg: core.ModelClassConfig,
        data: dict,
        ensemble_size: int,
        trajectory_l2: float,
        key: jax.Array,
        ):
    """Train either architecture with its two-stage schedule; returns (ensemble, history, seconds)."""
    return core.train_ensemble_for_model_class(
        cfg, data, ensemble_size, key,
        node_derivative_steps=NODE_DERIVATIVE_STEPS,
        node_derivative_lr=NODE_DERIVATIVE_LR,
        node_finetune_steps=NODE_FINETUNE_STEPS,
        node_finetune_lr=NODE_FINETUNE_LR,
        sphnn_derivative_steps=SPHNN_DERIVATIVE_STEPS, sphnn_derivative_lr=SPHNN_DERIVATIVE_LR,
        sphnn_finetune_steps=SPHNN_FINETUNE_STEPS, sphnn_finetune_lr=SPHNN_FINETUNE_LR,
        trajectory_l2_regularization=trajectory_l2,
    )
