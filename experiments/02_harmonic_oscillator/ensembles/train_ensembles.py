"""Train the K = 10 MAP ensemble of one model class on one data set; every method in this folder starts from these members.

Recipe: Adam with learning rate 1e-3, 5000 full-batch steps, objective MSE + 0.5 * lambda * ||theta||^2 with lambda = 1e-4, noisy data of seed 17,
member seeds 0..9. The sPHNN is conservative: no dissipation matrix and no input matrix (ensemble_lib.build_sphnn_conservative_member).
Writes the checkpoint to checkpoints/ and the training RMSE of every member to training/training_<model>_<dataset>.json; an existing checkpoint is kept unless --force is given.

Usage:  python -u train_ensembles.py --model-class sPHNN --dataset lower_half
"""

from __future__ import annotations

import argparse
import json

import ensemble_lib as ens_lib  # (sets jax_enable_x64 first)

import equinox as eqx
import jax
import jax.random as jrd
import klax
import optax


def main():
    """Command-line entry: train and store one ensemble."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-class", choices=ens_lib.MODEL_CLASSES, required=True)
    parser.add_argument("--dataset", choices=ens_lib.DATASETS, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ckpt = ens_lib.checkpoint_path(args.model_class, args.dataset)
    summary_path = ens_lib.TRAINING_DIR / f"training_{args.model_class}_{args.dataset}.json"
    if ckpt.exists() and summary_path.exists() and not args.force:
        print(f"[skip] {ckpt.name} exists")
        return

    # --- data, untrained ensemble and the training loss
    dataset = ens_lib.build_dataset(args.dataset)
    ensemble = ens_lib.build_ensemble(args.model_class, ens_lib.SEEDS)
    members0 = ens_lib.split_ensemble(ensemble, ens_lib.K_ENSEMBLE)
    n_params = ens_lib.base.active_parameter_count(members0[0], args.model_class)
    loss = ens_lib.base.TrajectoryMSEL2(args.model_class, ens_lib.L2)

    print(f"[train] {args.model_class}/{args.dataset}: K={ens_lib.K_ENSEMBLE}, {n_params} active params, "
          f"{ens_lib.STEPS} steps, lambda={ens_lib.L2:g}", flush=True)
    # --- train all K members at once (vmapped ensemble, full batch)
    ensemble, _ = klax.fit(
        ensemble,
        (dataset["t_train"], dataset["x0s"], dataset["ys_noisy"]),
        batch_axes=(None, 0, 0),
        batch_size=ens_lib.base.N_TRAJECTORIES,
        steps=ens_lib.STEPS,
        loss=loss,
        optimizer=optax.adam(ens_lib.LEARNING_RATE),
        make_logger=False,
        key=jrd.fold_in(
            jrd.key(ens_lib.TRAINING_KEY),
            ens_lib.MODEL_KEY_OFFSETS[args.model_class] + ens_lib.DATASET_KEY_OFFSETS[args.dataset],
        ),
        vmap_ensemble=True,
    )
    jax.block_until_ready(ensemble)

    # --- training RMSE of every member (convergence filter of compute_methods.py)
    members = ens_lib.split_ensemble(ensemble, ens_lib.K_ENSEMBLE)
    per_seed = []
    for seed, member in zip(ens_lib.SEEDS, members):
        m = {"seed": seed, **ens_lib.base.train_rmse(args.model_class, member, dataset)}
        per_seed.append(m)
        print(f"  seed {seed}: clean_rmse={m['train_rmse_clean']:.4f}", flush=True)

    # --- checkpoint and summary
    ens_lib.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ens_lib.TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(ckpt, ensemble)
    summary_path.write_text(json.dumps({
        "model_class": args.model_class,
        "dataset": args.dataset,
        "K": ens_lib.K_ENSEMBLE,
        "seeds": list(ens_lib.SEEDS),
        "steps": ens_lib.STEPS,
        "learning_rate": ens_lib.LEARNING_RATE,
        "l2": ens_lib.L2,
        "dataset_seed": ens_lib.DATASET_SEED,
        "active_parameter_count": n_params,
        "per_seed": per_seed,
    }, indent=2) + "\n")
    print(f"[done] -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
