"""Run the stages of the SWAG learning-rate study for ONE (model, data set).

  --stage sweep       7 learning rates x the first 4 converged members
  --stage comparison  the learning rate of the comparisons (ens_lib.SWAG_LR) for ALL converged members

Every configuration is cached (cache/), reruns are free.
Usage: python -u run_sweep.py --model sPHNN --dataset lower_half --stage sweep
"""

import argparse

import swag_dep_lib as L


def main():
    """Command-line entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=L.MODELS, required=True)
    ap.add_argument("--dataset", choices=L.DATASETS, required=True)
    ap.add_argument("--stage", choices=["sweep", "comparison"], required=True)
    a = ap.parse_args()
    mc, ds = a.model, a.dataset
    if a.stage == "sweep":
        members = L.sweep_members(mc, ds)
        L.log(f"{mc}/{ds}: members {members}, lr grid {L.LR_GRID}")
        for k in members:
            for lr in L.LR_GRID:
                L.run_config(mc, ds, k, lr)
    else:
        for k in L.kept_members(mc, ds):
            L.run_config(mc, ds, k, L.ens_lib.SWAG_LR)


if __name__ == "__main__":
    main()
