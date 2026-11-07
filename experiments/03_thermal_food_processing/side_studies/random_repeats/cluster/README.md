# Random repeats on the cluster

Cluster-side copies of `landscape_run.py`, `plan.py`, `random_repeat_run.py` and `control.py` of the parent folder, used to train the 880 unique ensembles of the repeated Random baseline (ten sampled paths per start, n_D = 3..5; n_D = 2 reuses the landscape results). `submit.sh` submits the tasks of `../plan.json` to Slurm. Stored records: `runs/03_AL_chicken/investigations/random_repeats/cluster/run/`.
