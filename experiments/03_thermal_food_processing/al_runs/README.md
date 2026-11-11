# AL and Random paths: 15 starts, 200 tasks

The AL workflow of the thermal food processing experiment (steps and code per box of Figure 3.1: [`docs/AL_WORKFLOW.md`](../../../docs/AL_WORKFLOW.md)). Each path (one start, one architecture, one method) is a task of `study/plan.json`; `python control.py progress` lists the 200 tasks. The 15 starts are the 15 trajectories of the selection pool: ten original starts (paths 0-9, n_D = 1..15, with Random) and five additional starts (paths 10-14: 745, 795, 861, 873, 905; n_D = 1..5, without Random).

| tasks | paths | numbering |
|---|---|---|
| 0-139 | 0-9 (original starts) | 70 × architecture (NODE 0, sPHNN 1) + 10 × method (Random, DE-MGE, DE-VAR, LA-MGE, LA-VAR, MoLA-MGE, MoLA-VAR) + path |
| 140-199 | 10-14 (additional starts) | 140 + 30 × architecture + 5 × method (DE-MGE, DE-VAR, LA-MGE, LA-VAR, MoLA-MGE, MoLA-VAR) + (path − 10) |

| file | content |
|---|---|
| `design.py` | the design: tasks, seeds, starts, random orders, the frozen plan (`prepare`) and its verification (`verify`) |
| `run.py` | one phase of one task: `train`, `evaluate`, `acquire` (`run` picks the phase from the state on disk) |
| `store.py` | committed records with SHA-256 of their artifacts, the per-path lock and `state`, which returns the next phase |
| `numerical.py` | data preparation, batched inference, error statistics and `acquisition_kernels` (DE, LA, MoLA variances) |
| `control.py` | writes the plan, lists progress, checks the smoke tests of the cluster launcher |
| `run_local.sh`, `launch.sh`, `run_cpu.sbatch` | the shell loop over phases on one machine; the Slurm launcher and job script of the thesis runs |
| `study/` | `plan.json` (the frozen plan) and `results/` (written by local runs) |

## Running a path on one machine

From the repository root, after `uv sync`; requires `data/chicken_data/dataset.nc` ([`docs/DATA_ACCESS.md`](../../../docs/DATA_ACCESS.md)):

```bash
python experiments/03_thermal_food_processing/al_runs/control.py progress | head -40   # task numbers
bash experiments/03_thermal_food_processing/al_runs/run_local.sh 30 9                  # task 30 = NODE / LA / sum_marginal / path 0, first 9 phases = n_D = 1..3
python tools/compare_local_thermal.py 30                                               # member RMSEs and picks next to the stored records
```

Without the second argument `run_local.sh TASK` runs the complete path. The second argument is the maximum number of phases of this call; every call continues where the previous one stopped, and finished phases are not repeated. Results are written to `study/results/<model>/<method>/<criterion>/pathXX/`. On an Apple-silicon laptop the first three training-set sizes of a NODE path take about 15 minutes, and the time grows with the number of training trajectories. The 140 paths of the original starts alone took about 2000 hours of training.

The thesis runs used x86-64 Linux CPUs, so a local run is not bitwise identical: most member RMSEs agree to a few percent, a single member can end in a different optimum (up to a factor of about 3 in the paths checked), and the first acquisitions normally pick the same trajectories (task 30: 899, 861, 874 for n_D = 1..3; task 150: 874, 861 for n_D = 1..2, as stored).

## Stored records and plan

The records of the thesis are in `runs/03_AL_chicken/warmstart_path_AL/small/results/` (tasks 0-139) and `.../small_extra5/run/results/` (tasks 140-199; paths 10-14 are stored as `path00`-`path04`). The additional starts were run on the cluster with a variant of the launcher (shared n_D = 1 ensemble per architecture and start, comparisons with earlier paths before release); it is not part of this repository, and a path of this plan trains the identical n_D = 1 ensemble itself, with the same seeds.

`study/plan.json` fixes the design and the SHA-256 values of data and code; every phase checks them. After editing a code file (`src/`, the files of this folder, `kernels/core/train_common.py`, `chicken_data.py`), delete the plan and run `python control.py prepare`. The plans of the stored runs are `experiments/03_thermal_food_processing/metadata/plans/warmstart_small_plan.json` and `warmstart_small_extra5_plan.json`; this plan agrees with them in starts, keys, Random orders, recipes and acquisition settings and differs in the file hashes. `launch.sh` needs a cluster configuration (`config.local.sh`, see `04_ded/cluster/config.example.sh`).
