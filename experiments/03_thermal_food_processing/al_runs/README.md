# AL and Random paths: 15 starts, 200 tasks

Fresh training at every training-set size (K = 3 members, NODE 30k and sPHNN 20k Adam steps, learning rate 1e-4, L2 1e-6) for DE, LA and MoLA with the criteria `sum_marginal` (MGE) and `sum_variance` (VAR), and for Random. The 15 starts are the 15 trajectories of the selection pool: ten original starts (paths 0-9, drawn without replacement, n_D = 1..15, with Random) and the five remaining trajectories as additional starts (paths 10-14: 745, 795, 861, 873, 905; n_D = 1..5, without Random). Keys `7_000_000 + 100_000 p + 1000 (N-1) + k` (p = path, k = member).

Each path (one start, one architecture, one method) is a task of `study/plan.json`; `python control.py progress` lists the 200 tasks with their numbers. A training-set size takes three phases (train, evaluate, acquire), and every phase runs in a fresh process.

| tasks | paths | numbering |
|---|---|---|
| 0-139 | 0-9 (original starts) | 70 x architecture (NODE 0, sPHNN 1) + 10 x method (Random, DE-MGE, DE-VAR, LA-MGE, LA-VAR, MoLA-MGE, MoLA-VAR) + path |
| 140-199 | 10-14 (additional starts) | 140 + 30 x architecture + 5 x method (DE-MGE, DE-VAR, LA-MGE, LA-VAR, MoLA-MGE, MoLA-VAR) + (path - 10) |

| file | content |
|---|---|
| `design.py` | the design: tasks, seeds, starts, random orders, the frozen plan (`prepare`) and its verification (`verify`) |
| `run.py` | one phase of one task: `train`, `evaluate`, `acquire` (`run` picks the phase from the state on disk) |
| `store.py` | committed records with SHA-256 of their artifacts, the per-path lock and `state`, which returns the next phase |
| `numerical.py` | data preparation, batched inference, error statistics and `acquisition_kernels` (DE, LA, MoLA variances) |
| `control.py` | writes the plan, lists progress, checks the smoke tests of the cluster launcher |
| `run_local.sh`, `launch.sh`, `run_cpu.sbatch` | the shell loop over phases on one machine; the Slurm launcher and job script of the thesis runs |
| `study/` | `plan.json` (the frozen plan) and `results/` (written by local runs) |

**On a single machine (no Slurm)** from the repository root, after `uv sync`:

```bash
python experiments/03_thermal_food_processing/al_runs/control.py progress | head -40   # task numbers
bash experiments/03_thermal_food_processing/al_runs/run_local.sh 30 9                  # task 30 = NODE / LA / sum_marginal / path 0, first 9 phases = n_D = 1..3
python tools/compare_local_thermal.py 30                                                  # member RMSEs and picks next to the stored records
```

`run_local.sh TASK` without the second argument runs the complete path (n_D = 1..15; the additional starts end after n_D = 5). The second argument is the maximum number of phases of this call; every call continues where the previous one stopped (also after an interruption), finished phases are not repeated. Requires `data/chicken_data/dataset.nc` (see `docs/DATA_ACCESS.md`). Results are written to `study/results/<model>/<method>/<criterion>/pathXX/`. On an Apple-silicon laptop the first three training-set sizes of a NODE path take about 15 minutes, and the time grows with the number of training trajectories. The complete study needs a cluster (the 140 paths of the original starts alone took about 2000 hours of training in the stored records).

The runs of the thesis were produced on x86-64 Linux CPUs, so a local run is not bitwise identical: most member RMSEs agree to a few percent, but a single member can end in a different optimum on another CPU (up to a factor of about 3 in the paths checked), while the first acquisitions normally pick the same trajectories (task 30: 899, 861, 874 for n_D = 1..3; task 150, the first additional start: 874, 861 for n_D = 1..2, as stored).

**Stored records.** The records of the thesis are in `runs/03_AL_chicken/warmstart_path_AL/small/results/` (tasks 0-139, same path numbers) and `.../small_extra5/run/results/` (tasks 140-199; the paths 10-14 are stored as `path00`-`path04`). The additional starts were run on the cluster with a variant of the launcher below: one CPU per task, the n_D = 1 ensemble of each (architecture, start) trained once and shared by the six methods (`small_extra5/run/shared_n1/`), and comparisons with earlier paths before release. The seeds are the same as here, so a path of this plan trains the identical n_D = 1 ensemble itself. That variant is not part of this repository.

**Slurm launcher of the thesis runs:** `launch.sh` and `run_cpu.sbatch` are kept as the record of how the 140 tasks of the original starts were launched (smoke-test arrays first, then one task per path). `launch.sh` needs a cluster configuration (`config.local.sh`, see `04_ded/cluster/config.example.sh`); on one machine use `run_local.sh`.

`study/plan.json` fixes the design and records the SHA-256 values of the data and of the code files; every phase checks them. If you edit a code file (`src/`, the files of this folder, `kernels/core/train_common.py` or `chicken_data.py`), delete `study/plan.json` and run `python control.py prepare` to write a new plan. The plans that belong to the stored runs are `experiments/03_thermal_food_processing/metadata/plans/warmstart_small_plan.json` (original starts) and `warmstart_small_extra5_plan.json` (additional starts). This plan contains both: its paths 0-9 and 10-14 agree with them in starts, keys, Random orders, recipes and acquisition settings, and it differs in the file hashes.
