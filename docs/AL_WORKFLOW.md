# Active-learning workflow

How the two active-learning (AL) experiments run, step by step. Paths are relative to the repository root.

## The protocol of Figure 3.1

![Figure 3.1: the trajectory-pool AL loop](figures/al_flowchart.png)

*Figure 3.1 of the thesis (generator: `experiments/03_thermal_food_processing/figures/experiment_graph/al_flowchart.py`). The experiments stop at the fixed budget n_D = n_D,max; the optional tolerance E_val ≤ E_tol is not used. UQ = uncertainty quantification.*

The Random baseline selects uniformly from the same pool. A *path* is one run of this loop (one start, one architecture, one method); the experiments are many paths.

The two experiments implement this protocol with separate workflow code. They share the library in `src/` (Laplace approximation, MoLA, ensembles and the training-RMSE filter, acquisition scores), the launcher and hardware snapshot in `experiments/shared/cluster_recovery/`, and the plotting style.

Both workflows are built for long cluster runs and therefore share three properties:

- **One phase per process.** A path is not one long Python process. Train, evaluate and acquire of every training-set size are separate calls of a script, each in a fresh process.
- **State on disk.** What is done is recorded by committed result files (JSON with the SHA-256 of the artifacts). A script reads the committed files to see what comes next, so a run can be stopped and started again at any time and finished phases are never repeated.
- **Frozen plan.** A plan file fixes the design (starts, seeds, recipes, settings) and the SHA-256 of the data and code files. Every phase checks them and refuses to run if they changed.

---

## Thermal food processing (`experiments/03_thermal_food_processing/al_runs/`)

200 paths = *tasks* in `study/plan.json`: 2 architectures (NODE, sPHNN) × methods × 15 starts. Ten starts run to n_D = 15 (with Random), five to n_D = 5 (without Random). A path has 3 · n_D,max − 1 phases (train, evaluate, acquire per size; the last size has no acquire). `iterNN` in file names is n_D − 1. Commands: `al_runs/README.md`.

### Figure 3.1 and the code

| box | where | what it does |
|---|---|---|
| Start | `design.py::warmstarts`, `store.py::state` | the start trajectory of the path (`warmstart_ids` of the plan) is the first element of the training set |
| Train n_ens models | `run.py::train` → `kernels/core/train_common.py::train_ensemble` | K = 3 members from scratch, full-batch Adam (NODE 30k, sPHNN 20k steps, lr 1e-4, L2 1e-6), seeds `7_000_000 + 100_000·path + 1000·(n_D−1) + k` |
| Evaluate models | `run.py::evaluate` | training filter (members whose training RMSE is 5 × the median or more are excluded from the ensemble statistics), then rollouts on the 15 fixed test trajectories (AP15) and member RMSEs in K |
| n_D ≥ n_D,max? | `store.py::state` | returns `done` when the report of the last size exists (`max_N` of the path in the plan) |
| Score pool | `run.py::acquire` → `numerical.py::acquisition_kernels`, `src/active_learning.py` | predictive variance of the two probe temperatures over the rollout of every candidate (DE: variance over the retained members; LA: Laplace variance around member 0; MoLA: mean of the Laplace variances plus variance of the linearised means), then the scores MGE (`score_sum_marginal`) and VAR (`score_sum_variance`) |
| Select candidate | `run.py::acquire` | the candidate with the largest score of the task's criterion (ties: smaller trajectory id) |
| Acquire and add | `selections/iterNN.json`, `store.py::state` | the pick is stored; the next call of `state` appends it to the training set |
| (loop) | `run_local.sh` | the shell loop that starts one phase after the other |

### Differences from Figure 3.1

- **Random** does not draw at every step: the plan fixes, for every path, the order in which Random adds trajectories (`random_orders`), which is a uniform draw without replacement from the same pool.
- **Evaluate** only reports. The test errors never enter the selection, and the tolerance E_val ≤ E_tol is not used; the stop is the fixed budget.
- **Score pool** always computes both criteria. The task's criterion decides the pick; the other ranking is stored as well.
- **Training filter.** A member whose training fit failed is excluded from the ensemble statistics (DE, MoLA) and from the reported ensemble mean; a failed member 0 stops an LA path instead of being replaced.

---

## DED (`experiments/04_ded/al_runs/start_study/`)

The same loop, organised differently. All paths of a plan advance together, size by size: a coordinator (`control.py`) collects the work of one training-set size from all paths, removes duplicates, and hands it out as *tasks* (one training or one acquisition each), every task in a fresh process (`worker.py`). A path is one model (NODE, sPHNN), one start trajectory and one *stream*: `LA`, `DE`, `MoLA` or `random0` .. `random5`. Path ids look like `NODE/start12/LA`. The thesis runs have ten starts (the five of `start_study` and the five of `../final_extension`), n_D = 1..5, and the MGE criterion only. Identical training requests are trained once: all paths of a model and start share the n_D = 1 ensemble. Commands: `start_study/README.md`.

### Figure 3.1 and the code

| box | where | what it does |
|---|---|---|
| Start | `study_design.py::plan`, `control.py::frontier` | the start simulation of the path is the training set at n_D = 1 |
| Train n_ens models | `study_numerical.py::train` → `experiments/shared/al_path_repeats/repeat_adapter.py` → `kernels/core/_ded_al_core.py::train_hybrid_stages` | K = 3 members from scratch on the POD coordinates of the training set (40 modes, fitted on exactly this set): derivative fitting (20k steps), then trajectory fine-tuning (NODE 15k, sPHNN 10k steps) |
| Evaluate models | the same task, `repeat_adapter.py::filter_and_report` | training filter (training RMSE 5 × the median or more), then the Kelvin RMSE of every member on all 25 simulations; stored in `train.json`, published per path by `control.py::publish_paths` |
| n_D ≥ n_D,max? | `control.py::advance_step` | the loop over n_D ends at `--through-N` (at most 5, the `max_N` of the plan) |
| Score pool | `study_numerical.py::acquire` | variance of the rollout of every simulation that is not in the training set (LA: separate single-member model with Laplace fit, key `9_500_000 + …`; DE: variance over the retained reporting members; MoLA: uniform mixture over members 0, 1, 2, each with its own Laplace approximation, in `kernels/mola_acquisition/`), then the MGE score (`src/active_learning.py::score_sum_marginal`) |
| Select candidate | `study_store.py::ranking` | descending score, ties: smaller id; the first entry is the pick (`selection.json`) |
| Acquire and add | `control.py::frontier` | the training set of size n_D + 1 is built from the committed selection of size n_D |

### Differences from Figure 3.1

- **Evaluate** is part of the training task, and the test set is the full set of 25 simulations, including the training and pool simulations. There is no held-out test set; the errors are only reported.
- **Random** paths have no acquisition tasks: their training sets follow a predeclared order (`study_design.py::tail`).
- **Stop** is a command-line limit of the coordinator, not a decision of the path.
