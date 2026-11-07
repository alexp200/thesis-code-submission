# Repository map

Where to look, by what you want to do. Paths are relative to the repository root.

| I want to ...                                        | go to                                                                                                                                                                                     |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| regenerate a figure or table of the thesis           | the generator table in [REPRODUCTION.md](REPRODUCTION.md) (Level 1)                                                                                                                       |
| recompute stored model errors                        | `tools/recompute_*.py` ([REPRODUCTION.md](REPRODUCTION.md), Level 2)                                                                                                                      |
| see how an active-learning (AL) path is run          | [AL_WORKFLOW.md](AL_WORKFLOW.md); the code is in `experiments/03_thermal_food_processing/al_runs/` and `experiments/04_ded/al_runs/start_study/`; both have a README and a `run_local.sh` |
| see the Laplace approximation, MoLA, ensembles, SWAG | `src/` (library, with tests in `tests/`)                                                                                                                                                  |
| find where a seed or key came from                   | `experiments/*/metadata/`                                                                                                                                                                 |

## Layout

```
src/                              library: models, Laplace approximation, MoLA, ensembles, SWAG, acquisition scores, POD
experiments/
  01_cubic_regression/            Section 4.1.1, Appendix B.1
  02_harmonic_oscillator/         Section 4.1.2, Appendix B.2
  03_thermal_food_processing/     Section 4.2.1, Appendix B.3
    al_runs/                        AL workflow: plan, phases, resume, launchers (study/: plan and local results)
    kernels/core/                   computational building blocks the workflow imports (training recipe, data preparation)
    side_studies/                   n2_landscape, criteria_offline, random_repeats
    figures/                        generators of the figures and tables (+ experiment_graph/, al_overview/, authored_tables/)
    metadata/                       seed and provenance manifests
  04_ded/                         Section 4.2.2, Appendix B.4
    al_runs/start_study/            AL workflow of the five first starts (also runs the extension paths on one machine)
    al_runs/final_extension/        five further starts and more Random paths (cluster coordinator)
    kernels/                        core/ (data, POD, training, Laplace scoring), la_de_runner/, mola_acquisition/
    figures/                        generators of the figures and tables
    metadata/                       seed and provenance manifests
  shared/                         thesis_plot_style/ (figure style and sizes), cluster_recovery/ (CPU launcher, POD cache, memory release), al_path_repeats/ (adapter that trains and reports DED ensembles)
  cost_analysis/                  Figures B.8 and B.10
runs/                             stored run records and checkpoints (folder names of the original runs)
tools/                            recomputation of stored errors, comparison of local runs with the stored records
tests/                            unit tests of src/
docs/                             environment, data access, reproduction, this map
data/                             raw simulation data (not in git, see DATA_ACCESS.md)
large_assets/                     POD bases and caches (archives, delivered separately, not in git)
```

## The AL loop and its vocabulary

One **path** is a sequence of acquisitions from one **start** for one architecture (NODE or sPHNN), one **method** (Random, DE, LA, MoLA) and one **criterion** (MGE = sum of log marginal variances, VAR = sum of marginal variances):

1. Start with the training set {start} (n_D = 1). The start is one trajectory of the selection pool.
2. **train**: a new ensemble (K = 3) is trained from scratch on the current training set.
3. **evaluate**: member RMSEs on the fixed test trajectories; members with a training RMSE above 5 times the median are marked as dropped.
4. **acquire**: every remaining pool trajectory gets an acquisition score and the best one is added to the training set (Random follows a fixed permutation). Back to step 2 until n_D = 15 (thermal) or 5 (DED).

| term | meaning |
|---|---|
| kernels | the computational steps a workflow imports: data preparation, training recipe, Laplace and MoLA scoring, reporting. The workflow (`al_runs/`) only plans, runs the phases and commits their results |
| n_D | number of training trajectories |
| pool, AP15 | the 15 selectable trajectories and the 15 fixed test trajectories of the thermal data set (30 simulations, called `small`); DED has 25 simulations, all used for evaluation |
| warm start | the start trajectory, drawn at random with a fixed seed and shared by all methods of a path |
| task | the unit of work of a workflow: one path (thermal), one training or acquisition request (DED) |
| phase | train, evaluate or acquire; each runs in its own process and is committed atomically, so a run can be resumed |
| plan | the frozen design of a study (tasks, seeds, Random orders, recipes) with the SHA-256 values of data and code |
| seed set A, B, C | independent sets of training keys; A is the set of the production runs |
| n2_landscape | landscape of all n_D = 2 training sets {start, candidate} (all possible first acquisitions) |
| L2, K3, p10 | L2 regularization, ensemble of three members, ten paths |

## Data flow

raw simulation data (`data/`, not in git) -> AL workflows (Level 3) -> run records and checkpoints (`runs/`) -> generators (`experiments/*/figures/`, Level 1) -> figures and tables (`thesis_figures/`). Level 2 recomputes the errors in the records from the checkpoints and the raw data.
