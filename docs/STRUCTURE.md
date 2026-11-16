# Repository map

Paths are relative to the repository root. How an active-learning (AL) path runs is described in [AL_WORKFLOW.md](AL_WORKFLOW.md); the generators and tools are listed in [REPRODUCTION.md](REPRODUCTION.md).

```
src/                              library: models, Laplace approximation, MoLA, ensembles, SWAG, acquisition scores, POD
experiments/
  01_cubic_regression/            Section 4.1.1, Appendix B.1
  02_harmonic_oscillator/         Section 4.1.2, Appendix B.2
  03_thermal_food_processing/     Section 4.2.1, Appendix B.3
    al_runs/                        AL workflow: plan, phases, resume, launchers (study/: plan and local results)
    kernels/core/                   building blocks the workflow imports (training recipe, data preparation)
    side_studies/                   n2_landscape, criteria_offline, random_repeats
    figures/                        generators of figures and tables (+ experiment_graph/, al_overview/, authored_tables/)
    metadata/                       seed and provenance manifests
  04_ded/                         Section 4.2.2, Appendix B.4
    al_runs/start_study/            AL workflow of the five first starts (also runs the extension paths on one machine)
    al_runs/final_extension/        five further starts and more Random paths (cluster coordinator)
    kernels/                        core/ (data, POD, training, Laplace scoring), la_de_runner/, mola_acquisition/
    figures/                        generators of figures and tables
    metadata/                       seed and provenance manifests
  shared/                         thesis_plot_style/, cluster_recovery/ (CPU launcher, POD cache, memory release), al_path_repeats/ (adapter that trains and reports DED ensembles)
  cost_analysis/                  Figures B.9 and B.11
runs/                             stored run records and checkpoints
tools/                            recomputation of stored errors, comparison of local runs with stored records
tests/                            unit tests of src/
docs/                             this documentation
data/                             raw simulation data (not in git)
large_assets/                     POD bases and caches (archives, delivered separately)
```

## Folder names in the records

The records keep the names of the original working tree:

| in the records | in this repository |
|---|---|
| `final_experiments/02_harmonic_oscilator/` | `experiments/02_harmonic_oscillator/` |
| `final_experiments/03_AL_chicken/` | `experiments/03_thermal_food_processing/` (`warmstart_path_AL/` is `al_runs/`, `evaluation/` is `figures/`) |
| `final_experiments/04_AL_DED/` | `experiments/04_ded/` |
| `final_experiments/ded_start_study/`, `ded_final_extension/` | `experiments/04_ded/al_runs/start_study/`, `final_extension/` |
| `final_experiments/{cluster_recovery,al_path_repeats,thesis_plot_style}/` | `experiments/shared/` |

## Vocabulary

| term | meaning |
|---|---|
| path | one AL run: one start, one architecture (NODE or sPHNN), one method (Random, DE, LA, MoLA), one criterion (MGE or VAR) |
| n_D | number of training trajectories |
| pool, AP15 | the 15 selectable and the 15 fixed test trajectories of the thermal data set (30 simulations); DED has 25 simulations, all used for evaluation |
| start | the first training trajectory of a path, drawn with a fixed seed and shared by all methods |
| phase, task | phase: train, evaluate or acquire, each in its own process; task: the unit of work of a workflow (one path for thermal, one training or acquisition request for DED) |
| plan | the frozen design of a study (tasks, seeds, Random orders, recipes) with SHA-256 values of data and code |
| seed set A, B, C | independent sets of training keys; A is the set of the production runs |
| kernels | computational steps a workflow imports (data, training recipe, Laplace and MoLA scoring, reporting) |
