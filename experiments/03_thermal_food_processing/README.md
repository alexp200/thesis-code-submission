# Thermal food processing (Section 4.2.1, Appendix B.3)

Active learning with NODE and sPHNN surrogates on simulated thermal food processing trajectories (the identifier "chicken" in the code, in `data/chicken_data` and in `runs/03_AL_chicken/` is a historical name). Starts: 15 selection-pool trajectories; test set: 15 fixed trajectories (AP15).

## Layout

| folder | content |
|---|---|
| `al_runs/` | the AL workflow: training, evaluation and acquisition of the AL and Random paths of the ten original starts (n_D = 1..15) and of the five additional starts (n_D = 1..5), 200 tasks in `study/plan.json`. `control.py` prepares the plan and checks progress, `run.py` runs one phase of one path, `run_local.sh` runs a path on one machine, `launch.sh` is the Slurm launcher of the thesis runs |
| `kernels/core/` | training recipe (`train_common.py`) and normalisation (`chicken_data.py`) imported by the workflow and the side studies |
| `side_studies/n2_landscape/` | exhaustive first acquisition: a model for every second trajectory of every start (`landscape_run.py`), its table (`landscape_analysis.py`), the n_D = 1 models of the additional starts and of seed sets B, C (`n1_run.py`); stored results in `merged/`, `n1_models/` |
| `side_studies/random_repeats/` | ten sampled Random paths per start (`plan.py`, `random_repeat_run.py`, `random_repeats_analysis.py`); stored results in `results/`, Slurm copies in `cluster/` |
| `side_studies/criteria_offline/` | acquisition scores (MGE, VAR) of the n_D = 1 models for seed sets A, B, C (`criteria_offline.py`, `acq_criteria.py`); stored in `results/` |
| `figures/` | `make_results_figures.py`: Figs. 4.8-4.11, B.7 and Tabs. B.14-B.21; `make_dataset_figure.py`: Fig. 4.7; `experiment_graph/`: schematics (Figs. 3.1, B.5, B.6); `al_overview/studies_io.py`: loader of the AL run records; `authored_tables/`: hand-written tables B.11-B.13, copied by the generator |
| `common/` | output locations of the generators |
| `metadata/` | seed manifests: which training used which keys and data (see its README) |

## How the AL loop is organised

One path = one start, one architecture, one method; the 200 paths are the tasks of `al_runs/study/plan.json`: 2 architectures x 7 methods x 10 original starts (tasks 0-139) and 2 architectures x 6 methods x 5 additional starts (tasks 140-199); `python al_runs/control.py progress` lists them.
A path alternates three phases per training-set size n_D, each in a fresh process (`al_runs/run.py --task T`):

1. **train**: K = 3 members from scratch on the current training set (`kernels/core/train_common.py`);
2. **evaluate**: training filter (RMSE > 5 x median) and member errors on the AP15 test set;
3. **acquire**: Random follows a predeclared order; DE, LA, MoLA score every remaining pool trajectory with both criteria (`numerical.acquisition_kernels`, `src/active_learning.py`) and pick the best of the path's criterion.

`al_runs/store.py` keeps the state on disk (committed records with SHA-256 of their artifacts), so a path resumes where it stopped; the shell loop that repeats `run.py` is `run_local.sh` (one machine) or `run_cpu.sbatch` (Slurm).

## Pipeline

| step | script | output |
|---|---|---|
| 1. AL and Random paths | `al_runs/run_local.sh TASK` or the Slurm launchers | `al_runs/study/results/` (the stored records of the thesis: `runs/03_AL_chicken/`) |
| 2. first-acquisition landscape and n_D = 1 models | `side_studies/n2_landscape/run_landscape.sh`, `n1_run.py`, `landscape_analysis.py` | `n2_landscape/results/`, `merged/landscape_candidates.csv` |
| 3. repeated Random paths | `side_studies/random_repeats/plan.py`, `run_random_repeats.sh`, `random_repeats_analysis.py` | `random_repeats/results/` |
| 4. offline acquisition scores | `side_studies/criteria_offline/run_criteria.sh` | `criteria_offline/results/` |
| 5. figures and tables | `figures/make_results_figures.py`, `make_dataset_figure.py`, `experiment_graph/*.py` | `thesis_figures/final/` |

Run records and checkpoints are in `runs/03_AL_chicken/`. Training needs `data/chicken_data/dataset.nc`; the figures are regenerated from the stored records (Level 1 of `docs/REPRODUCTION.md`).
