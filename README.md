# Code and results of the Master's thesis

**Modelling of dynamic systems using stochastic physics-aware machine learning methods**
Alexander Pahl, Technische Universität Darmstadt, Fachgebiet Cyber-Physische Simulation, submitted: 18.11.2026

This repository contains the experiment code and the stored results of the experiments (run records and checkpoints), so that the figures and tables of the thesis can be regenerated and the reported model errors can be recomputed. As retraining all models from scratch is expensive, the repository offers several levels of result reproduction.

## Reproduction levels

| Level | What | Needs |
|---|---|---|
| 1 | Regenerate the figures and tables of the thesis from the stored results (minutes) | this repository; the dataset figures 4.7 and 4.12 also need the raw data |
| 2 | Recompute model errors from the stored checkpoints | raw data in `data/`; for DED also the POD bases from `large_assets/` |
| 3 | Retrain the models (the full studies took about 2000 CPU hours each; single paths of the thermal food processing and DED studies run on one machine) | raw data; CPU cluster for the full studies* |

\* See [Changes to the original code](#changes-to-the-original-code).

The scripts that generate the thesis figures and tables (Level 1), the recomputation tools (Level 2) and the entry points for retraining (Level 3) are listed in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).
The cubic-regression experiment (Section 4.1.1) and the Ott-style check (Appendix B.2.1) are computed from scratch by their generators and need no external data.

## Quick start

Needs [`uv`](https://docs.astral.sh/uv/) (it fetches Python 3.12 itself) and internet access: `klax` and `dynax` are installed from GitHub at pinned commits.

```bash
uv sync --extra dev               # Python 3.12, CPU only; exact versions in uv.lock
source .venv/bin/activate
export PYTHONPATH=$PWD

python -m pytest                  # unit tests of the library in src/

# Level 1, example: thermal food processing figures and tables (see docs/REPRODUCTION.md for all generators)
cd experiments/03_thermal_food_processing/figures && python make_results_figures.py
```

Generated figures and tables are written to `thesis_figures/` in the folder of the respective experiment.

## Layout

[`docs/STRUCTURE.md`](docs/STRUCTURE.md) is a map of the experiment folders (by thesis chapter and by role) with the vocabulary of the AL workflows; [`docs/AL_WORKFLOW.md`](docs/AL_WORKFLOW.md) follows an AL run step by step, in terms of Figure 3.1 of the thesis.

| path | content |
|---|---|
| `src/` | library modules (Laplace approximation, MoLA, models, ensembles, acquisition criteria, SWAG) |
| `experiments/` | experiment code per thesis chapter: `01_cubic_regression`, `02_harmonic_oscillator`, `03_thermal_food_processing`, `04_ded`, `cost_analysis`, and `shared/` (plot style, CPU launcher, adapter). Seed manifests are in `*/metadata/` |
| `runs/` | stored run records and checkpoints of the final cohorts (their folder names are those of the original runs: `03_AL_chicken`, `ded_start_study`, `ded_final_extension`) |
| `tools/` | Level-2 recomputation tools |
| `tests/` | unit tests of `src/` |
| `docs/` | environment, data access, reproduction details, walk-through of the AL workflows |
| `data/` | raw simulation data, not part of the repository, see [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) |
| `large_assets/` | optional archives (about 0.9 GB), delivered separately and not in git: POD bases of the DED nodes and cached results of the thermal runs, see [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md#large-assets) |

The identifier "chicken" (in `runs/03_AL_chicken`, `data/chicken_data` and in the code) refers to the thermal food processing experiment.

**Old folder names.** The folders of `experiments/` (formerly `final_experiments/`) were renamed and grouped for the submission. The stored records (`runs/`, the plans and manifests in `*/metadata/`, `phase_integration.json`, `restart_snapshot.json`) are records of the thesis runs and were left unchanged, so they still contain the paths of the original working tree:

| in the records | in this repository |
|---|---|
| `final_experiments/02_harmonic_oscilator/` | `experiments/02_harmonic_oscillator/` |
| `final_experiments/03_AL_chicken/` | `experiments/03_thermal_food_processing/` (`warmstart_path_AL/` is `al_runs/`, `evaluation/` is `figures/`) |
| `final_experiments/04_AL_DED/` | `experiments/04_ded/` |
| `final_experiments/ded_start_study/`, `ded_final_extension/` | `experiments/04_ded/al_runs/start_study/`, `final_extension/` |
| `final_experiments/{cluster_recovery,al_path_repeats,thesis_plot_style}/` | `experiments/shared/` |

## Changes to the original code

The library `src/` and the experiment code were reduced to what the thesis reports (earlier AL loops, investigations that the thesis does not use, unused library functions). The retained paths compute the same numbers as the original code.

The AL experiments were run on a cluster, and the AL code was designed for that: it handles timeouts and resumes of jobs, which explains the extensive orchestration and fallbacks. This repository contains that code, not a separate version of it. To make the results reproducible, the same code can be run locally: `run_local.sh` runs the phases of a path one after the other on one machine (for DED switched on by `DED_LOCAL`). The Slurm launchers of the thesis runs (`launch.sh`, `run_cpu.sbatch`, `submit.sh`) are kept next to it for reference. The cluster variant of the five additional starts of the thermal experiment (shared n_D = 1 ensemble, release checks against earlier runs) is described in `experiments/03_thermal_food_processing/al_runs/README.md` and is not part of the repository.

Further changes: paths (folders were renamed; run records are read from `runs/`) and a stored table of mean oven temperatures (`experiments/03_thermal_food_processing/metadata/oven_means.csv`) that replaces a read of the raw data. For DED, the memory release between phases and the cache of POD fits belong to the production code (`experiments/shared/cluster_recovery/`).
