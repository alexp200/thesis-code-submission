# Code and results of the Master's thesis

**Modelling of dynamic systems using stochastic physics-aware machine learning methods**
Alexander Pahl, Technische Universität Darmstadt, Fachgebiet Cyber-Physische Simulation, submitted: 18.11.2026

This repository contains the experiment code and the stored results (run records and checkpoints) of the thesis. The figures and tables can be regenerated and the reported model errors recomputed. Retraining all models is expensive, so reproduction is offered in three levels.

## Reproduction levels

| Level | What                                                                                                                                                          | Needs                                                                    |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 1     | Regenerate the figures and tables of the thesis from the stored results (minutes)                                                                             | this repository; the dataset figures 4.7 and 4.12 also need the raw data |
| 2     | Recompute model errors from the stored checkpoints                                                                                                            | raw data in `data/`; for DED also the POD bases from `large_assets/`     |
| 3     | Retrain the models (the full studies took about many thousand CPU hours each; single paths of the thermal food processing and DED studies run on one machine) | raw data; a CPU cluster for the full studies                             |

The generators (Level 1), recomputation tools (Level 2) and entry points for retraining (Level 3) are listed in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md). The cubic-regression experiment (Section 4.1.1) and the Ott-style check (Appendix B.2.1) are computed from scratch by their generators and need no external data.

## Quick start

Needs [`uv`](https://docs.astral.sh/uv/) (it fetches Python 3.12) and internet access: `klax` and `dynax` are installed from GitHub at pinned commits.

```bash
uv sync --extra dev               # Python 3.12, CPU only; exact versions in uv.lock
source .venv/bin/activate
export PYTHONPATH=$PWD

python -m pytest                  # unit tests of src/ and a smoke run of the examples

# Level 1, example: thermal food processing figures and tables
cd experiments/03_thermal_food_processing/figures && python make_results_figures.py
```

Generated figures and tables are written to `thesis_figures/` in the folder of the respective experiment.

## Layout

| path | content |
|---|---|
| `src/` | library: Laplace approximation, MoLA, models, ensembles, acquisition criteria, SWAG |
| `experiments/` | code per thesis chapter (`01_cubic_regression`, `02_harmonic_oscillator`, `03_thermal_food_processing`, `04_ded`), `cost_analysis`, and `shared/` (plot style, CPU launcher, adapter); seed manifests in `*/metadata/` |
| `runs/` | stored run records and checkpoints of the final cohorts |
| `tools/` | Level-2 recomputation tools |
| `examples/` | two short programs that show the library: a UQ workflow and an active-learning loop on the harmonic oscillator, see [`examples/README.md`](examples/README.md) |
| `tests/` | unit tests of `src/`, smoke run of the examples |
| `docs/` | [environment](docs/ENVIRONMENT.md), [data access](docs/DATA_ACCESS.md), [reproduction](docs/REPRODUCTION.md), [folder map](docs/STRUCTURE.md), [AL workflow](docs/AL_WORKFLOW.md), [design of `src/`](docs/DESIGN.md) |
| `data/` | raw simulation data, not in the repository, see [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) |
| `large_assets/` | optional archives (about 0.9 GB), delivered separately: POD bases of the DED nodes and cached results of the thermal runs |

`runs/`, `*/metadata/plans/`, `phase_integration.json` and `restart_snapshot.json` are unchanged records of the thesis runs and contain the folder names of the original working tree (`final_experiments/...`); [`docs/STRUCTURE.md`](docs/STRUCTURE.md) maps them to the current ones.

## Changes to the original code

`src/` and the experiment code contain what the thesis reports; the retained paths compute the same numbers as the original code. The AL experiments were run on a cluster, and the AL code handles timeouts and resumes of jobs, which explains its orchestration; `examples/02_active_learning.py` shows the same loop without it. It is the same code that runs locally: `run_local.sh` runs the phases of a path one after the other on one machine (for DED switched on by `DED_LOCAL`). The Slurm launchers of the thesis runs are kept for reference. The cluster variant of the five additional thermal starts is not part of the repository. Other changes: paths, and a stored table of mean oven temperatures (`experiments/03_thermal_food_processing/metadata/oven_means.csv`) that replaces a read of the raw data.
