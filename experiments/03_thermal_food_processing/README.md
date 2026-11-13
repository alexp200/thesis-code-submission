# Thermal food processing (Section 4.2.1, Appendix B.3)

Active learning with NODE and sPHNN surrogates on simulated thermal food processing trajectories (the identifier "chicken" in the code, in `data/chicken_data` and in `runs/03_AL_chicken/` is a historical name). Starts: 15 selection-pool trajectories; test set: 15 fixed trajectories (AP15). The AL loop is described in [`docs/AL_WORKFLOW.md`](../../docs/AL_WORKFLOW.md).

## Layout

| folder | content |
|---|---|
| `al_runs/` | the AL workflow: paths of the ten original starts (n_D = 1..15) and five additional starts (n_D = 1..5), 200 tasks in `study/plan.json` (see its README) |
| `kernels/core/` | training recipe (`train_common.py`) and normalisation (`chicken_data.py`) imported by the workflow and the side studies |
| `side_studies/n2_landscape/` | exhaustive first acquisition: a model for every second trajectory of every start (`landscape_run.py`), its table (`landscape_analysis.py`), the n_D = 1 models of the additional starts and of seed sets B, C (`n1_run.py`); stored results in `merged/`, `n1_models/` |
| `side_studies/random_repeats/` | ten sampled Random paths per start (`plan.py`, `random_repeat_run.py`, `random_repeats_analysis.py`); stored results in `results/`, Slurm copies in `cluster/` |
| `side_studies/criteria_offline/` | acquisition scores (MGE, VAR) of the n_D = 1 models for seed sets A, B, C (`criteria_offline.py`, `acq_criteria.py`); stored in `results/` |
| `figures/` | generators: `make_results_figures.py` (Figs. 4.8-4.11, B.7, Tabs. B.10-B.17), `make_dataset_figure.py` (Fig. 4.7), `experiment_graph/` (Figs. 3.2, B.5, B.6); `al_overview/studies_io.py` loads the AL run records; `authored_tables/` holds the hand-written Tabs. B.8 and B.9, copied by the generator |
| `common/` | output locations of the generators |
| `metadata/` | seed manifests: which training used which keys and data |

## Pipeline

| step | script | output |
|---|---|---|
| 1. AL and Random paths | `al_runs/run_local.sh TASK` or the Slurm launchers | `al_runs/study/results/` (stored records of the thesis: `runs/03_AL_chicken/`) |
| 2. first-acquisition landscape and n_D = 1 models | `side_studies/n2_landscape/run_landscape.sh`, `n1_run.py`, `landscape_analysis.py` | `n2_landscape/results/`, `merged/landscape_candidates.csv` |
| 3. repeated Random paths | `side_studies/random_repeats/plan.py`, `run_random_repeats.sh`, `random_repeats_analysis.py` | `random_repeats/results/` |
| 4. offline acquisition scores | `side_studies/criteria_offline/run_criteria.sh` | `criteria_offline/results/` |
| 5. figures and tables | `figures/make_results_figures.py`, `make_dataset_figure.py`, `experiment_graph/*.py` | `thesis_figures/final/` |

Training needs `data/chicken_data/dataset.nc`; the figures are regenerated from the stored records in `runs/03_AL_chicken/` ([`docs/REPRODUCTION.md`](../../docs/REPRODUCTION.md), Level 1).
