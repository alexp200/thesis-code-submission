# Harmonic oscillator (Section 4.1.2, Appendix B.2)

Comparison of NODE and conservative sPHNN ensembles (K = 10) under LA, MoLA, DE, SWAG and MultiSWAG on two noisy half-period data sets (`lower_half`, `left_half`), plus an Ott-style check of the LA implementation.

## Layout

| folder | content |
|---|---|
| `ensembles/` | data, models and settings (`oscillator_lib.py`, `ensemble_lib.py`), training of the ensembles (`train_ensembles.py`), `checkpoints/`, `training/` (training RMSE of every member) |
| `long_horizon/` | LA, MoLA and DE over 15 true periods (`compute_methods.py`), `results/` |
| `short_horizon/` | all five methods on t in [0, 10] and the vector-field uncertainty (`compute_short_rollout.py`), `data/` |
| `swag_dependence/` | SWAG learning-rate study and the SWAG moments of the comparisons (`run_sweep.py`, `aggregate.py`), `cache/`, `comparison/` |
| `figures/` | generators of the thesis figures and tables, `plot_lib.py`, authored tables in `tables/authored/` |
| `ott_check/` | Ott-style check of the LA implementation (`make_ott_figure.py`) |
| `common/` | `export_paths.py`: output locations (`HO_EXPORT_ROOT`, `HO_AUDIT_ROOT` override them) |

The vector-field uncertainty does not depend on the rollout horizon; the maps of all figures (also the background of Figure 4.5) are computed together with the short rollouts in `short_horizon/` because both use the same LA fits and SWAG samples.

## Pipeline

| step | script | output |
|---|---|---|
| 1. train the ensembles | `ensembles/train_ensembles.py --model-class {NODE,sPHNN} --dataset {lower_half,left_half}` | `ensembles/checkpoints/*.eqx`, `ensembles/training/training_*.json` |
| 2. long-horizon moments of LA, MoLA, DE | `long_horizon/compute_methods.py --model-class M --dataset D` | `long_horizon/results/methods_M_D.npz/json` (and per-member caches) |
| 3. SWAG learning-rate study | `swag_dependence/run_sweep.py --model M --dataset D --stage sweep`, `run_sweep.py ... --stage comparison`, `aggregate.py` | `swag_dependence/cache`, `swag_dependence/comparison` |
| 4. short-horizon data of all five methods | `short_horizon/compute_short_rollout.py --model-class M --dataset D` | `short_horizon/data/` |
| 5. figures and tables | `figures/make_figures.py` (Figures 4.3-4.5, B.2, B.3), `figures/make_envelope_and_frequencies.py` (Figures 4.6 and B.4, Table B.10), `figures/make_tables.py` (Tables B.5-B.9) | `thesis_figures/final/` |
| Ott-style check | `ott_check/make_ott_figure.py` (Figure B.1, trains the network, about 1 min) | `thesis_figures/final/` |

Steps 1-4 are stored; step 5 regenerates the figures and tables in seconds. Steps 2 and 4 are recomputed and compared with the stored results by `tools/recompute_ho.py` (docs/REPRODUCTION.md).
