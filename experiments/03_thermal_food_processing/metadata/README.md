# Thermal food processing: seed and provenance metadata

Seed keys, training-set orders and provenance of the stored trainings, extracted from the stored plans and records by `build_submission_metadata.py` (read-only; nothing is retrained or redrawn). Rebuild: `python experiments/03_thermal_food_processing/metadata/build_submission_metadata.py`.

## Files

| file | content |
|---|---|
| `seed_manifest.json` | constants (selection pool, AP15 test set, starts in key-index order, key and RNG recipes, tie tolerance) and the summary of all cross-checks |
| `seed_manifest_al.csv` | AL and stored-Random paths, one row per recorded training (2400): start, architecture, strategy, n_D, training-set IDs in training order, recorded keys and seeds, checkpoint path and hash, host, job ID |
| `seed_manifest_landscape.csv` | 840 exhaustive two-trajectory fits (every {start, candidate} pair, seed sets A and B) |
| `seed_manifest_n1.csv` | 90 n_D = 1 initial models (seed sets A, B, C) |
| `seed_manifest_random_repeats.csv` | repeated Random baseline (ten paths per start, n_D = 2..5): 880 planned or recorded trainings and 300 rows at n_D = 2 that reuse the exhaustive set-A fit |
| `plans/` | verbatim copies of the plans of the stored runs, hashes in `seed_manifest.json` |
| `random_permutations/` | stored and recomputed Random orders |
| `pick_audit.md`, `.csv` | actual AL picks against separately evaluated acquisition scores |
| `environment_recorded.json` | recorded software versions and implementation hashes |

## Keys and seeds

- Original ten starts in key-index order p = 0..9: `[853, 917, 874, 906, 888, 835, 868, 899, 902, 851]`, drawn with `np.random.default_rng(20261012).choice(sorted(pool), size=10, replace=False)` (not the sorted order). Additional starts, i = 0..4: `[745, 795, 861, 873, 905]`.
- Training key `b = 7000000 + 100000 p + 1000 (n_D-1) + delta_X` (additional starts: `8000000 + 100000 i + ...`); delta_A = 0, delta_B = 50000, delta_C = 70000; member keys `b + k`, optimiser key `b + 1`.
- Stored Random extension (original starts): `np.random.default_rng(20262012 + p).permutation(sorted(remaining))`. Repeated Random: `np.random.default_rng([20263012, s, r]).permutation(sorted(remaining))`, r = 0..9; prefixes define the training sets, draws are shared across architectures. The additional starts have no stored Random path.
- Cross-check against the records: 0 key mismatches in all 2400 AL records, 840 landscape fits, 90 n_D = 1 models and 880 Random-repeat trainings; all stored Random orders equal their recomputation.

## Shared fits

Several path records of one start (identical training set and key) carry the same checkpoint hash and read one fit (`records_sharing_this_fit` > 1). They are not independent runs. The exhaustive set-A fit is likewise read by the Random repeats at n_D = 2 and by every AL path that picked that candidate.

## Picks against scores

`pick_audit.md`: 177 of 180 first selections agree with the separately evaluated scores; the differences are NODE / DE at start 873 (MGE and VAR) and start 905 (MGE). The diagnostic models were evaluated locally while the AL paths ran on the cluster, and equal keys do not imply bitwise identical fits across machines.

## Not recorded

Package versions and revision of the original ten-start run (only implementation hashes and the dataset hash); those of the five-start run are in `environment_recorded.json`, its recorded git commit is empty. Host and job IDs only where the records contain them.

The trained checkpoints (`*.eqx`) and per-iteration records that the CSV columns `checkpoint` and `record` point to are under `runs/03_AL_chicken/` and `experiments/03_thermal_food_processing/side_studies/{n2_landscape,random_repeats}/`.
