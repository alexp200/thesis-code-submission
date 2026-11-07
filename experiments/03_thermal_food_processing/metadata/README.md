# Thermal food processing: seed and provenance metadata

Built from the stored plans and records (`build_submission_metadata.py`, read-only; no training, no new random draws). It preserves what was removed from the thesis appendix
(seed-key equation, cohort-to-key indexing, start groups, per-run discrepancy bookkeeping). The thesis refers to this folder for exact seed bookkeeping.

## The three data sources of the thermal-food-processing appendix

1. **AL / Random paths** (learning curves, path distributions, costs): ten original starts (n_D = 1..15, one AL path per strategy, one stored Random path) and five additional starts (n_D = 1..5), six AL strategies per architecture
   (MoLA, DE, LA x MGE, VAR), seed set A; the repeated Random baseline (ten paths per start, n_D = 2..5) of all 15 starts. -> `seed_manifest_al.csv`, `seed_manifest_random_repeats.csv`, `random_permutations/`.
2. **Exhaustive first-acquisition errors**: models trained on every one of the 14 two-trajectory sets {start, candidate} per start, seed sets A and B. -> `seed_manifest_landscape.csv`.
3. **Acquisition / seed sensitivity**: acquisition scores from the n_D = 1 initial models of seed sets A, B, C; candidate errors from source 2 (A, B). C is not a full AL-path repetition. -> `seed_manifest_n1.csv`.

## Files

| file | content |
|---|---|
| `seed_manifest.json` | constants (selection pool, AP15 test set, starts in KEY-INDEX order, key recipe, RNG recipes, tie tolerance) and the summary of all cross-checks |
| `seed_manifest_al.csv` | one row per recorded training of an AL / stored-Random path (2400): cohort, key index, start ID, architecture, strategy, n_D, **training-set IDs in training order**, recorded training key, initialisation seeds, optimiser key, recipe key and match flag, checkpoint path and recorded hash, `fit_id` / `records_sharing_this_fit` (rows with the same recorded checkpoint hash read ONE fit), host, job ID, source record |
| `seed_manifest_landscape.csv` | 840 exhaustive two-trajectory fits (420 per seed set A, B): start, candidate, architecture, recorded keys, recipe check, checkpoint / record paths |
| `seed_manifest_n1.csv` | 90 n_D = 1 initial models (set A of the original starts referenced from the AL iter00 records; sets A of the additional starts, B, C from `n1_models/`) |
| `seed_manifest_random_repeats.csv` | 880 planned / recorded new trainings of the repeated Random baseline (n_D = 3..5; `paths_r` lists the paths that read a fit) and 300 rows for n_D = 2 that **reuse** the exhaustive set-A fit (no training) |
| `plans/` | verbatim copies of the existing plans: warm-start plan (original starts; contains the stored Random order of every path), additional-start plan and launch manifest, Random-repeat plan and task list, with hashes in `seed_manifest.json` |
| `random_permutations/` | `stored_random_extension_original_starts.json` (plan order and recomputation per key index), `repeated_random_tails.json` (all 150 tails, plan and recomputation) |
| `pick_audit.md`, `pick_audit.csv` | actual AL picks versus separately evaluated acquisition scores (177 of 180 first selections agree; the three differences are listed) |
| `environment_recorded.json` | recorded software versions / revisions (only where recorded, see below) and the recorded implementation hashes |

## Seed bookkeeping (as recorded, checked)

* Selection pool (sorted) `[745, 795, 835, 851, 853, 861, 868, 873, 874, 888, 899, 902, 905, 906, 917]`; AP15 test set `[313, 320, 344, 378, 383, 407, 412, 415, 461, 462, 466, 467, 474, 508, 528]`.
* Original ten starts in key-index order p = 0..9: `[853, 917, 874, 906, 888, 835, 868, 899, 902, 851]` (drawn with `np.random.default_rng(20261012).choice(sorted(pool), size=10, replace=False)`; **not** the sorted display order);
  additional starts in key-index order i = 0..4: `[745, 795, 861, 873, 905]`.
* `b = 7000000 + 100000 p + 1000 (n_D-1) + delta_X` (original), `8000000 + 100000 i + ...` (additional); delta_A = 0, delta_B = 50000, delta_C = 70000; member keys `b + k`, optimiser key `b + 1`.
  **Cross-check against the saved records: 0 mismatches** in all 2400 AL records, 840 landscape fits, 90 n_D = 1 models and 880 Random-repeat trainings (recorded keys are kept in the tables; the recipe key is a separate column).
* Stored Random extension (original starts only): `np.random.default_rng(20262012 + p).permutation(sorted(remaining))`; repeated Random: `np.random.default_rng([20263012, s, r]).permutation(sorted(remaining))`, r = 0..9, prefixes define the
  training sets through n_D = 5, draws shared across architectures; at n_D = 2 the corresponding set-A exhaustive result is reused. **All stored orders equal their recomputation** (`seed_manifest.json`).
* The five additional starts have no stored Random path (their Random baseline is the repeated one).

## Reused fits versus repeated path occurrences

In `seed_manifest_al.csv` several path occurrences of one start (strategies / paths with identical training set and key) carry the **same recorded checkpoint hash**: they read one fit (`records_sharing_this_fit` > 1). They must not be counted as independent runs.
Reused in the same sense: the exhaustive set-A fit is read by the Random repeats at n_D = 2 and by every AL path that picked that candidate (`seed_manifest_random_repeats.csv`, kind "reused fit").

## Actual picks versus separately evaluated scores

`pick_audit.md`: 177 of 180 first selections agree; the differences are NODE / DE at start 873 (MGE and VAR) and at start 905 (MGE). The first-acquisition figures use the actual AL-path picks; error rankings and seed sensitivity use their diagnostic data.
The diagnostic models were evaluated locally while those AL paths ran on the cluster; equal keys do not imply bitwise identical fits across machines. Nothing was recomputed or replaced.

## Unknown / not recorded

* Package versions and revision of the ORIGINAL ten-start run: not recorded (only implementation hashes and the dataset hash); the versions of the five-start run are recorded in its launch manifest (`environment_recorded.json`). Current versions are not substituted. The recorded git commit of the five-start launch is empty.
* Host-level provenance is available only where the records contain it (`host`, `slurm_job_id` columns).

## What the code delivery contains / refers to

Contained in this folder: the files above (CSV, JSON, plan copies, permutation files, audit, generator script). **Referenced, not copied** (paths relative to the repository root, columns `checkpoint` / `record`): the trained checkpoints (`*.eqx`, several thousand files) and the per-iteration result JSON
files under `runs/03_AL_chicken/` and `experiments/03_thermal_food_processing/side_studies/{n2_landscape,random_repeats}/`. They should be added to the archive with the same relative paths (or are listed with their recorded hashes where the records contain them).
Rebuild: `python experiments/03_thermal_food_processing/metadata/build_submission_metadata.py`.
