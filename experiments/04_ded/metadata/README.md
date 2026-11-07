# DED: provenance metadata of the final comparison

Built from the stored plans and records (`build_submission_metadata.py`, read-only; no training, no new random draws; recorded keys and seeds are copied, recipe values only checked).
The final comparison: **ten starts, n_D = 1..5, three UQ methods (LA, DE, MoLA with MGE) per start, six Random paths per start, two architectures**, evaluation on all 25 simulations (training trajectories included, no held-out set).

## Cohort and packages

Ten starts in their ORIGINAL KEY-INDEX ORDER i = 0..9: `[12, 4, 24, 22, 10, 20, 2, 14, 18, 6]` (the thesis table shows them sorted; do not derive the key index from the sorted order).

| package | run | content |
|---|---|---|
| A | `start_study/run/main` | starts 12, 4, 24, 22, 10 (i = 0..4): LA / DE / MoLA, Random 0..2 |
| T | `final_extension/run` (paths/T) | the same five starts: Random 3..5 |
| G6 | `final_extension/run` (paths/G6) | starts 20, 2, 14, 18, 6 (i = 5..9): LA / DE / MoLA and Random 0..5 |

Not part of the final figures and tables (not in the manifest): start 0 (packages F, C), the H2 extra first-step Random draws 6..9, the candidate-only pilot studies. Packages are under `runs/`.

## Files

| file | content |
|---|---|
| `seed_manifest_paths.csv` | **900 path records** (one per start / architecture / stream / n_D): key index, start ID, architecture, method, stream, n_D, **training-set IDs in acquisition order and in training order**, picked ID, role, `shared_node_id` and `records_sharing_this_node`, recorded training key, recipe key and match flag, **recorded initialisation seeds** and re-derivation flag, effective members, checkpoint / POD basis / training-record paths with hashes, host, job, timestamp, run-manifest hash, source record |
| `seed_manifest_acquisition_models.csv` | 80 separately trained single-member LA acquisition models (role `acquisition`, ensemble size 1): training set, recorded key and seed, recipe check, POD hash and whether it equals the POD of the reporting node of the same training set, selection cache, training record |
| `seed_manifest.json` | constants (cohort, packages, key recipes, collision note) and the summary of all cross-checks |
| `random_tails.json` | the 120 Random tails of the plans (60 distinct (start, r) pairs x 2 architectures) with their recomputation; training sets are prefixes |
| `plans/` | verbatim copies of the existing plans and launch manifests (`start_study_main_*`, `final_extension_*`), hashes in `seed_manifest.json` |
| `environment_recorded.json` | recorded dataset hash, plan hashes, implementation hashes, Python / package versions and VCS revisions (dynax `4cd21efe2675f9a875f6a628493ad7bcc7973f37`, klax `5e8856e8...`), `git_commit` recorded empty (stated as unknown) |
| `inspected_settings.json` | active parameter counts (NODE 2248, sPHNN 6425), Tsit5 with PID control, atol = rtol = 1e-6, max_steps 262144, the two training stages, and hashes of the inspected source files |

## Keys and seeds (recorded, checked)

* Reporting key `9000000 + 100000 i + 1000 (n_D - 1)`; LA acquisition key `9500000 + 100000 i + 1000 (n_D - 1)`. **0 mismatches** in 900 reporting records and 80 acquisition models.
* Member seeds are **not** `key + k`: they follow the recorded JAX key splitting (`la_de_runner/run_al.py::member_seeds`, reached through `repeat_adapter`, float64 / x64 mode). The recorded `initialization_seeds` are what the manifest carries;
  they were re-derived with the same function as a cross-check: **all 900 + 80 agree**.
* The numeric reporting keys of i >= 5 (`9500000 + ...`) coincide with the numeric acquisition-key range of i = 0; the two roles are separate training requests (role, ensemble size, training set, record), see `seed_manifest.json` (`key_collision_example`).
* Random tail: `np.random.default_rng([20264001, s, r]).permutation([j for j in range(25) if j != s])`, r = 0..5; **all plan tails equal their recomputation**, identical for both architectures. Training sets are prefixes; **training order = start ID followed by the other selected IDs in increasing order**, not the acquisition order.
* Reporting keys are paired across AL and Random at fixed start and n_D. LA's one-member acquisition model has its own key and is **not** a member of the three-member reporting ensemble; it uses the POD basis of the reporting node of the same training set.
* The two training stages (derivative fit, trajectory fine-tuning) with separate optimiser keys / state reset are recorded in `training_hyperparameters` of the records and summarised in `inspected_settings.json`.

## Shared fits

Many records read one trained node (`records_sharing_this_node` > 1, e.g. the n_D = 1 node of a start, identical training sets of different streams): 900 path records correspond to 621 distinct reporting nodes (98 read by several records).
Count these nodes once; they are not independent runs.

## Cohort check

The manifest contains exactly the records that the figure / table generator uses (loader set == manifest set, 900 records; AL rows 300, Random rows 600 in `cohort()`).

## What the code delivery contains / refers to

Contained here: the files above and the generator script. **Referenced, not copied:** the trained checkpoints (`model.eqx`), POD bases and per-iteration records under `runs/{start_study,final_extension}/run/` (paths and hashes are in the CSVs).
No final submission archive has been assembled by this step. Rebuild: `python experiments/04_ded/metadata/build_submission_metadata.py`.
