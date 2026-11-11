# DED: seed and provenance metadata

Keys, seeds and training-set orders of the final comparison, extracted from the stored plans and records by `build_submission_metadata.py` (read-only; recorded keys and seeds are copied, recipe values only checked). Rebuild: `python experiments/04_ded/metadata/build_submission_metadata.py`.

The final comparison: ten starts, n_D = 1..5, LA, DE and MoLA (MGE) per start, six Random paths per start, two architectures, evaluation on all 25 simulations (training trajectories included, no held-out set).

## Cohort

Ten starts in key-index order i = 0..9: `[12, 4, 24, 22, 10, 20, 2, 14, 18, 6]` (the thesis table shows them sorted; do not derive the key index from the sorted order).

| package | run | content |
|---|---|---|
| A | `start_study/run/main` | starts 12, 4, 24, 22, 10 (i = 0..4): LA / DE / MoLA, Random 0..2 |
| T | `final_extension/run` (paths/T) | the same five starts: Random 3..5 |
| G6 | `final_extension/run` (paths/G6) | starts 20, 2, 14, 18, 6 (i = 5..9): LA / DE / MoLA and Random 0..5 |

Not in the final figures and tables, and not in the manifest: start 0, the extra first-step Random draws 6..9, the candidate-only pilot studies.

## Files

| file | content |
|---|---|
| `seed_manifest_paths.csv` | 900 path records (start / architecture / stream / n_D): start ID, method, training-set IDs in acquisition and in training order, picked ID, shared node, recorded training key and initialisation seeds, paths and hashes of checkpoint, POD basis and training record, host, job |
| `seed_manifest_acquisition_models.csv` | 80 separately trained single-member LA acquisition models: training set, key, seed, POD hash |
| `seed_manifest.json` | constants (cohort, packages, key recipes) and the summary of all cross-checks |
| `random_tails.json` | the 120 Random tails (60 distinct (start, r) pairs × 2 architectures) with their recomputation |
| `plans/` | verbatim copies of the plans and launch manifests, hashes in `seed_manifest.json` |
| `environment_recorded.json` | recorded dataset and implementation hashes, Python and package versions, VCS revisions (`git_commit` recorded empty) |
| `inspected_settings.json` | parameter counts (NODE 2248, sPHNN 6425), Tsit5 with PID control, atol = rtol = 1e-6, the two training stages |

## Keys and seeds

- Reporting key `9000000 + 100000 i + 1000 (n_D - 1)`; LA acquisition key `9500000 + 100000 i + 1000 (n_D - 1)`. 0 mismatches in 900 reporting records and 80 acquisition models. The numeric reporting keys of i ≥ 5 coincide with the acquisition-key range of i = 0; the two roles are separate training requests (`key_collision_example` in `seed_manifest.json`).
- Member seeds are not `key + k`: they follow the recorded JAX key splitting (`la_de_runner/run_al.py::member_seeds`, float64). The recorded seeds were re-derived with the same function: all 900 + 80 agree.
- Random tail: `np.random.default_rng([20264001, s, r]).permutation([j for j in range(25) if j != s])`, r = 0..5, identical for both architectures; all plan tails equal their recomputation. Training order is the start ID followed by the other selected IDs in increasing order, not the acquisition order.
- Reporting keys are paired across AL and Random at fixed start and n_D. LA's acquisition model has its own key and is not a member of the three-member reporting ensemble.

## Shared fits

Many records read one trained node (`records_sharing_this_node` > 1, e.g. the n_D = 1 node of a start): 900 path records correspond to 621 distinct reporting nodes. Count these nodes once; they are not independent runs.

The checkpoints (`model.eqx`), POD bases and per-iteration records are under `runs/{start_study,final_extension}/run/`.
