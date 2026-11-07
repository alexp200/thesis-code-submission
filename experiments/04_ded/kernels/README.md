# DED kernels

The computational code that the workflows in `../al_runs/` import. The call chain of a task is `study_numerical.py` -> `experiments/shared/al_path_repeats/repeat_adapter.py` (training and reporting of one architecture) -> `la_de_runner/run_al.py` (environment check, member seeds, handles) -> `core/`.

* `core/`: `ded_data.py` (data set, POD fit, encoding / decoding, derivative targets), `ded_split.py` (warm-start, pool and evaluation ids), `_ded_al_core.py` (model builders, POD-latent data preparation, two-stage training, training-RMSE filter, GGN / Laplace fit, MGE scoring of the pool), `al_loop_ded.py` (training schedule per architecture and `train_ensemble`).
* `la_de_runner/`: `run_al.py` (see above), `reporting.py` (Kelvin errors of all members on the 25 simulations; DE score).
* `mola_acquisition/`: the MoLA acquisition. `benchmark.py` runs the Laplace components in fresh processes (`worker.py`), `moments.py` combines them (uniform mixture, within + between variance) and scores the pool, `common.py` holds helpers. `seed_selection.json` records why the reporting members 0, 1, 2 are the MoLA components (chosen by the training filter only, never by test errors).

Training recipe: derivative stage (20k steps, 1e-3) followed by trajectory fine-tuning (NODE 15k steps at 1e-4, L2 1e-5; sPHNN 10k steps at 1e-5, L2 1e-6).
