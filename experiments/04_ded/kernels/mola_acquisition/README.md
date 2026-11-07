# MoLA acquisition for DED

K = 3 Laplace components (the reporting members 0, 1, 2 that pass the training filter), each in a fresh CPU process, combined to a uniform mixture; the acquisition score is the MGE of the total variance.

* `benchmark.py`: starts the component processes (`run_group`) and loads their results (`load_outputs`).
* `worker.py`: one component: exact GGN and marginal-likelihood calibration of one trained member, mean and marginal variance of every pool candidate.
* `moments.py`: mean, within- and between-component variance of the mixture and the acquisition score.
* `common.py`: file hashing, saved POD loading, atomic writes, environment of the processes.
* `seed_selection.json`: the choice of the members 0, 1, 2 (see `../README.md`).

Called from `../../al_runs/start_study/study_numerical.py` (`acquire`).
