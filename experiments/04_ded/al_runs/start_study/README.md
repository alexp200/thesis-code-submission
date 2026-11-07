# DED start study (five starts)

Starts 12, 4, 24, 22, 10; NODE and sPHNN; n_D = 1..5; LA, DE and MoLA with the MGE criterion plus three Random paths per start (`study_design.py` defines starts, keys and Random tails).

* Reporting key `9_000_000 + 100_000 i + 1000 (N-1)` (i = index of the start); the separate LA acquisition model uses `9_500_000 + ...`. Random tail: `default_rng([20264001, start, r]).permutation(other 24 ids)`.
* Every ensemble is trained from scratch with a POD basis fitted on its training set; identical training sets are trained once and shared.
* Reporting evaluates all three members on all 25 simulations.

**On a single machine (no Slurm)** from the repository root, after `uv sync`:

```bash
bash experiments/04_ded/al_runs/start_study/run_local.sh NODE/start12/LA 2          # path id = model/start/stream, here up to n_D = 2
python tools/compare_local_ded.py NODE/start12/LA                               # member RMSEs and added trajectories next to the stored records
```

Path ids: model `NODE` or `sPHNN`, start `start12`, `start04`, `start24`, `start22`, `start10` (the five starts of this study) and, for the extension, `start20`, `start02`, `start14`, `start18`, `start06`; streams `LA`, `DE`, `MoLA`, `random0` .. `random5`. Several paths are separated by commas. `run_local.sh` sets `DED_LOCAL=1`: the plan is restricted to the selected paths and written together with the results to `run/main_local_<hash of the selection>/`; the Slurm submission, the release gate (a comparison with earlier runs of the thesis that are not part of this repository) and the CPU checks of the cluster are switched off; the local plan also contains the five further starts and the Random paths 3-5 of `../final_extension` (same keys and request ids), so that every DED path of the thesis can be selected. Training, POD, Laplace approximation and scoring are the code of the thesis runs. Every call continues where the previous one stopped. Requires `data/directed_energy_deposition/max_simulation/dataset.nc` (see `docs/DATA_ACCESS.md`). If the POD bases of the stored ensembles are unpacked (`tar -xzf large_assets/ded_pod_bases.tar.gz` from the repository root), a local run uses them, because the sign convention of an SVD depends on the platform.

`run_local.sh PATHS THROUGH_N` also trains the acquisition model of the last size, because the coordinator prepares the next size. NODE / LA up to n_D = 2 takes about 35 minutes on an Apple-silicon laptop with 16 GB (stored cluster times: 13 and 33 minutes for the reporting ensembles of n_D = 1 and 2); sPHNN is slower (n_D = 1: 21 minutes for the reporting ensemble, 6 minutes for the acquisition model, 7 minutes for the acquisition), MoLA acquisition at n_D = 1 took 3 minutes for NODE, and the exact GGN of a larger training set needs memory in proportion to the number of training trajectories (not tested beyond n_D = 2 on 16 GB). The complete study took about 2000 hours of training on the cluster. As for the thermal studies, a local run is not bitwise identical to the stored records, which were produced on x86-64 Linux CPUs.

Files: `study_design.py` (design), `study_numerical.py` (training and acquisition steps), `study_store.py` (atomic result store), `control.py` (prepare, progress, recovery), `worker.py` (one task), `submit.sh`, `*_cpu.sbatch` (Slurm). The first launch step reproduces an earlier start-0 path as a gate; its reference data are not part of this repository. Stored results: `runs/ded_start_study/`.
