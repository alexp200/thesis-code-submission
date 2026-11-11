# DED start study (five starts)

Starts 12, 4, 24, 22, 10; NODE and sPHNN; n_D = 1..5; LA, DE and MoLA with the MGE criterion plus three Random paths per start. `study_design.py` defines starts, keys and Random tails (keys and recipes: [`experiments/04_ded/metadata/README.md`](../../metadata/README.md)). Every ensemble is trained from scratch with a POD basis fitted on its training set; identical training sets are trained once and shared. Reporting evaluates all three members on all 25 simulations. Steps and code per box of Figure 3.1: [`docs/AL_WORKFLOW.md`](../../../../docs/AL_WORKFLOW.md).

## Running a path on one machine

From the repository root, after `uv sync`; requires `data/directed_energy_deposition/max_simulation/dataset.nc` ([`docs/DATA_ACCESS.md`](../../../../docs/DATA_ACCESS.md)):

```bash
bash experiments/04_ded/al_runs/start_study/run_local.sh NODE/start12/LA 2     # path id = model/start/stream, here up to n_D = 2
python tools/compare_local_ded.py NODE/start12/LA                               # member RMSEs and added trajectories next to the stored records
```

Path ids: model `NODE` or `sPHNN`, start `start12`, `start04`, `start24`, `start22`, `start10` and, for the extension, `start20`, `start02`, `start14`, `start18`, `start06`; streams `LA`, `DE`, `MoLA`, `random0` .. `random5`. Several paths are separated by commas.

`run_local.sh` sets `DED_LOCAL=1`: the plan is restricted to the selected paths and written with the results to `run/main_local_<hash of the selection>/`. The Slurm submission, the release gate (a comparison with earlier runs that are not part of this repository) and the CPU checks of the cluster are switched off. The local plan also contains the five further starts and the Random paths 3-5 of `../final_extension` (same keys and request ids), so every DED path of the thesis can be selected. Every call continues where the previous one stopped. If the stored POD bases are unpacked (`tar -xzf large_assets/ded_pod_bases.tar.gz` from the repository root), a local run uses them, because the sign convention of an SVD depends on the platform.

`run_local.sh PATHS THROUGH_N` also trains the acquisition model of the last size, because the coordinator prepares the next size. NODE / LA up to n_D = 2 takes about 35 minutes on an Apple-silicon laptop with 16 GB (cluster: 13 and 33 minutes for the reporting ensembles of n_D = 1 and 2); sPHNN is slower (n_D = 1: 21 minutes for the reporting ensemble, 6 for the acquisition model, 7 for the acquisition). The exact GGN needs memory in proportion to the number of training trajectories (not tested beyond n_D = 2 on 16 GB). The complete study took about 2000 hours of training on the cluster. As for the thermal studies, a local run is not bitwise identical to the stored records, which were produced on x86-64 Linux CPUs.

## Files

`study_design.py` (design), `study_numerical.py` (training and acquisition steps), `study_store.py` (atomic result store), `control.py` (prepare, progress, recovery), `worker.py` (one task), `submit.sh`, `*_cpu.sbatch` (Slurm). Stored results: `runs/ded_start_study/`.
