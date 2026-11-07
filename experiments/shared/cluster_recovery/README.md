# CPU configuration of the cluster runs

DED trainings on the cluster produced non-finite POD bases with multi-threaded linear algebra; with one native thread they were finite. All final runs therefore use one thread per process, float64 and serial XLA code generation.

* `run_cpu.sh`: launcher that sets `OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`, `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1` and the XLA flag, and refuses to start without `phase_integration.json`/`restart_snapshot.json` in the repository root.
* `runtime.py`: `release_after_phase` decorator and `release_phase_cache` (free compiled-code memory after a computational phase); the decorators are applied in the production file of the DED runs (`04_ded/kernels/core/_ded_al_core.py`).
* `pod_cache.py`: cache of POD fits keyed by the training set and the implementation files.
* `phase_integration.json`, `restart_snapshot.json` (repository root): launch records of the cluster tree. `phase_integration.json` lists the functions that carry the `release_after_phase` decorator and the Slurm launchers of that tree (not all of the listed files are part of this repository); `restart_snapshot.json` holds the SHA-256 values of the implementation files of the tree at the time of the restart. The launchers and controllers check that both files exist, and the run manifests store the hash of `restart_snapshot.json`; the files must therefore stay unchanged and in the repository root.
