# Environment

Tested with Python 3.12.13 on macOS 26.6.2 (arm64), CPU only, float64, using `uv` 0.11.14 and the versions pinned in `uv.lock`:
JAX/jaxlib 0.10.1, equinox 0.13.8, diffrax 0.7.2, optax 0.2.8, numpy 2.4.6, scipy 1.17.1, pandas 3.0.5, xarray 2026.7.0, matplotlib 3.10.9, together with `klax` (commit `5e8856e8…`) and `dynax` (commit `4cd21efe…`) installed from GitHub.
The tests, the regeneration and the recomputation tools were run from a fresh environment created with `uv sync --extra dev`.

The stored runs were produced on Linux with Python 3.12.13 and the same package versions (recorded in `experiments/03_thermal_food_processing/metadata/environment_recorded.json` and `experiments/04_ded/metadata/environment_recorded.json`). The original ten-start thermal run did not record its package versions.
All cluster runs used one native thread per process and serial XLA code generation (`experiments/shared/cluster_recovery/run_cpu.sh`); the recomputation tools set the same configuration.
Recomputing a checkpoint on a different CPU gives small differences (relative 1e-5 to 4e-4 in the checked nodes) because the ODE solver uses adaptive step sizes.
