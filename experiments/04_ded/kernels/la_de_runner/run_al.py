"""Entry module of the DED kernels: environment check, derivation of the member seeds and the handles on the training, data and reporting code that the study workflows use
(`experiments/shared/al_path_repeats/repeat_adapter.py` loads this module as `base`).
"""

import os
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.random as jrd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIMARY = HERE.parent / "core"
for path in (ROOT, PRIMARY, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import al_loop_ded as original
import _ded_al_core as core
from ded_data import decode_state, load_dataset
from ded_split import EVALUATION_IDS
from reporting import evaluate_members, score_pool_de


def check_environment():
    """Require the validated setting: CPU, float64, one native thread, serial XLA code generation, and the prepared repository."""
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise ValueError("Use the validated x64 CPU environment")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise ValueError(f"Set {variable}=1 before starting Python")
    if "--xla_cpu_parallel_codegen_split_count=1" not in os.environ.get("XLA_FLAGS", "").split():
        raise ValueError("Use cluster_recovery/run_cpu.sh (serial CPU codegen)")
    if not (ROOT / "phase_integration.json").is_file() or not hasattr(core.prepare_iteration_data, "__wrapped__"):
        raise ValueError("Run in the isolated CPU-validated repository, not the original repository")


def member_seeds(key_value, size):
    """Initialisation seeds of the members of an ensemble, derived from the training key by the same key splitting as train_hybrid_stages."""
    stage1_key, _ = jrd.split(jrd.key(key_value))
    ensemble_key, _ = jrd.split(stage1_key)
    return [int(seed) for seed in jrd.randint(ensemble_key, (size,), 0, 2**31 - 1)]
