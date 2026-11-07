"""Small JAX-free helpers of the MoLA acquisition: file hashing, atomic writes, loading a saved POD and the environment of the component processes."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIMARY = Path("experiments/04_ded/kernels")


def digest(path):
    """SHA-256 of a file."""
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    """Write a JSON file so that it appears completely or not at all."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_pod(path):
    """Saved POD (modes_T, modes_q, x_eq); checks that it is finite and orthogonal."""
    with np.load(path, allow_pickle=False) as stored:
        arrays = tuple(stored[name].copy() for name in ("modes_T", "modes_q", "x_eq"))
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Nonfinite saved POD")
    for basis in arrays[:2]:
        c2 = np.sum(basis[:, 0] ** 2)
        if c2 <= 0:
            raise ValueError("Zero POD scale")
        np.testing.assert_allclose(basis.T @ basis / c2, np.eye(basis.shape[1]),
                                   rtol=1e-8, atol=1e-8)
    return arrays


def native_environment():
    """Environment of a component process: CPU, float64, one native thread, serial XLA code generation."""
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    flags = env.get("XLA_FLAGS", "").split()
    if any("xla_cpu_parallel_codegen_split_count" in flag and
           flag != "--xla_cpu_parallel_codegen_split_count=1" for flag in flags):
        raise ValueError("Conflicting CPU codegen setting")
    if "--xla_cpu_parallel_codegen_split_count=1" not in flags:
        flags.append("--xla_cpu_parallel_codegen_split_count=1")
    env["XLA_FLAGS"] = " ".join(flags)
    return env
