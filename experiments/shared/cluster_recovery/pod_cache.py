"""Cache of the POD coordinates of the DED runs: the fit for a training set is stored once (keyed by training ids, data set and implementation files) and reused by every process that needs the same basis.

The basis is the plain unscaled SVD fit of the thesis runs; concurrent processes all use the first published coordinates.
"""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import jax
import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=None)
def file_hash(path):
    """SHA-256 of a file (cached per path)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_or_fit(ds, train_ids, n_modes, *, fitter, dataset_path, cache_dir, implementation_paths):
    """POD coordinates (modes_T, modes_q, x_eq) for the training ids from the cache, or fitted with `fitter` and published; checks shapes, finiteness and orthogonality."""
    if not jax.config.jax_enable_x64:
        raise ValueError("Restart POD snapshots require JAX x64 before fitting/loading")
    threads = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    if any(value != "1" for value in threads.values()):
        raise ValueError("Restart POD snapshots require the validated one-thread environment")
    metadata = {
        "schema_version": 1, "train_ids": list(train_ids), "n_modes": n_modes,
        "dataset_sha256": file_hash(str(Path(dataset_path).resolve())),
        "implementation_sha256": {Path(path).name: file_hash(str(Path(path).resolve())) for path in implementation_paths},
        "normalization": "original unscaled SVD, global std(U_k*S_k), latent equilibrium shift",
        "jax_version": jax.__version__,
        "threads": threads,
    }
    serialized = json.dumps(metadata, sort_keys=True)
    identity = hashlib.sha256(serialized.encode("ascii")).hexdigest()
    path = Path(cache_dir) / f"pod_{identity}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["metadata"].item()) != serialized:
                raise ValueError(f"POD snapshot metadata mismatch: {path}")
            arrays = tuple(stored[name].copy() for name in ("modes_T", "modes_q", "x_eq"))
    else:
        arrays = tuple(np.asarray(value) for value in fitter(ds, train_ids, n_modes, n_modes))
    n_nodes = ds.sizes["node"]
    if [array.shape for array in arrays] != [(n_nodes, n_modes), (n_nodes, n_modes), (n_modes,)]:
        raise ValueError(f"Unexpected POD snapshot shapes: {path}")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Nonfinite POD; do not train or publish a checkpoint")
    for modes in arrays[:2]:
        c2 = np.sum(modes[:, 0]**2)
        if c2 <= 0 or not np.allclose(modes.T @ modes / c2, np.eye(n_modes), rtol=1e-8, atol=1e-8):
            raise ValueError("Invalid POD basis normalization/orthogonality")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            np.savez(temporary_path, modes_T=arrays[0], modes_q=arrays[1], x_eq=arrays[2], metadata=serialized)
            # Concurrent streams must all use the first published coordinates.
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                pass
        finally:
            temporary_path.unlink(missing_ok=True)
    # A different node may have won publication while this process was fitting.
    with np.load(path, allow_pickle=False) as stored:
        if str(stored["metadata"].item()) != serialized:
            raise ValueError(f"POD snapshot metadata mismatch: {path}")
        arrays = tuple(stored[name].copy() for name in ("modes_T", "modes_q", "x_eq"))
    print(f"POD snapshot: {path}", flush=True)
    return tuple(jnp.asarray(value) for value in arrays)
