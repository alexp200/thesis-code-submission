"""Memory management of the CPU runs: free the compiled-code caches of JAX when a computational phase is finished."""

import gc
import json
import resource
import time
from functools import wraps
from pathlib import Path


def resource_snapshot():
    """Address-space limit, memory use and number of memory maps of the process (Linux)."""
    result = {"address_space_limit": list(resource.getrlimit(resource.RLIMIT_AS))}
    status = Path("/proc/self/status")
    if status.exists():
        fields = dict(line.split(":", 1) for line in status.read_text().splitlines() if ":" in line)
        result.update({name: fields.get(name, "").strip()
                       for name in ("VmRSS", "VmPeak", "VmSize", "Threads")})
    maps = Path("/proc/self/maps")
    if maps.exists():
        with maps.open() as handle:
            result["memory_map_count"] = sum(1 for _ in handle)
    limit = Path("/proc/sys/vm/max_map_count")
    if limit.exists():
        result["max_map_count"] = int(limit.read_text())
    return result


def release_phase_cache(phase, result):
    """Wait for the result, clear the caches and print what was freed; returns the result."""
    import equinox as eqx
    import jax

    # Finish asynchronous work before discarding executables, not model arrays.
    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(result)):
        raise RuntimeError("Cache release must be outside JIT/autodiff transforms")
    jax.block_until_ready(result)
    before = resource_snapshot()
    start = time.perf_counter()
    eqx.clear_caches()
    jax.clear_caches()
    gc.collect()
    elapsed = time.perf_counter() - start
    print("CACHE_RELEASE " + json.dumps({
        "phase": phase, "seconds": elapsed,
        "before": before, "after": resource_snapshot(),
    }), flush=True)
    return result


def release_after_phase(function):
    """Keep the function's result; free compilation caches after it returns."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        result = function(*args, **kwargs)
        return release_phase_cache(f"{function.__module__}.{function.__name__}", result)
    return wrapped
