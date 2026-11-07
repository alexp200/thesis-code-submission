"""Executor of the MoLA acquisition: runs the Laplace components (one per retained member) in fresh CPU processes (worker.py) and loads their results."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from common import HERE, atomic_json, digest, native_environment


def cpu_slots(count, per_worker):
    """CPU ids for each worker process (empty lists on one machine: no pinning)."""
    if os.environ.get("DED_LOCAL") == "1":
        return [[] for _ in range(count)]        # one machine: no CPU pinning
    if not hasattr(os, "sched_getaffinity"):
        raise ValueError("CPU affinity (Linux) is required")
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < count * per_worker:
        raise ValueError(f"Need {count * per_worker} assigned CPUs, have {len(allowed)}")
    return [allowed[i * per_worker:(i + 1) * per_worker] for i in range(count)]


def current_rss(pid):
    """Resident memory of a process in bytes (Linux; 0 elsewhere)."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def completed(prefix, task_hash):
    """True if the component has been computed for exactly this task (checks the hash of its arrays)."""
    path = prefix.with_suffix(".json")
    if not path.exists():
        return False
    metadata = json.loads(path.read_text())
    if metadata.get("status") != "passed":
        return False
    if metadata["task_sha256"] != task_hash or digest(prefix.with_suffix(".npz")) != metadata["artifact_sha256"]:
        raise ValueError("Cached component checksum mismatch")
    return True


def run_group(task_path, members, workers, per_worker, folder, *, resume=False):
    """Run the component processes of `members` (up to `workers` at a time, resumable) and write group.json with the timings."""
    folder.mkdir(exist_ok=True)
    slots = cpu_slots(workers, per_worker)
    task_hash = digest(task_path)
    cached = [index for index in members if resume and completed(folder / f"component{index:02d}", task_hash)]
    pending = [index for index in members if index not in cached]
    active = {}
    peak_rss = 0
    start = time.perf_counter()
    try:
        while pending or active:
            for slot, cpus in enumerate(slots):
                if slot in active or not pending:
                    continue
                index = pending.pop(0)
                prefix = folder / f"component{index:02d}"
                stdout = prefix.with_suffix(".out").open("w")
                stderr = prefix.with_suffix(".err").open("w")
                command = [sys.executable, str(HERE / "worker.py"), "--task", str(task_path),
                           "--member-index", str(index), "--output", str(prefix),
                           "--cpu-ids", ",".join(map(str, cpus))]
                process = subprocess.Popen(command, env=native_environment(), stdout=stdout, stderr=stderr)
                active[slot] = (process, stdout, stderr, index)
            peak_rss = max(peak_rss, sum(current_rss(p[0].pid) for p in active.values()))
            for slot, (process, stdout, stderr, index) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                process.wait()
                stdout.close()
                stderr.close()
                del active[slot]
                if code != 0:
                    raise RuntimeError(f"Worker {index} exited {code}; inspect {folder}/component{index:02d}.err")
                if not completed(folder / f"component{index:02d}", task_hash):
                    raise RuntimeError(f"Worker {index} did not publish a complete artifact")
            if active:
                time.sleep(.1)
    finally:
        # Only processes launched here are ever terminated.
        for process, stdout, stderr, _ in active.values():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stdout.close()
            stderr.close()
    metadata = [json.loads((folder / f"component{i:02d}.json").read_text()) for i in members]
    result = {"workers": workers, "cpus_per_worker": per_worker,
              "wall_seconds": time.perf_counter() - start,
              "sampled_concurrent_worker_peak_rss_bytes": peak_rss,
              "sum_worker_cpu_seconds": sum(m["cpu_seconds"] for m in metadata),
              "cached_members": cached, "fresh_timing": not cached, "components": metadata}
    atomic_json(folder / "group.json", result)
    return result


def load_outputs(folder, members):
    """Arrays of the components of `members`."""
    arrays = []
    for index in members:
        with np.load(folder / f"component{index:02d}.npz", allow_pickle=False) as stored:
            arrays.append({name: stored[name].copy() for name in stored.files})
    return arrays
