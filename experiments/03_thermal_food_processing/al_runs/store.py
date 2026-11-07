"""State of one path on disk: atomic commits of phase records (JSON + npz with SHA-256 of the artifacts), a per-path lock and the state machine that returns the next phase."""

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from design import atomic_json, digest


# ------------------------------------------------------------------------------------------------ commits
@contextmanager
def locked(folder):
    """Exclusive non-blocking lock on a path folder (a second process on the same path fails)."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read(path):
    """Committed record or None; verifies the SHA-256 of the artifacts it lists."""
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    for name, expected in value.get("artifact_sha256", {}).items():
        if digest(path.parent / name) != expected:
            raise ValueError(f"Corrupted committed artifact: {name}")
    return value


def commit(path, value, artifacts=()):
    """Write a phase record once, together with the hashes of its artifacts."""
    if read(path) is not None:
        raise FileExistsError(path)
    atomic_json(path, dict(value, artifact_sha256={p.name: digest(p) for p in artifacts}))


def save_npz(path, **arrays):
    """Write an npz file so that it appears completely or not at all."""
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


# ------------------------------------------------------------------------------------------------ state machine
def validate_ranking(ranking, pool, picked):
    """Check that a ranking covers the pool, is sorted by score and starts with the pick."""
    import math
    ids = [r["simulation_id"] for r in ranking]
    values = [r["score"] for r in ranking]
    if (len(ids) != len(pool) or set(ids) != set(pool) or ids[0] != picked
            or not all(math.isfinite(v) for v in values)
            or any(a < b for a, b in zip(values, values[1:]))):
        raise ValueError("Invalid acquisition ranking")


def state(folder, plan, task):
    """Return (iteration, phase, training ids) of the next uncommitted phase, verifying the committed chain train -> evaluate -> acquire of every earlier training-set size."""
    spec = plan["tasks"][task]
    ids = [plan["warmstart_ids"][str(spec["path"])]]
    max_n = plan["max_N"][str(spec["path"])]
    for i in range(max_n):
        trained = read(folder / f"iter{i:02d}_train.json")
        if trained is None:
            return i, "train", ids
        expected_seeds = plan["reporting_seeds"][str(spec["path"])][i]
        if trained["train_ids"] != ids or trained["initialization_seeds"] != expected_seeds:
            raise ValueError("Checkpoint path or seed pairing changed")
        report = read(folder / f"iter{i:02d}.json")
        if report is None:
            return i, "evaluate", ids
        if report["train_commit_sha256"] != digest(folder / f"iter{i:02d}_train.json"):
            raise ValueError("Report uses a different trained model")
        if i == max_n-1:
            return i, "done", ids
        decision = read(folder / "selections" / f"iter{i:02d}.json")
        if decision is None:
            return i, "acquire", ids
        if decision["report_sha256"] != digest(folder / f"iter{i:02d}.json"):
            raise ValueError("Selection input report changed")
        pool = [sid for sid in plan["pool_ids"] if sid not in ids]
        if decision["pool_ids"] != pool:
            raise ValueError("Selection pool changed")
        if spec["method"] != "random":
            validate_ranking(decision["ranking"], pool, decision["picked_id"])
        elif decision["picked_id"] != plan["random_orders"][str(spec["path"])][i+1]:
            raise ValueError("Random ordering changed")
        ids = [*ids, decision["picked_id"]]
    raise AssertionError("Unreachable")
