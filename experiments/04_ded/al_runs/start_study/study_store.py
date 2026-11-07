"""Result store of the DED workflows: immutable JSON commits (create-if-absent, artifacts listed with their SHA-256), writer claims by directory creation, and validation of reporting and selection records."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import uuid

from study_design import digest, identity


def atomic(path, value):
    """Publish a JSON record once; an identical re-publication is accepted, a different one raises."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic create-if-absent. Never replace a published commit.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != value:
                raise ValueError(f"Attempted mutation of immutable artifact: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def execution():
    """Where and how the current process runs (host, Slurm ids, threads, CPU affinity)."""
    return {"hostname": socket.gethostname(), "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "native_threads": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


@contextmanager
def writer(folder):
    """Exclusive write claim on a folder (a second writer fails instead of waiting)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    claim = folder / ".writer_claim"
    try:
        claim.mkdir()
    except FileExistsError:
        raise RuntimeError(f"Writer already claimed: {claim}; inspect owner and use explicit recover") from None
    atomic(claim / "owner.json", execution())
    try:
        yield
    finally:
        (claim / "owner.json").unlink()
        claim.rmdir()


def read(path):
    """Committed record or None; verifies the SHA-256 of its artifacts."""
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    for item in value.get("artifacts", {}).values():
        target = path.parent / item["path"]
        if not target.is_file() or digest(target) != item["sha256"]:
            raise ValueError(f"Corrupt committed artifact: {target}")
    return value


def artifact(path, parent):
    """Relative path and SHA-256 of a file."""
    return {"path": os.path.relpath(path, parent), "sha256": digest(path)}


def training(run, request):
    """Committed training of a request or None; checks that it belongs to this request and run."""
    path = run / "shared" / request["id"] / "train.json"
    value = read(path)
    if value is not None:
        if (value["request"] != request or value["training_key"] != request["training_key"]
                or value["train_ids"] != request["train_ids"]
                or value["ensemble_size"] != request["ensemble_size"]
                or value["manifest_sha256"] != digest(run / "manifest.json")):
            raise ValueError(f"Shared training configuration mismatch: {path}")
        if request["role"] == "reporting":
            validate_members(value["evaluation"])
    return value


def shared_ref(folder, shared, ids, incoming=None):
    """Record of one path at one size that points to the shared training it reads."""
    metadata = read(shared / "train.json")
    if metadata is None:
        raise ValueError(f"Shared training not published: {shared}")
    return dict(metadata, shared_training=artifact(shared / "train.json", folder),
                canonical_train_ids=metadata["train_ids"], train_ids=list(ids),
                acquisition_order=list(ids), picked_id=None if incoming is None else incoming["picked_id"],
                ranking=None if incoming is None else incoming["ranking"],
                acquisition=None if incoming is None else incoming,
                artifacts={name: artifact(shared / item["path"], folder)
                           for name, item in metadata["artifacts"].items()})


def validate_members(evaluation):
    """All three members must be evaluated on all 25 simulations with finite errors."""
    import math
    members = evaluation["members"]
    if len(members) != 3 or [m["member_index"] for m in members] != [0, 1, 2]:
        raise ValueError("All three member errors must be retained")
    for member in members:
        errors = member["per_trajectory_rmse_K"]
        if (member["status"] != "passed" or errors is None or len(errors) != 25
                or not math.isfinite(member["rmse_K"]) or any(not math.isfinite(v) for v in errors)):
            raise ValueError(f"Failed or nonfinite member evaluation: {member}")
    if evaluation["evaluation_ids"] != list(range(25)):
        raise ValueError("Evaluation must cover all 25 trajectories in ID order")


def ranking(scores, pool):
    """Pool ids sorted by descending score (ties: smaller id first)."""
    import math
    if len(scores) != len(pool) or not all(math.isfinite(float(v)) for v in scores):
        raise ValueError("Incomplete/nonfinite pool scores")
    order = sorted(range(len(pool)), key=lambda i: (-float(scores[i]), i))
    return [{"batch_id": int(pool[i]), "score": float(scores[i])} for i in order]


def validate_selection(result, ids):
    """The ranking covers the remaining pool, is sorted and starts with the pick."""
    pool = [i for i in range(25) if i not in ids]
    rows = result["ranking"]
    if result["picked_id"] != rows[0]["batch_id"] or sorted(r["batch_id"] for r in rows) != pool:
        raise ValueError("Acquisition ranking/pick does not cover the remaining pool")
    if rows != ranking([r["score"] for r in rows], [r["batch_id"] for r in rows]):
        raise ValueError("Ranking is not descending/finite")
