"""One task (a training or an acquisition of a frozen batch) in a fresh process; checks the plan, the environment and the identity of the task first."""

import argparse
import json
import os
from pathlib import Path
import resource
import sys

from control import verify
from study_design import LOCAL, digest, identity
from study_store import atomic, execution


def main():
    """Run task --task of the frozen batch --batch in the run folder --run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    folder = args.run.resolve()
    spec, manifest = verify(folder)
    if any(os.environ.get(name) != "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise ValueError("All native thread variables must equal 1 before Python starts")
    if not LOCAL and resource.getrlimit(resource.RLIMIT_AS)[0] != resource.RLIM_INFINITY:
        raise ValueError("Set ulimit -v unlimited before Python starts")
    batch = json.loads(args.batch.read_text())
    if digest(args.batch) != json.loads(args.batch.with_suffix(".commit.json").read_text())["sha256"]:
        raise ValueError("Changed frozen task batch")
    request = batch["requests"][args.task]
    keys = (("model_class", "start", "train_ids", "training_key", "ensemble_size", "role")
            if batch["action"] == "train" else ("model_class", "method", "train_ids", "model_id"))
    if identity({k: request[k] for k in keys}) != request["id"]:
        raise ValueError("Task identity differs from its data/key/model settings")
    allocation = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    required = 32 if batch["action"] == "acquire" and request["method"] == "MoLA" else 16
    if not LOCAL:
        if len(allocation) < required:
            raise ValueError(f"This task requires {required} allocated CPUs")
        os.sched_setaffinity(0, set(allocation[:16]))
    import study_numerical
    try:
        if batch["action"] == "train":
            study_numerical.train(folder, request, manifest)
        else:
            study_numerical.acquire(folder, request, manifest, allocation)
    except Exception as error:
        # Diagnostic only, not a committed result and not a hidden retry.
        target = folder / "failures" / f"{os.environ.get('SLURM_JOB_ID', 'local')}_{args.task}.json"
        atomic(target, {"request": request, "error": repr(error), "execution": execution()})
        raise


if __name__ == "__main__":
    main()
