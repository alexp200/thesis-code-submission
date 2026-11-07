"""One task (a training or an acquisition of a frozen batch) of the extension in a fresh Slurm process; same checks as ../start_study/worker.py."""

import argparse
import json
import os
from pathlib import Path
import resource

from control import verify
from study_design import digest, identity
from study_store import atomic, execution


def main():
    """Run task --task of the frozen batch --batch in the run folder --run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    _, manifest = verify(args.run)
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise ValueError("Native BLAS threads must be one before Python starts")
    if resource.getrlimit(resource.RLIMIT_AS)[0] != resource.RLIM_INFINITY:
        raise ValueError("Set ulimit -v unlimited")
    if digest(args.batch) != json.loads(args.batch.with_suffix(".commit.json").read_text())["sha256"]:
        raise ValueError("Changed frozen task batch")
    batch = json.loads(args.batch.read_text())
    request = batch["requests"][args.task]
    keys = (("model_class", "start", "train_ids", "training_key", "ensemble_size", "role")
            if batch["action"] == "train" else ("model_class", "method", "train_ids", "model_id"))
    if identity({k: request[k] for k in keys}) != request["id"]:
        raise ValueError("Task cache identity differs")
    allocation = sorted(os.sched_getaffinity(0))
    required = 32 if batch["action"] == "acquire" and request["method"] == "MoLA" else 16
    if len(allocation) < required:
        raise ValueError(f"Requires {required} allocated CPUs")
    os.sched_setaffinity(0, set(allocation[:16]))
    import study_numerical
    try:
        if batch["action"] == "train":
            study_numerical.train(args.run, request, manifest)
        else:
            study_numerical.acquire(args.run, request, manifest, allocation)
    except Exception as error:
        atomic(args.run / "failures" / f"{os.environ.get('SLURM_JOB_ID', 'local')}_{args.task}.json",
               {"error": repr(error), "request": request, "execution": execution()})
        raise


if __name__ == "__main__":
    main()
