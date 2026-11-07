"""Coordinator of the DED extension (cluster): freezes the plan on top of the validated start study and advances size by size with the kernels and stores of ../start_study.

    submit.sh prepare | check | start [through-N] [throttle]      # wrapper around this script
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import design as d
from study_design import canonical, digest, identity, training_request
from study_store import artifact, atomic, execution, read, training, validate_members, validate_selection, writer

definition = importlib.util.spec_from_file_location("ded_original_control", d.BASE / "control.py")
base = importlib.util.module_from_spec(definition)
definition.loader.exec_module(base)
RUN = d.HERE / "run"


# ------------------------------------------------------------------------------------------------ plan and manifest
def freeze():
    """Write plan, manifest and counts after verifying the start study and its release gate."""
    spec = dict(d.plan(), dataset_sha256=digest(d.original.DATASET))
    _, parent = base.verify(d.BASE / "run/main")
    _, gate_manifest = base.verify(d.BASE / "run/gate")
    base.gate(d.BASE / "run/gate", gate_manifest)
    manifest = {"plan_sha256": identity(spec), "dataset_sha256": spec["dataset_sha256"],
                "environment": base.environment(),
                "implementation_sha256": {str(p.relative_to(d.ROOT)): digest(p) for p in d.sources()},
                "original_main_manifest_sha256": digest(d.BASE / "run/main/manifest.json"),
                "original_gate_manifest_sha256": digest(d.BASE / "run/gate/manifest.json"),
                "original_G1_sha256": digest(d.BASE / "run/gate/G1_passed.json"),
                "restart_snapshot_sha256": digest(d.ROOT / "restart_snapshot.json"),
                "git_commit": parent.get("git_commit"), "dataset_path": str(d.original.DATASET.relative_to(d.ROOT))}
    references = {}
    for model in d.original.MODELS:
        for repeat in range(6):
            filename = base.old_checkpoint(model, repeat, 0)
            base.verify_old(filename, manifest)
            for path in (filename, filename.with_suffix(".eqx"), filename.with_name(filename.stem + "_pod.npz"),
                         filename.with_name(filename.stem + "_history.pkl")):
                references[str(path.relative_to(d.ROOT))] = digest(path)
    manifest["F_N1_reference_sha256"] = references
    atomic(RUN / "plan.json", spec)
    atomic(RUN / "manifest.json", manifest)
    atomic(RUN / "counts.json", d.counts(spec))
    verify(RUN)


def archive_unused_preparation():
    """Explicit upgrade of a plan that has never submitted or published work."""
    from datetime import datetime, timezone
    forbidden = list((RUN / "submissions").glob("*.json")) + list((RUN / "shared").glob("*/train.json"))
    forbidden += list((RUN / "paths").rglob("iter*.json")) + list((RUN / "acquisitions").glob("*/selection.json"))
    forbidden += list((RUN / "batches").glob("*.json")) + list(RUN.rglob(".writer_claim"))
    if forbidden:
        raise ValueError("Extension already has work or writer claims. Do not refreeze; inspect before migrating.")
    archive = RUN / "preparation_archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for filename in ("manifest.json", "plan.json", "counts.json"):
        source = RUN / filename
        if source.exists():
            archive.mkdir(parents=True, exist_ok=True)
            source.rename(archive / filename)
    print("Unused preparation preserved at:", archive)


def verify(folder=RUN):
    """Regenerate the plan and check data, environment, code and the provenance of the start study against the manifest."""
    folder = Path(folder)
    if folder.resolve() != RUN.resolve():
        raise ValueError("Worker run is not this extension's namespace")
    spec = json.loads((folder / "plan.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if (spec != dict(d.plan(), dataset_sha256=manifest["dataset_sha256"])
            or identity(spec) != manifest["plan_sha256"]):
        raise ValueError("G3 plan/key/tail regeneration differs")
    if base.environment() != manifest["environment"] or digest(d.original.DATASET) != manifest["dataset_sha256"]:
        raise ValueError("Dependency/build environment or dataset changed")
    for filename, key in ((d.ROOT / "restart_snapshot.json", "restart_snapshot_sha256"),
                          (d.BASE / "run/main/manifest.json", "original_main_manifest_sha256"),
                          (d.BASE / "run/gate/manifest.json", "original_gate_manifest_sha256"),
                          (d.BASE / "run/gate/G1_passed.json", "original_G1_sha256")):
        if digest(filename) != manifest[key]:
            raise ValueError(f"Changed validated provenance: {filename}")
    for relative, expected in manifest["implementation_sha256"].items():
        if digest(d.ROOT / relative) != expected:
            raise ValueError(f"Frozen implementation changed: {relative}")
    for relative, expected in manifest["F_N1_reference_sha256"].items():
        if digest(d.ROOT / relative) != expected:
            raise ValueError(f"Frozen F N1 reference changed: {relative}")
    return spec, manifest


# ------------------------------------------------------------------------------------------------ one size N of all paths
def checkpoint(path, N):
    """Record of a path at size N."""
    return base.path_checkpoint(RUN, path, N)


def frontier(spec, N):
    """Requests of size N (reporting trainings, LA acquisition models, acquisitions) of all paths that reach this size."""
    filename = RUN / "frontiers" / f"N{N:02d}.json"
    if filename.exists():
        return json.loads(filename.read_text())
    reporting, la, acquisitions, items = {}, {}, {}, []
    for path in spec["paths"]:
        if not d.active(path, N):
            continue
        incoming = None
        if N == 1:
            ids = [path["start"]]
        elif path["candidate"] is not None:
            ids = [path["start"], path["candidate"]]
        elif path["random_tail"] is not None:
            ids = [path["start"], *path["random_tail"][:N-1]]
        else:
            prior = read(checkpoint(path, N-1))
            if prior is None:
                raise ValueError("Previous reporting frontier is incomplete")
            source = RUN / "acquisitions" / prior["next_acquisition_id"] / "selection.json"
            incoming = read(source)
            if incoming is None:
                raise ValueError("Previous acquisition is incomplete")
            validate_selection(incoming, prior["canonical_train_ids"])
            incoming = {"picked_id": incoming["picked_id"], "ranking": incoming["ranking"],
                        "timings": incoming.get("timings", {}), "wall_seconds": incoming.get("wall_seconds"),
                        "cache": artifact(source, checkpoint(path, N).parent)}
            ids = [*prior["train_ids"], incoming["picked_id"]]
        request = training_request(path, ids, N)
        reporting[request["id"]] = request
        item = {"path": path, "ids": ids, "reporting_id": request["id"], "incoming": incoming}
        if path["stream"] == "LA" and N < path["max_N"]:
            acq_model = training_request(path, ids, N, True)
            acq_model["reference_shared"] = request["id"]
            la[acq_model["id"]] = acq_model
            item["LA_model_id"] = acq_model["id"]
        if path["stream"] in ("LA", "DE", "MoLA") and N < path["max_N"]:
            acq = {"model_class": path["model_class"], "method": path["stream"],
                   "train_ids": canonical(ids), "model_id": item.get("LA_model_id", request["id"])}
            acq["id"] = identity(acq)
            acquisitions[acq["id"]] = acq
            item["acquisition_id"] = acq["id"]
        items.append(item)
    stage = {"N": N, "reporting": list(reporting.values()), "LA_training": list(la.values()),
             "acquisitions": list(acquisitions.values()), "items": items}
    atomic(filename, stage)
    return stage


def reuse_source(request, manifest):
    """Committed record of the start study (or of the earlier start-0 run) that already contains this training, or None."""
    source = d.BASE / "run/main/shared" / request["id"] / "train.json"
    if source.exists():
        original = training(d.BASE / "run/main", request)
        if original is None or original["training_hyperparameters"] != d.original.recipe(request["model_class"]):
            raise ValueError("Original study recipe/request mismatch")
        return source, original, False
    if request["start"] != 0 or request["role"] != "reporting" or request["dataset_size"] > 2:
        return None
    iteration = request["dataset_size"] - 1
    if iteration == 0:
        offset = request["training_key"] - 5000000
        if offset < 0 or offset % 100000 or offset // 100000 not in range(6):
            raise ValueError("Invalid F/C N1 reference key")
        source = base.old_checkpoint(request["model_class"], offset // 100000, 0)
        row = base.verify_old(source, manifest)
        if row["train_ids"] != [0] or row["training_key"] != request["training_key"] or row["ensemble_size"] != 3:
            raise ValueError("F N1 reference is not the exact LA path/key/K")
        return source, row, True
    for source in sorted((d.original.OLD / request["model_class"]).glob(f"*/*/path*/*_iter{iteration:02d}.json")):
        row = json.loads(source.read_text())
        if (row["training_key"] == request["training_key"] and row["train_ids"] == request["train_ids"]
                and row["ensemble_size"] == 3):
            return source, base.verify_old(source, manifest), True
    return None


def import_reuse(request, manifest):
    """Copy a reusable training into this run (checkpoint, POD, history, record) after checking its hashes."""
    import shutil
    if training(RUN, request) is not None:
        return True
    source = reuse_source(request, manifest)
    if source is None:
        if request["start"] == 0 and request["dataset_size"] == 1:
            raise ValueError("F/C require the exact stored N1 AL checkpoint; never retrain it")
        if request["start"] in d.STARTS[:5] and request["dataset_size"] == 1:
            raise ValueError("Existing-start source missing; finish original A before extension")
        return False
    filename, row, legacy = source
    target = RUN / "shared" / request["id"]
    with writer(target):
        if legacy:
            src = {"checkpoint": filename.with_suffix(".eqx"),
                   "pod": filename.with_name(filename.stem + "_pod.npz"),
                   "history": filename.with_name(filename.stem + "_history.pkl")}
            expected = {"checkpoint": row["checkpoint_sha256"], "pod": row["pod_sha256"],
                        "history": row.get("history_sha256") or digest(src["history"])}
        else:
            src = {name: filename.parent / item["path"] for name, item in row["artifacts"].items()}
            expected = {name: item["sha256"] for name, item in row["artifacts"].items()}
        for name, output in (("checkpoint", "model.eqx"), ("pod", "pod.npz"), ("history", "history.pkl")):
            if digest(src[name]) != expected[name]:
                raise ValueError("Reuse source artifact checksum mismatch")
            shutil.copyfile(src[name], target / (output + ".tmp"))
            (target / (output + ".tmp")).replace(target / output)
            if digest(target / output) != expected[name]:
                raise ValueError("Reuse copy checksum mismatch")
        metadata = dict(row, status="passed", request=request, train_ids=request["train_ids"],
                        dataset_size=request["dataset_size"], model_class=request["model_class"],
                        training_hyperparameters=d.original.recipe(request["model_class"]),
                        manifest_sha256=digest(RUN / "manifest.json"), reused_from=artifact(filename, target),
                        reuse_execution=execution(),
                        reused_training_seconds=row["train_seconds"], train_seconds=0., wall_seconds=0.,
                        reuse_history_checksum_origin="legacy_record" if row.get("history_sha256") else "measured_at_import",
                        artifacts={name: artifact(target / output, target) for name, output in
                                   (("checkpoint", "model.eqx"), ("pod", "pod.npz"), ("history", "history.pkl"))})
        validate_members(metadata["evaluation"])
        atomic(target / "train.json", metadata)
    return True


def pairing(spec):
    """Check every published record against the plan; returns the number of cells."""
    identities = {}
    count = 0
    for path in spec["paths"]:
        for N in range(1, path["max_N"] + 1):
            filename = checkpoint(path, N)
            row = read(filename)
            if row is None:
                continue
            request = training_request(path, row["train_ids"], N)
            expected_ids = ([path["start"], path["candidate"]] if N == 2 and path["candidate"] is not None
                            else [path["start"], *path["random_tail"][:N-1]] if path["random_tail"] is not None
                            else None)
            if (row["path_spec"] != path or row["training_key"] != request["training_key"]
                    or row["canonical_train_ids"] != request["train_ids"]
                    or (expected_ids is not None and row["train_ids"] != expected_ids)
                    or row["train_ids"][0] != path["start"]):
                raise ValueError("G2 dataset/path/seed mismatch")
            shared = filename.parent / row["shared_training"]["path"]
            if digest(shared) != row["shared_training"]["sha256"] or read(shared)["evaluation"] != row["evaluation"]:
                raise ValueError("G2 shared evaluation/commit mismatch")
            validate_members(row["evaluation"])
            fingerprint = (row["shared_training"]["sha256"], row["artifacts"]["checkpoint"]["sha256"])
            if request["id"] in identities and identities[request["id"]] != fingerprint:
                raise ValueError("G2 identical sets/keys were trained more than once")
            identities[request["id"]] = fingerprint
            count += 1
    return count


# ------------------------------------------------------------------------------------------------ scheduling on the cluster
def original_complete(through_N=4):
    """Raise unless the start study has published all its cells through size through_N."""
    spec, _ = base.verify(d.BASE / "run/main")
    base.pairing(d.BASE / "run/main", spec)
    missing = [str(base.path_checkpoint(d.BASE / "run/main", p, N)) for p in spec["paths"]
               for N in range(1, through_N+1) if read(base.path_checkpoint(d.BASE / "run/main", p, N)) is None]
    if missing:
        raise ValueError(f"A is not published through N={through_N}: {len(missing)} missing cells.")


def original_requests():
    """Requests that belong to the start study and must not be trained again here."""
    return {r["id"] for filename in (d.BASE / "run/main/frontiers").glob("N*.json")
            for r in json.loads(filename.read_text())["reporting"]}


def submitted_work():
    """Last submitted job of every request (and whether it is still queued or running), and the live jobs of the foreground stages."""
    output = subprocess.check_output(["squeue", "-r", "-h", "-u", os.environ["USER"],
                                      "-o", "%i|%F|%T"], text=True)
    live = {line.split("|")[0].strip() for line in output.splitlines()}
    attempts, foreground = {}, []
    own = os.environ.get("SLURM_JOB_ID")
    for filename in sorted((RUN / "submissions").glob("*.json")):
        record = json.loads(filename.read_text())
        for job in record["job_ids"]:
            if job != own and record["stage"] != "background":
                foreground.extend(j for j in live if j == job or j.startswith(job + "_"))
            if "batch" in record:
                batch = json.loads(Path(record["batch"]).read_text())
                for i, request in enumerate(batch["requests"]):
                    attempts[request["id"]] = {"job": f"{job}_{i}", "active": f"{job}_{i}" in live}
    return attempts, foreground


def scheduling_options():
    """sbatch options from the environment: reservation, partition, earliest start."""
    args = []
    for variable, flag in (("FINAL_RESERVATION", "reservation"), ("FINAL_PARTITION", "partition")):
        value = os.environ.get(variable)
        if value:
            args.append(f"--{flag}={value}")
    start = reservation_begin()
    if start:
        args.append(f"--begin={start}")
    return args


def reservation_begin():
    """Start time of the confirmed reservation."""
    from datetime import datetime, timezone
    value = os.environ.get("FINAL_RESERVATION_BEGIN")
    if not value:
        return None
    start = datetime.fromisoformat(value)
    if start.tzinfo is None:
        raise ValueError("FINAL_RESERVATION_BEGIN must include timezone")
    if start > datetime.now(timezone.utc):
        return start.astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    return None


def submit_coordinator(dependencies, limit, throttle):
    """Submit the job that calls `advance` again after the given jobs."""
    sequence = len(list((RUN / "submissions").glob("*.json")))
    command = ["sbatch", "--parsable", f"--account={os.environ['PROJECT_ACCOUNT']}",
               "--job-name=ded-final-next", "--ntasks=1", "--cpus-per-task=1", "--mem-per-cpu=2G",
               "--time=00:15:00", "--kill-on-invalid-dep=yes", "--dependency=afterok:" + ":".join(dependencies),
               "--export=ALL", *scheduling_options(), f"--output={RUN}/logs/%x-%j.out",
               f"--error={RUN}/logs/%x-%j.err", str(d.HERE / "next_cpu.sbatch"), str(limit), str(throttle)]
    job = base.parse_job(subprocess.check_output(command, text=True))
    atomic(RUN / "submissions" / f"{sequence:04d}_{job}.json", {"job_ids": [job], "stage": "advance"})
    print(f"Coordinator={job}; waits only for {len(dependencies)} required job(s)")


def submit_stage(name, requests, action, throttle, limit, *, background=False):
    """Submit the requests of a stage as Slurm arrays."""
    jobs = []
    groups = {}
    for request in requests:
        cpus = 32 if action == "acquire" and request["method"] == "MoLA" else 16
        groups.setdefault(cpus, []).append(request)
    for cpus, group in groups.items():
        sequence = len(list((RUN / "submissions").glob("*.json")))
        batch = RUN / "batches" / f"{sequence:04d}_{name}_{cpus}cpu.json"
        atomic(batch, {"action": action, "requests": group})
        atomic(batch.with_suffix(".commit.json"), {"sha256": digest(batch)})
        command = ["sbatch", "--parsable", f"--account={os.environ['PROJECT_ACCOUNT']}",
                   "--job-name=ded-final-bg" if background else "--job-name=ded-final",
                   f"--array=0-{len(group)-1}%{min(throttle, 192//cpus)}",
                   f"--cpus-per-task={cpus}", "--mem-per-cpu=3500M", "--time=08:00:00",
                   "--nice=10000" if background else "--nice=0",
                   "--kill-on-invalid-dep=yes", "--export=ALL", *scheduling_options(),
                   f"--output={RUN}/logs/%x-%A_%a.out", f"--error={RUN}/logs/%x-%A_%a.err",
                   str(d.HERE / "worker_cpu.sbatch"), str(RUN), str(batch)]
        job = base.parse_job(subprocess.check_output(command, text=True))
        jobs.append(job)
        atomic(RUN / "submissions" / f"{sequence:04d}_{job}.json",
               {"job_ids": [job], "stage": "background" if background else name, "batch": str(batch)})
    if not background:
        submit_coordinator(jobs, limit, throttle)
    print(f"Submitted {name}: workers={jobs}")


def fill_background(spec, manifest, limit, throttle, attempts, retry):
    """Submit trainings that do not depend on earlier results to fill idle slots."""
    reserved = original_requests()
    pending = []
    for request in d.static_requests(spec):
        if request["dataset_size"] > limit or training(RUN, request) is not None:
            continue
        previous = attempts.get(request["id"])
        if previous and (previous["active"] or not retry):
            continue
        if import_reuse(request, manifest):
            continue
        if request["id"] not in reserved:
            pending.append(request)
    if pending:
        submit_stage("independent", pending, "train", throttle, limit, background=True)


def wait_for_original(limit, throttle):
    """Wait for the start study to publish a model that this run reads; never retrains it."""
    live = {row.split()[0] for row in base.active_jobs(d.BASE / "run/main")}
    for filename in sorted((d.BASE / "run/main/submissions").glob("*.json"), reverse=True):
        record = json.loads(filename.read_text())
        if record["stage"] == "advance":
            for job in record["job_ids"]:
                if job in live:
                    submit_coordinator([job], limit, throttle)
                    print("Waiting for original A publisher; no duplicate A training")
                    return
    raise ValueError("A-owned model missing and no live A publisher. Inspect/resume A; do not retrain it here.")


def advance(limit, throttle, retry=False):
    """Start the next open stage (needs the approved reservation)."""
    if os.environ.get("FINAL_LAUNCH_APPROVED") != "yes" or not os.environ.get("FINAL_RESERVATION"):
        raise ValueError("Explicit approval and the HPC-confirmed reservation are required, including for child jobs")
    spec, manifest = verify()
    if not 1 <= limit <= 5 or not 1 <= throttle <= 32:
        raise ValueError("through-N=1..5; throttle=1..32 per resource group")
    original_complete()
    with writer(RUN / "coordinator"):
        attempts, active = submitted_work()
        if active:
            print("Already queued/running; no duplicate submission:\n" + "\n".join(active))
            return
        (RUN / "logs").mkdir(exist_ok=True)
        fill_background(spec, manifest, limit, throttle, attempts, retry)
        attempts, _ = submitted_work()
        reserved = original_requests()
        for N in range(1, limit + 1):
            stage = frontier(spec, N)
            for request in stage["reporting"]:
                if training(RUN, request) is None:
                    import_reuse(request, manifest)
            for name, action, requests in (("reporting", "train", stage["reporting"]),
                                           ("LA_training", "train", stage["LA_training"])):
                pending = [r for r in requests if training(RUN, r) is None]
                if pending:
                    owned = [r for r in pending if r["id"] in reserved]
                    if owned:
                        wait_for_original(limit, throttle)
                        return
                    waiting = [attempts[r["id"]]["job"] for r in pending
                               if r["id"] in attempts and attempts[r["id"]]["active"]]
                    failed = [r for r in pending if r["id"] in attempts and not attempts[r["id"]]["active"]]
                    if failed and not retry:
                        raise ValueError("Previously submitted training is uncommitted/terminal. Inspect logs, recover claims, then explicitly --retry-failed.")
                    unscheduled = [r for r in pending if r["id"] not in attempts or (retry and not attempts[r["id"]]["active"])]
                    if unscheduled:
                        submit_stage(f"N{N:02d}_{name}", unscheduled, action, throttle, limit)
                    else:
                        submit_coordinator(sorted(set(waiting)), limit, throttle)
                    return
            base.publish_paths(RUN, stage, manifest)
            if N < limit:
                pending = [r for r in stage["acquisitions"]
                           if read(RUN / "acquisitions" / r["id"] / "selection.json") is None]
                if pending:
                    submit_stage(f"N{N:02d}_acquire", pending, "acquire", throttle, limit)
                    return
            print(f"N={N} complete: {len(stage['items'])} cells, {len(stage['reporting'])} distinct K3 models")
        print(f"G2/G3 passed; {pairing(spec)} published cells")


def progress():
    """Published cells per package."""
    spec = json.loads((RUN / "plan.json").read_text()) if (RUN / "plan.json").exists() else d.plan()
    for package in d.PACKAGES:
        paths = [p for p in spec["paths"] if p["package"] == package]
        completed = sum(read(checkpoint(p, N)) is not None for p in paths for N in range(1, p["max_N"]+1))
        expected = sum(p["max_N"] for p in paths)
        print(f"{package}: {completed}/{expected} cells (including N1 references); {len(paths)} paths/draws")
    print("G2 validated published cells:", pairing(spec))
    print("Unreleased claims:", len(list(RUN.rglob(".writer_claim"))))


def recover():
    """Remove writer claims whose Slurm job has ended."""
    verify()
    for claim in sorted(RUN.rglob(".writer_claim")):
        owner = json.loads((claim / "owner.json").read_text())
        job = owner.get("slurm_job_id")
        if owner.get("slurm_array_job_id") and owner.get("slurm_array_task_id") is not None:
            job = f"{owner['slurm_array_job_id']}_{owner['slurm_array_task_id']}"
        if not job:
            raise ValueError("Cannot prove that this non-Slurm writer has stopped")
        output = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", job,
                                          "--format=JobID,State%40"], text=True)
        states = [line.split("|")[1].strip().split()[0] for line in output.splitlines()
                  if line.split("|")[0].strip() == job]
        if len(states) != 1 or states[0] not in {
                "COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}:
            raise ValueError(f"Owner active or accounting ambiguous: {job}; keep {claim}")
        atomic(RUN / "recovery" / f"{identity(owner)}.json", {"owner": owner, "claim": str(claim), "state": states[0]})
        (claim / "owner.json").unlink()
        claim.rmdir()
        print("Recovered inactive claim:", claim)


def main():
    """Command-line entry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "freeze", "check", "advance", "progress", "recover", "upgrade-preparation"))
    parser.add_argument("--through-N", type=int, default=5)
    parser.add_argument("--throttle", type=int, default=12)
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly retry uncommitted terminal attempts after log/claim inspection")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps({"design": d.plan(), "counts": d.counts(d.plan())}, indent=2))
    elif args.command == "freeze":
        freeze()
        print("Frozen. No jobs submitted.")
    elif args.command == "upgrade-preparation":
        archive_unused_preparation()
        freeze()
        print("Unused preparation upgraded. No jobs submitted.")
    elif args.command == "check":
        spec, _ = verify()
        original_complete()
        print("Plan, source hashes, environment, original G1/A and pairing passed:", pairing(spec))
    elif args.command == "progress":
        progress()
    elif args.command == "recover":
        recover()
    else:
        advance(args.through_N, args.throttle, args.retry_failed)


if __name__ == "__main__":
    main()
