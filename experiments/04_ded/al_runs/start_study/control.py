"""Coordinator of the DED start study: freezes the plan, then advances size by size (shared trainings, then acquisitions) either by Slurm arrays or, in local mode, task by task.

    python control.py freeze --part main         # write plan.json and manifest.json of run/<part>
    python control.py advance --part main --through-N 5
    python control.py progress --part main       # paths with their completed sizes
    python control.py recover --part main        # release writer claims of finished Slurm jobs
"""

import argparse
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import subprocess
import sys

from study_design import DATASET, HERE, LOCAL, OLD, PACKAGES, ROOT, RUN_TAG, canonical, digest, identity, plan, sources, training_request
from study_store import artifact, atomic, read, shared_ref, training, validate_members, validate_selection, writer


def run_folder(part):
    """Folder of the plan, the shared trainings and the paths of a part."""
    return HERE / "run" / (part + RUN_TAG)


def environment():
    """Package versions and pinned revisions, recorded in the manifest."""
    return {"versions": {name: version(name) for name in PACKAGES}, "numpy": version("numpy"),
            "python": sys.version.split()[0], "platform": sys.platform,
            "vcs": {name: json.loads(distribution(name).read_text("direct_url.json") or "null")
                    for name in ("dynax", "klax")}}


# ------------------------------------------------------------------------------------------------ plan and manifest
def freeze(part):
    """Write the plan and the manifest (hashes of data, code and environment) of a part."""
    folder = run_folder(part)
    value = plan(part)
    value["dataset_sha256"] = digest(DATASET)
    expected = {"plan_sha256": identity(value), "dataset_sha256": value["dataset_sha256"],
                "dataset_path": str(DATASET.relative_to(ROOT)), "resolved_dataset_path": str(DATASET.resolve()),
                "implementation_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources()},
                "environment": environment()}
    if not (ROOT / "phase_integration.json").is_file() or not (ROOT / "restart_snapshot.json").is_file():
        raise ValueError("Freeze and submit in the CPU-validated cluster repository")
    previous = folder / "manifest.json"
    if previous.exists():
        expected["git_commit"] = json.loads(previous.read_text()).get("git_commit")
    else:
        expected["git_commit"] = None
        if (ROOT / ".git").exists():
            try:
                expected["git_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
    expected["restart_snapshot_sha256"] = digest(ROOT / "restart_snapshot.json")
    if part == "gate":
        for model in ("NODE", "sPHNN"):
            for iteration in (0, 1):
                verify_old(old_checkpoint(model, 0, iteration), expected)
    atomic(folder / "plan.json", value)
    atomic(folder / "manifest.json", expected)
    verify(folder)
    return folder, value, expected


def verify(folder):
    """Regenerate the plan and check data, environment and code against the manifest; returns plan and manifest."""
    value = json.loads((folder / "plan.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    regenerated = dict(plan(value["part"]), dataset_sha256=manifest["dataset_sha256"])
    if value != regenerated or identity(value) != manifest["plan_sha256"]:
        raise ValueError("G3: plan/key/tail regeneration differs")
    if environment() != manifest["environment"] or digest(DATASET) != manifest["dataset_sha256"]:
        raise ValueError("Dataset/dependency environment changed; no mixed resume")
    if digest(ROOT / "restart_snapshot.json") != manifest["restart_snapshot_sha256"]:
        raise ValueError("Validated restart snapshot changed")
    for relative, expected in manifest["implementation_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen implementation changed: {relative}")
    return value, manifest


# ------------------------------------------------------------------------------------------------ release gate (comparison with an earlier run)
def old_checkpoint(model, repeat, iteration):
    """Record of the earlier run that the gate reproduces."""
    return OLD / model / "LA/sum_marginal" / f"path{repeat:02d}" / f"{model}_LA_sum_marginal_iter{iteration:02d}.json"


def verify_old(path, manifest):
    """Check that an earlier record was produced with the same data, environment, recipe and computational code."""
    old = json.loads(Path(path).read_text())
    source_manifest = json.loads((Path(path).parent / "manifest.json").read_text())
    if source_manifest["dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("Reference dataset differs")
    env = source_manifest["environment"]
    actual = manifest["environment"]
    if (env["versions"] != actual["versions"] or env["python"] != actual["python"]
            or env["numpy_version"] != actual["numpy"] or env["vcs_packages"] != actual["vcs"]):
        raise ValueError("Reference dependency/build environment differs")
    if (env["jax_backend"] != "cpu" or not env["jax_enable_x64"]
            or any(v != "1" for v in env["native_threads"].values())
            or old["run_config"]["training_hyperparameters"] != plan("main")["training_recipes"][old["run_config"]["model_class"]]):
        raise ValueError("Reference recipe/native-thread/backend settings differ")
    # Check the computational production files only.
    for relative, expected in source_manifest["implementation_sha256"].items():
        if relative.startswith("src/") or "/primary_investigation/" in relative or relative.endswith(("runtime.py", "pod_cache.py")):
            if digest(ROOT / relative) != expected:
                raise ValueError(f"Reference computational implementation differs: {relative}")
    for name, suffix in (("checkpoint_sha256", ".eqx"), ("pod_sha256", "_pod.npz")):
        file = Path(path).with_name(Path(path).stem + suffix)
        if digest(file) != old[name]:
            raise ValueError(f"Reference artifact changed: {file}")
    return old


# ------------------------------------------------------------------------------------------------ one size N of all paths
def path_checkpoint(folder, path, N):
    """Record of a path at size N."""
    return folder / "paths" / path["id"] / f"iter{N-1:02d}.json"


def frontier(folder, spec, N, manifest):
    """Requests of size N: reporting trainings, LA acquisition models and acquisitions (for N > 1 from the committed selections of size N-1)."""
    filename = folder / "frontiers" / f"N{N:02d}.json"
    if filename.exists():
        return json.loads(filename.read_text())
    reporting, la, acquisitions, items = {}, {}, {}, []
    for path in spec["paths"]:
        incoming = None
        if spec["part"] == "C":
            ids = [0, path["candidate"]]
        elif N == 1:
            ids = [path["start"]]
        else:
            prior = read(path_checkpoint(folder, path, N - 1))
            if prior is None:
                raise ValueError("Incomplete previous frontier")
            if path["random_tail"] is not None:
                ids = [path["start"], *path["random_tail"][:N - 1]]
            else:
                source = folder / "acquisitions" / prior["next_acquisition_id"] / "selection.json"
                incoming = read(source)
                if incoming is None:
                    raise ValueError("Previous acquisition is not committed")
                validate_selection(incoming, prior["canonical_train_ids"])
                incoming = {"picked_id": incoming["picked_id"], "ranking": incoming["ranking"],
                            "timings": incoming.get("timings", {}), "wall_seconds": incoming.get("wall_seconds"),
                            "cache": artifact(source, path_checkpoint(folder, path, N).parent)}
                ids = [*prior["train_ids"], incoming["picked_id"]]
        request = training_request(path, ids, N)
        if spec["part"] == "gate":
            reference = old_checkpoint(path["model_class"], 0, N - 1)
            old = verify_old(reference, manifest)
            if ids != old["train_ids"]:
                raise ValueError("G1: acquisition chose a different N2 dataset")
            pod = reference.with_name(reference.stem + "_pod.npz")
            request["reference_pod"] = {"path": str(pod), "sha256": digest(pod)}
        elif path["reuse_n1"] and N == 1:
            source = old_checkpoint(path["model_class"], int(path["stream"][-1]), 0)
            verify_old(source, manifest)
            request["reuse_source"] = str(source)
        elif path["reuse_existing"]:
            matches = []
            for source in sorted((OLD / path["model_class"]).glob("*/*/path*/*_iter01.json")):
                old = json.loads(source.read_text())
                if old["training_key"] == request["training_key"] and canonical(old["train_ids"]) == request["train_ids"]:
                    verify_old(source, manifest)
                    matches.append(source)
            if matches:
                request["reuse_source"] = str(matches[0])
        reporting[request["id"]] = request
        item = {"path": path, "ids": ids, "reporting_id": request["id"], "incoming": incoming}
        if path["stream"] == "LA" and N < spec["max_N"]:
            acq_model = training_request(path, ids, N, True)
            acq_model["reference_shared"] = request["id"]
            la[acq_model["id"]] = acq_model
            item["LA_model_id"] = acq_model["id"]
        if path["stream"] in ("LA", "DE", "MoLA") and N < spec["max_N"]:
            value = {"model_class": path["model_class"], "method": path["stream"],
                     "train_ids": canonical(ids), "model_id": item.get("LA_model_id", request["id"])}
            value["id"] = identity(value)
            acquisitions[value["id"]] = value
            item["acquisition_id"] = value["id"]
        items.append(item)
    value = {"N": N, "reporting": list(reporting.values()), "LA_training": list(la.values()),
             "acquisitions": list(acquisitions.values()), "items": items}
    atomic(filename, value)
    return value


def publish_paths(folder, stage, manifest):
    """Write the records of all paths at this size, pointing to the shared trainings."""
    for item in stage["items"]:
        path = item["path"]
        filename = path_checkpoint(folder, path, stage["N"])
        shared = folder / "shared" / item["reporting_id"]
        expected = shared_ref(filename.parent, shared, item["ids"], item["incoming"])
        expected.update(iteration=stage["N"] - 1, path_spec=path,
                        next_acquisition_id=item.get("acquisition_id"))
        expected["run_config"] = {"model_class": path["model_class"], "method": path["stream"],
                                  "criterion": "sum_marginal" if path["stream"] in ("LA", "DE", "MoLA") else None,
                                  "warmstart_id": path["start"], "reporting_ensemble_size": 3,
                                  "evaluation_ids": list(range(25)), "path_spec": path}
        if path["random_tail"] is not None and stage["N"] > 1:
            expected["picked_id"] = item["ids"][-1]
        validate_members(expected["evaluation"])
        atomic(filename, expected)
        # Explicit JSON pointers are authoritative; links make EQX/POD discovery ergonomic.
        for name, suffix in (("checkpoint", ".eqx"), ("pod", "_pod.npz"), ("history", "_history.pkl")):
            link = filename.with_name(filename.stem + suffix)
            target = expected["artifacts"][name]["path"]
            if link.is_symlink():
                if os.readlink(link) != target:
                    raise ValueError("Changed checkpoint link")
            elif link.exists():
                raise ValueError("Unexpected non-link checkpoint alias")
            else:
                link.symlink_to(target)
        if "LA_model_id" in item:
            acq = folder / "shared" / item["LA_model_id"]
            value = dict(read(acq / "train.json"), shared_training=artifact(acq / "train.json", filename.parent),
                         artifacts={k: artifact(acq / v["path"], filename.parent)
                                    for k, v in read(acq / "train.json")["artifacts"].items()})
            acq_file = filename.with_name(filename.stem + "_acq.json")
            atomic(acq_file, value)
            link = acq_file.with_suffix(".eqx")
            target = value["artifacts"]["checkpoint"]["path"]
            if not link.exists() and not link.is_symlink():
                link.symlink_to(target)


def gate(folder, manifest):
    """Release gate: the paths of the gate part must reproduce the earlier run bitwise."""
    checks = []
    for path in plan("gate")["paths"]:
        for N in (1, 2):
            current = read(path_checkpoint(folder, path, N))
            old = verify_old(old_checkpoint(path["model_class"], 0, N - 1), manifest)
            a, b = current["evaluation"]["members"], old["evaluation"]["members"]
            if ([m["rmse_K"] for m in a] != [m["rmse_K"] for m in b]
                    or [m["per_trajectory_rmse_K"] for m in a] != [m["per_trajectory_rmse_K"] for m in b]
                    or current["initialization_seeds"] != old["initialization_seeds"]
                    or current["dropped_indices"] != old["dropped_indices"]):
                raise ValueError(f"G1: bitwise member/per-trajectory reproduction failed: {path['id']} N={N}")
            if N == 2 and current["ranking"] != old["ranking"]:
                raise ValueError(f"G1: full acquisition ranking/scores differ: {path['id']}")
            checks.append({"model": path["model_class"], "N": N, "bitwise_errors": True,
                           "full_ranking_equal": True if N == 2 else None})
    atomic(folder / "G1_passed.json", {"status": "passed", "checks": checks,
                                      "manifest_sha256": digest(folder / "manifest.json")})


# ------------------------------------------------------------------------------------------------ Slurm and local execution
def active_jobs(folder):
    """Slurm jobs of this run that are still queued or running."""
    tracked = set()
    own = os.environ.get("SLURM_JOB_ID")
    for record in sorted((folder / "submissions").glob("*.json")):
        for job in json.loads(record.read_text())["job_ids"]:
            if job != own:
                tracked.add(job)
    if not tracked:
        return []
    output = subprocess.check_output(["squeue", "-r", "-h", "-u", os.environ["USER"],
                                      "-o", "%F|%i|%T"], text=True)
    names = []
    for line in output.splitlines():
        array, job, state = [value.strip() for value in line.split("|")]
        if job in tracked or array in tracked:
            names.append(f"{job} {state}")
    return names


def run_stage_locally(folder, name, requests, action):
    """One machine: the tasks of the batch run one after another, each in a fresh process (instead of a Slurm array)."""
    sequence = len(list((folder / "submissions").glob("*.json")))
    batch = folder / "batches" / f"{sequence:04d}_{name}_local.json"
    atomic(batch, {"action": action, "requests": requests})
    atomic(batch.with_suffix(".commit.json"), {"sha256": digest(batch)})
    atomic(folder / "submissions" / f"{sequence:04d}_local.json", {"job_ids": [], "stage": name, "batch": str(batch)})
    launcher = ROOT / "experiments/shared/cluster_recovery/run_cpu.sh"
    for task in range(len(requests)):
        print(f"{name}: task {task + 1}/{len(requests)} ({action})", flush=True)
        subprocess.run(["bash", str(launcher), "-u", str(HERE / "worker.py"), "--run", str(folder), "--batch", str(batch),
                        "--task", str(task)], check=True)


def submit_stage(folder, name, requests, action, throttle, limit):
    """Submit the requests of a stage as Slurm arrays (one array per CPU class) plus a dependent job that calls `advance` again; on one machine run them directly."""
    if LOCAL:
        return run_stage_locally(folder, name, requests, action)
    jobs = []
    groups = {}
    for request in requests:
        cpus = 32 if action == "acquire" and request["method"] == "MoLA" else 16
        groups.setdefault(cpus, []).append(request)
    for cpus, group in groups.items():
        sequence = len(list((folder / "submissions").glob("*.json")))
        batch = folder / "batches" / f"{sequence:04d}_{name}_{cpus}cpu.json"
        atomic(batch, {"action": action, "requests": group})
        atomic(batch.with_suffix(".commit.json"), {"sha256": digest(batch)})
        args = ["sbatch", "--parsable", f"--account={os.environ['PROJECT_ACCOUNT']}",
                f"--job-name=ded-start-{folder.name}", f"--array=0-{len(group)-1}%{throttle}",
                f"--cpus-per-task={cpus}", "--mem-per-cpu=4G", "--time=08:00:00",
                "--kill-on-invalid-dep=yes", "--export=ALL",
                f"--output={folder}/logs/%x-%A_%a.out", f"--error={folder}/logs/%x-%A_%a.err",
                str(HERE / "worker_cpu.sbatch"), str(folder), str(batch)]
        job = parse_job(subprocess.check_output(args, text=True))
        jobs.append(job)
        atomic(folder / "submissions" / f"{sequence:04d}_{job}.json", {"job_ids": [job], "stage": name, "batch": str(batch)})
        print(f"{name}: {len(group)} unique {action} jobs, {cpus} CPUs each; array={job}", flush=True)
    sequence = len(list((folder / "submissions").glob("*.json")))
    command = ["sbatch", "--parsable", f"--account={os.environ['PROJECT_ACCOUNT']}",
               f"--job-name=ded-start-next-{folder.name}", "--ntasks=1", "--cpus-per-task=1",
               "--mem-per-cpu=2G", "--time=00:15:00", "--kill-on-invalid-dep=yes",
               "--dependency=afterok:" + ":".join(jobs), "--export=ALL",
               f"--output={folder}/logs/%x-%j.out", f"--error={folder}/logs/%x-%j.err",
               str(HERE / "next_cpu.sbatch"), folder.name, str(limit), str(throttle)]
    job = parse_job(subprocess.check_output(command, text=True))
    atomic(folder / "submissions" / f"{sequence:04d}_{job}.json", {"job_ids": [job], "stage": "advance"})


def parse_job(output):
    """Job id from `sbatch --parsable`."""
    identifiers = [line.strip().split(";")[0] for line in output.splitlines()
                   if line.strip().split(";")[0].isdecimal()]
    if len(identifiers) != 1:
        raise ValueError(f"Unexpected sbatch --parsable output: {output}")
    return identifiers[0]


# ------------------------------------------------------------------------------------------------ coordinator
def advance(part, limit, throttle):
    """Advance until nothing is left to start (local mode) or one stage was submitted (cluster)."""
    # On the cluster a dependent Slurm job calls this again after every stage; on one machine the loop does it.
    while advance_step(part, limit, throttle) and LOCAL:
        pass


def advance_step(part, limit, throttle):
    """Runs or submits the next open stage; returns True if a stage was started."""
    folder = run_folder(part)
    spec, manifest = verify(folder)
    if not 1 <= limit <= spec["max_N"] or not 1 <= throttle <= 8:
        raise ValueError("Invalid N limit or concurrency (1..8 per resource group)")
    if part != "gate" and not LOCAL:
        gate_folder = run_folder("gate")
        gate_spec, gate_manifest = verify(gate_folder)
        gate(gate_folder, gate_manifest)
    (folder / "logs").mkdir(exist_ok=True)
    with writer(folder / "coordinator"):
        active = [] if LOCAL else active_jobs(folder)
        if active:
            print("Already queued/running; no duplicate submission:\n" + "\n".join(active))
            return
        for N in range(spec["min_N"], limit + 1):
            stage = frontier(folder, spec, N, manifest)
            for name, action, requests in (("reporting", "train", stage["reporting"]),
                                            ("LA_training", "train", stage["LA_training"])):
                pending = [r for r in requests if training(folder, r) is None]
                if pending:
                    submit_stage(folder, f"N{N:02d}_{name}", pending, action, throttle, limit)
                    return True
            publish_paths(folder, stage, manifest)
            if N < limit:
                pending = [r for r in stage["acquisitions"] if read(folder / "acquisitions" / r["id"] / "selection.json") is None]
                if pending:
                    submit_stage(folder, f"N{N:02d}_acquire", pending, "acquire", throttle, limit)
                    return True
            print(f"{part}: N={N} committed, {len(stage['reporting'])} unique reporting ensembles / {len(stage['items'])} paths", flush=True)
        if part == "gate":
            gate(folder, manifest)
            print("G1 bitwise reproduction and complete ranking passed")
            intention = folder / "queue_main.json"
            if intention.exists():
                queued = json.loads(intention.read_text())
                advance("main", queued["through_N"], queued["throttle"])
        else:
            pairing(folder, spec)
            print(f"{part}: requested frontiers through N={limit} complete (G2/G3 checked)")


def pairing(folder, spec):
    """Check that every committed path record agrees with the plan (keys, training sets, Random tails) and that identical trainings share one ensemble."""
    identities = {}
    for path in spec["paths"]:
        for N in range(spec["min_N"], spec["max_N"] + 1):
            current = read(path_checkpoint(folder, path, N))
            if current is None:
                continue
            if (current["path_spec"] != path or len(current["train_ids"]) != N
                    or current["train_ids"][0] != path["start"]
                    or canonical(current["train_ids"]) != current["canonical_train_ids"]
                    or current["training_key"] != path["reporting_key_base"] + 1000 * (N - 1)):
                raise ValueError("G2: path/seed/data-order settings differ from the frozen plan")
            if path["random_tail"] is not None and current["train_ids"] != [path["start"], *path["random_tail"][:N-1]]:
                raise ValueError("G2: Random path differs from the frozen tail")
            validate_members(current["evaluation"])
            shared = current["shared_training"]
            source = path_checkpoint(folder, path, N).parent / shared["path"]
            if digest(source) != shared["sha256"] or read(source)["evaluation"] != current["evaluation"]:
                raise ValueError("G2: shared pointer/evaluation mismatch")
            key = identity({"model": path["model_class"], "start": path["start"],
                            "ids": canonical(current["train_ids"]), "key": current["training_key"]})
            value = (shared["sha256"], current["artifacts"]["checkpoint"]["sha256"])
            if key in identities and identities[key] != value:
                raise ValueError("G2: identical sets/keys did not reuse the same ensemble")
            identities[key] = value


def progress(part):
    """Print the completed size of every path."""
    folder = run_folder(part)
    spec, manifest = verify(folder)
    pairing(folder, spec)
    for path in spec["paths"]:
        complete = [N for N in range(spec["min_N"], spec["max_N"] + 1) if read(path_checkpoint(folder, path, N))]
        print(f"{path['id']:40s} N={max(complete) if complete else 0}/{spec['max_N']}")
    print("Shared ensembles:", len(list((folder / "shared").glob("*/train.json"))))
    print("Acquisitions:", len(list((folder / "acquisitions").glob("*/selection.json"))))
    print("Unreleased writer claims:", len(list(folder.rglob(".writer_claim"))))


def recover(part):
    """Remove writer claims whose Slurm job has ended."""
    folder = run_folder(part)
    for claim in sorted(folder.rglob(".writer_claim")):
        owner = json.loads((claim / "owner.json").read_text())
        job = owner["slurm_job_id"]
        if owner.get("slurm_array_job_id") and owner.get("slurm_array_task_id"):
            job = f"{owner['slurm_array_job_id']}_{owner['slurm_array_task_id']}"
        if not job:
            raise ValueError(f"Cannot automatically prove local owner has stopped: {claim}")
        states = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", job, "--format=JobID,State%40"], text=True)
        matching = [line.split("|")[1] for line in states.splitlines() if line.split("|")[0] == job]
        terminal = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
        if len(matching) != 1 or matching[0].split()[0] not in terminal:
            raise ValueError(f"Owner is active or accounting ambiguous; do not remove {claim}")
        atomic(folder / "recovery" / f"{identity(owner)}.json", {"owner": owner, "claim": str(claim), "state": matching[0]})
        (claim / "owner.json").unlink()
        claim.rmdir()
        print("Recovered inactive writer claim:", claim)


def main():
    """Command-line entry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "advance", "progress", "recover", "check-gate", "queue-main"))
    parser.add_argument("--part", choices=("gate", "main", "B", "C"), default="main")
    parser.add_argument("--through-N", type=int, default=5)
    parser.add_argument("--throttle", type=int, default=4)
    args = parser.parse_args()
    if LOCAL and (args.part != "main" or args.command in ("recover", "queue-main", "check-gate")):
        parser.error("DED_LOCAL supports only the main part with freeze, advance and progress")
    if args.command == "freeze":
        freeze(args.part)
        print("G3 plan frozen:", run_folder(args.part) / "plan.json")
    elif args.command == "advance":
        advance(args.part, args.through_N, args.throttle)
    elif args.command == "progress":
        progress(args.part)
    elif args.command == "recover":
        recover(args.part)
    elif args.command == "queue-main":
        if not 1 <= args.through_N <= 5 or not 1 <= args.throttle <= 8:
            raise ValueError("Invalid main-run N limit/throttle")
        verify(run_folder("main"))
        atomic(run_folder("gate") / "queue_main.json", {"through_N": args.through_N, "throttle": args.throttle})
        print("Main study will start only after G1 passes; B and C are not included")
    else:
        folder = run_folder("gate")
        _, manifest = verify(folder)
        gate(folder, manifest)
        print("G1 passed")


if __name__ == "__main__":
    main()
