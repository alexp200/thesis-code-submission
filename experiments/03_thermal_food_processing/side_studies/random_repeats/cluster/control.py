"""Frozen, isolated Random repeats with atomic ensemble publication."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[4]
RUN = Path(os.environ.get("RR_RUN_DIR", CODE / "run"))
N2 = ROOT / "experiments/03_thermal_food_processing/side_studies/n2_landscape"
PACKAGES = ("jax", "jaxlib", "equinox", "diffrax", "klax", "dynax", "optax",
            "numpy", "scipy", "xarray", "h5netcdf", "h5py")
DATA_SHA = "2ff82f83226396abe62c8b9662ad828a9c8db3abcbf06f8b005bc6e71d8590c8"
TEST = [313, 320, 344, 378, 383, 407, 412, 415, 461, 462, 466, 467, 474, 508, 528]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def dependencies():
    versions, urls = {}, {}
    for package in PACKAGES:
        dist = importlib.metadata.distribution(package)
        versions[package] = dist.version
        direct = dist.read_text("direct_url.json")
        if direct:
            urls[package] = json.loads(direct)
    return dict(python=platform.python_version(), versions=versions, direct_urls=urls)


def design():
    # Regenerate privately; never overwrite the binding/local design or old Landscape runner.
    import plan as generator
    with tempfile.TemporaryDirectory() as temporary:
        before = generator.HERE
        try:
            generator.HERE = Path(temporary)
            generator.main()
            generated = json.loads((generator.HERE / "plan.json").read_text())
            if (generator.HERE / "tasks.tsv").read_bytes() != (CODE.parent / "tasks.tsv").read_bytes():
                raise ValueError("Binding tasks.tsv differs from regeneration")
        finally:
            generator.HERE = before
    if generated != json.loads((CODE.parent / "plan.json").read_text()):
        raise ValueError("Binding plan.json differs from regeneration")
    if len(generated["tasks"]) != 880 or any(sum(t["arch"] == a for t in generated["tasks"]) != 440 for a in ("NODE", "sPHNN")):
        raise ValueError("Expected 880 tasks, 440 per architecture")
    return generated


def references(source):
    import landscape_run as landscape
    result = []
    for arch in ("NODE", "sPHNN"):
        for block, n in ((0, 3), (4, 4), (9, 5)):
            folder = source / f"experiments/03_thermal_food_processing/al_runs/study/results/{arch}/random/random/path{block:02d}"
            train = folder / f"iter{n-1:02d}_train.json"
            report = folder / f"iter{n-1:02d}.json"
            meta, value = json.loads(train.read_text()), json.loads(report.read_text())
            ids = meta["train_ids"]
            key = landscape.n1_key(ids[0]) + 1000 * (n-1)
            recipe = landscape.RECIPE[arch]
            if (landscape.BLOCK.get(ids[0]) != block or len(ids) != n
                    or meta["initialization_seeds"] != [key+k for k in range(3)]
                    or meta["training_key"] != key or meta["optimizer_key"] != key+1
                    or any(meta["recipe"][k] != v for k, v in recipe.items())
                    or value["train_ids"] != ids or value["initialization_seeds"] != [key+k for k in range(3)]
                    or [m["member_index"] for m in value["evaluation"]["members"]] != [0, 1, 2]
                    or value["evaluation"]["evaluation_ids"] != TEST):
                raise ValueError(f"Stored reference design differs: {train}")
            errors = [m["rmse_K"] for m in value["evaluation"]["members"]]
            finite_metrics(errors, [m["per_trajectory_rmse_K"] for m in value["evaluation"]["members"]])
            result.append(dict(task=len(result), arch=arch, block=block, N=n, train_ids=ids,
                               training_key=key, optimizer_key=key+1, paths=[], start=ids[0],
                               **recipe, reference_file=str(report), reference_sha256=digest(report),
                               train_file=str(train), train_sha256=digest(train), reference_members_K=errors,
                               reference_trajectory_K=[m["per_trajectory_rmse_K"] for m in value["evaluation"]["members"]]))
    return result


def source_paths(source):
    paths = [*CODE.glob("*.py"), *CODE.glob("*.sh"), *CODE.glob("*.sbatch"), CODE.parent / "plan.json", CODE.parent / "tasks.tsv"]
    paths += list((source / "src").rglob("*.py"))
    for relative in ("al_runs/design.py", "al_runs/numerical.py",
                     "kernels/core/train_common.py", "kernels/core/chicken_data.py"):
        paths.append(source / "experiments/03_thermal_food_processing" / relative)
    paths.append(source / "examples/dynamics/chicken_dataset/chicken_data.py")
    if not (source / "src").is_dir() or any(not p.is_file() for p in paths):
        raise ValueError("Missing validated source")
    return sorted(set(p.resolve() for p in paths))


def prepare():
    scientific = design()
    parent_file = Path(os.environ.get("RR_PARENT_PLAN", N2 / "cluster/run/plan.json"))
    parent = json.loads(parent_file.read_text())
    source, data = Path(parent["source"]), Path(parent["data"])
    identity = dependencies()
    if identity != parent["dependencies"]:
        raise ValueError("Use the original Landscape Python environment")
    for name, expected in parent["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Original frozen source changed: {name}")
    if digest(data / "dataset.nc") != DATA_SHA or parent["evaluation_ids"] != TEST or parent["ensemble_size"] != 3:
        raise ValueError("Original dataset/evaluation/K differs")
    import landscape_run as landscape
    if parent["recipe"] != landscape.RECIPE or parent["pool_ids"] != scientific["pool"]:
        raise ValueError("Original recipe/pool differs")
    ordered = sorted(scientific["tasks"], key=lambda t: (t["N"], t["arch"], t["start"], t["task"]))
    groups = [[t["task"] for t in ordered[i:i+4]] for i in range(0, len(ordered), 4)]
    frozen = dict(schema_version=1, scientific_design=scientific, groups=groups,
                  source=str(source), data=str(data), dataset_sha256=DATA_SHA, dependencies=identity,
                  source_sha256={str(p): digest(p) for p in source_paths(source)},
                  parent_plan_file=str(parent_file.resolve()), parent_plan_sha256=digest(parent_file),
                  landscape_roots=[str(parent_file.parent), str(N2 / "cluster_extra4/run")],
                  evaluation_ids=TEST, reproduction_tasks=references(source),
                  smoke_tasks=[next(t["task"] for t in scientific["tasks"] if t["arch"] == a and t["N"] == n)
                               for a in ("NODE", "sPHNN") for n in (3, 5)],
                  reproduction_review_threshold_K=1e-10)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = RUN / "manifest.json"
        if out.exists():
            old = json.loads(out.read_text())
            if {k: old[k] for k in frozen} != frozen:
                raise ValueError("Frozen source/design changed; do not overwrite or mix runs")
        else:
            git = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
            status = subprocess.run(["git", "-C", str(source), "status", "--short"], capture_output=True, text=True)
            write(out, dict(frozen, created_utc=datetime.now(timezone.utc).isoformat(),
                            git_commit=git.stdout.strip(), git_status=status.stdout.strip()))
    print("Frozen 880 ensembles / 220 Slurm groups; 4 smokes, 6 reproduction pilots")


def load():
    plan = json.loads((RUN / "manifest.json").read_text())
    if dependencies() != plan["dependencies"]:
        raise ValueError("Dependencies changed")
    for path, expected in plan["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    if digest(Path(plan["data"]) / "dataset.nc") != plan["dataset_sha256"] or digest(plan["parent_plan_file"]) != plan["parent_plan_sha256"]:
        raise ValueError("Dataset/parent plan changed")
    for t in plan["reproduction_tasks"]:
        if digest(t["reference_file"]) != t["reference_sha256"] or digest(t["train_file"]) != t["train_sha256"]:
            raise ValueError("Reproduction reference changed")
    return plan


def result_path(task, mode):
    folder = RUN / {"full": "results", "smoke": "smoke_results", "pilot": "reproduction"}[mode]
    return folder / task["arch"] / f"start{task['start']}" / f"N{task['N']}_task{task['task']}.json"


def finite_metrics(members, trajectories):
    if len(members) != 3 or len(trajectories) != 3 or any(len(row) != 15 for row in trajectories):
        raise ValueError("Expected all three members and AP15 per-trajectory errors")
    if any(not math.isfinite(x) or x < 0 for row in [members, *trajectories] for x in row):
        raise ValueError("Nonfinite/negative metric")


def validate(path, task, mode, model_name=None):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (value.get("cluster_manifest_sha256") != digest(RUN / "manifest.json")
            or value.get("task_spec") != task or value.get("mode") != mode):
        raise ValueError(f"Wrong identity: {path}")
    if (value["steps"] != (2 if mode == "smoke" else task["steps"])
            or value["train_ids"] != task["train_ids"] or value["training_key"] != task["training_key"]
            or value["optimizer_key"] != task["optimizer_key"]
            or value["initialization_seeds"] != [task["training_key"]+k for k in range(3)]
            or value["evaluation_ids"] != TEST or value["smoke"] != (mode == "smoke")
            or value["lr"] != task["lr"] or value["l2"] != task["l2"]):
        raise ValueError("Wrong recipe, seeds, training/evaluation IDs")
    finite_metrics(value["member_rmse_K"], value["per_trajectory_rmse_K"])
    if len(value["train_rmse_latent"]) != 3 or any(not math.isfinite(x) or x < 0 for x in value["train_rmse_latent"]):
        raise ValueError("Invalid training losses")
    artifacts = value["artifact_sha256"]
    if (model_name or path.with_suffix(".eqx").name) not in artifacts:
        raise ValueError("Missing committed model hash")
    for name, expected in artifacts.items():
        if Path(name).name != name or (path.parent / name).stat().st_size == 0 or digest(path.parent / name) != expected:
            raise ValueError(f"Corrupt artifact: {name}")
    return value


def check(plan, mode):
    if mode == "n2":
        cells, lookup = [], {}
        requested = [dict(arch=a, start=s, candidate=c) for a in ("NODE", "sPHNN")
                     for s in plan["scientific_design"]["pool"] for c in plan["scientific_design"]["pool"] if c != s]
        for cell in requested:
            reports = [Path(r) / "results" / cell["arch"] / f"start{cell['start']}" / f"cand{cell['candidate']}_setA.json" for r in plan["landscape_roots"]]
            found = [p for p in reports if p.exists()]
            if len(found) != 1:
                raise ValueError(f"N=2 cell missing/ambiguous: {cell}; finish Landscape A first")
            p = found[0]
            value = json.loads(p.read_text())
            parent = json.loads((p.parents[3] / "plan.json").read_text())
            if (value["cluster_plan_sha256"] != digest(p.parents[3] / "plan.json")
                    or value["seed_set"] != "A" or value["train_ids"] != [cell["start"], cell["candidate"]]
                    or value["training_key"] != __import__("landscape_run").n1_key(cell["start"])+1000
                    or value["optimizer_key"] != value["training_key"]+1 or value["smoke"]
                    or value["initialization_seeds"] != [value["training_key"]+k for k in range(3)]
                    or value["arch"] != cell["arch"] or value["steps"] != parent["recipe"][cell["arch"]]["steps"]
                    or value["evaluation_ids"] != TEST or value["dataset_sha256"] != DATA_SHA
                    or parent["dependencies"] != plan["dependencies"]
                    or parent["source"] != plan["source"] or parent["data"] != plan["data"]
                    or parent["recipe"] != __import__("landscape_run").RECIPE):
                raise ValueError(f"Wrong N=2 source: {p}")
            finite_metrics(value["member_rmse_K"], value["per_trajectory_rmse_K"])
            if p.with_suffix(".eqx").name not in value["artifact_sha256"]:
                raise ValueError("N=2 model hash missing")
            for name, expected in value["artifact_sha256"].items():
                if digest(p.parent / name) != expected:
                    raise ValueError(f"Corrupt N=2 artifact: {p}")
            lookup[(cell["arch"], cell["start"], cell["candidate"])] = dict(result_file=str(p), result_sha256=digest(p))
        for cell in plan["scientific_design"]["landscape_n2"]:
            cells.append(dict(**cell, **lookup[(cell["arch"], cell["start"], cell["candidate"])]))
        write(RUN / "landscape_n2_index.json", dict(cluster_manifest_sha256=digest(RUN / "manifest.json"), cells=cells))
        print("Validated all 300 N=2 path cells (reused Landscape A; no new training)")
        return
    ids = plan["smoke_tasks"] if mode == "smoke" else range(6)
    tasks = plan["scientific_design"]["tasks"] if mode == "smoke" else plan["reproduction_tasks"]
    for i in ids:
        t = tasks[i]
        value = validate(result_path(t, mode), t, mode)
        if value is None:
            raise ValueError(f"Missing {mode} task {i}")
        if mode == "pilot":
            differences = [x-y for x, y in zip(value["member_rmse_K"], t["reference_members_K"])]
            trajectory_differences = [x-y for row, ref in zip(value["per_trajectory_rmse_K"], t["reference_trajectory_K"]) for x,y in zip(row, ref)]
            largest = max(abs(x) for x in differences + trajectory_differences)
            print(f"{t['arch']} block={t['block']} N={t['N']}: max member/trajectory delta={largest:.12g} K; bitwise_metrics={largest == 0}")
            if largest > plan["reproduction_review_threshold_K"]:
                raise ValueError("Reproduction requires review; full run not released")
    print(f"{mode} checks passed")


def run(task_id, mode):
    plan = load()
    t = (plan["reproduction_tasks"] if mode == "pilot" else plan["scientific_design"]["tasks"])[task_id]
    if mode == "smoke" and task_id not in plan["smoke_tasks"]:
        raise ValueError("Not a smoke task")
    if mode != "smoke":
        check(plan, "smoke")
    if mode == "full":
        check(plan, "pilot")
        index = json.loads((RUN / "landscape_n2_index.json").read_text())
        if index["cluster_manifest_sha256"] != digest(RUN / "manifest.json") or len(index["cells"]) != 300:
            raise ValueError("N=2 index identity differs")
        for cell, expected in zip(index["cells"], plan["scientific_design"]["landscape_n2"]):
            if any(cell[k] != v for k,v in expected.items()) or digest(cell["result_file"]) != cell["result_sha256"]:
                raise ValueError("N=2 reference changed")
    if (os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get("JAX_ENABLE_X64", "").lower() not in ("1", "true")
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
            or "--xla_cpu_parallel_codegen_split_count=1" not in os.environ.get("XLA_FLAGS", "").split()
            or (os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_CPUS_PER_TASK") != "1")):
        raise ValueError("CPU/x64/one native thread/one codegen/one Slurm CPU required before Python")
    out = result_path(t, mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if validate(out, t, mode) is not None:
            print(f"Complete {mode} task {task_id}; skipped")
            return
        status = RUN / "status" / mode / f"task{task_id}.json"
        started = time.perf_counter()
        env = {k: os.environ.get(k) for k in ("JAX_PLATFORMS", "JAX_ENABLE_X64", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS", "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK")}
        write(status, dict(status="running", task_spec=t, environment=env, hostname=platform.node()))
        try:
            os.environ.update(LANDSCAPE_REPO=str(ROOT), LANDSCAPE_SOURCE=plan["source"], LANDSCAPE_DATA=plan["data"], CHICKEN_WARMSTART_STUDY="small")
            import random_repeat_run as worker
            import jax
            if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
                raise ValueError("CPU/x64 backend required")
            with tempfile.TemporaryDirectory(prefix=f"{mode}{task_id}-", dir=RUN) as temporary:
                staged = Path(temporary) / out.name
                value = worker.run(t["arch"], t["train_ids"], t["training_key"], 2 if mode == "smoke" else t["steps"], t["lr"], t["l2"], staged,
                                   dict(task=t["task"], start=t["start"], N=t["N"], paths=t["paths"], smoke=mode == "smoke"))
                artifacts = []
                for suffix in (".eqx", "_history.pkl"):
                    name = out.stem + suffix
                    target = out.parent / name
                    (Path(temporary) / name).replace(target)
                    artifacts.append(target)
                value.update(mode=mode, task_spec=t, cluster_manifest_sha256=digest(RUN / "manifest.json"),
                             dataset_sha256=DATA_SHA, environment=env, jax_backend=jax.default_backend(), jax_devices=[str(d) for d in jax.devices()],
                             affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                             cluster_wall_seconds=time.perf_counter()-started,
                             max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024),
                             cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                             artifact_sha256={p.name: digest(p) for p in artifacts})
                if mode == "pilot":
                    value["reproduction_difference_K"] = [x-y for x,y in zip(value["member_rmse_K"], t["reference_members_K"])]
                load()
                report = out.with_suffix(".publication.json")
                write(report, value)
                validate(report, t, mode, model_name=out.with_suffix(".eqx").name)
                report.replace(out)
                write(status, dict(status="passed", task_spec=t, result_file=str(out)))
        except BaseException as error:
            write(status, dict(status="failed", task_spec=t, error=f"{type(error).__name__}: {error}"))
            raise


def pending(plan, mode):
    if mode == "full":
        return [i for i, group in enumerate(plan["groups"]) if any(validate(result_path(plan["scientific_design"]["tasks"][t], mode), plan["scientific_design"]["tasks"][t], mode) is None for t in group)]
    tasks = plan["scientific_design"]["tasks"] if mode == "smoke" else plan["reproduction_tasks"]
    ids = plan["smoke_tasks"] if mode == "smoke" else range(6)
    return [i for i in ids if validate(result_path(tasks[i], mode), tasks[i], mode) is None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pending", "run", "check", "progress", "group"))
    parser.add_argument("--mode", choices=("smoke", "pilot", "full", "n2"), default="full")
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
        return
    plan = load()
    if args.action == "run":
        if args.task is None or args.task < 0 or args.task >= (6 if args.mode == "pilot" else 880) or args.mode == "n2":
            parser.error("Valid --task and smoke/pilot/full mode required")
        run(args.task, args.mode)
    elif args.action == "check":
        for mode in ("smoke", "pilot", "n2") if args.mode == "full" else (args.mode,):
            check(plan, mode)
    elif args.action == "pending":
        print(",".join(map(str, pending(plan, args.mode))))
    elif args.action == "group":
        if args.task is None or not 0 <= args.task < len(plan["groups"]):
            parser.error("Valid group --task required")
        print(" ".join(map(str, plan["groups"][args.task])))
    else:
        times = []
        for arch in ("NODE", "sPHNN"):
            for n in (3, 4, 5):
                tasks = [t for t in plan["scientific_design"]["tasks"] if t["arch"] == arch and t["N"] == n]
                completed = [v for t in tasks if (v := validate(result_path(t, "full"), t, "full")) is not None]
                times += [v["cluster_wall_seconds"] for v in completed]
                print(f"{arch} N={n}: {len(completed)}/{len(tasks)} ensembles; {sum(len(v['paths']) for v in completed)}/150 path cells")
        print(f"N=2 index: {'available' if (RUN/'landscape_n2_index.json').exists() else 'not validated yet'}")
        if times:
            print(f"Observed median ensemble walltime: {statistics.median(times)/60:.1f} min (not a queue/finish forecast)")
        for p in sorted((RUN / "status/full").glob("*.json")):
            v = json.loads(p.read_text())
            if v["status"] != "passed":
                print("STATUS", p.name, v["status"], v.get("error", "check sacct/logs for timeout/native exit"))


if __name__ == "__main__":
    main()
