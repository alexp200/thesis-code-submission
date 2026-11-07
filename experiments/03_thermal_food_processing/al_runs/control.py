"""Plan, progress and release checks of the AL study (no JAX needed).

    python control.py prepare [--smoke]     # write the plan (needs data/chicken_data/dataset.nc)
    python control.py progress              # task numbers with model, method, criterion, path and the next phase
    python control.py done --task T         # exit code 0 if task T is complete
    python control.py pending --group G     # unfinished tasks of group G (10 tasks)
    python control.py check                 # validate the smoke tests (cluster launcher)
"""

import argparse
import importlib.metadata
import json
import platform

from design import atomic_json, case, digest, load_plan, plan_path, prepare
from store import read, state


def progress(plan, task):
    """Next phase of a task as (iteration, phase, training ids); checks that the stored manifest belongs to this plan."""
    folder = case(plan, task)
    manifest = read(folder / "manifest.json")
    if manifest is not None and (manifest["plan_sha256"] != digest(plan_path(plan["smoke"])) or manifest["task"] != plan["tasks"][task]):
        raise ValueError("Manifest differs from plan")
    return state(folder, plan, task)


def check_smoke():
    """Release check of the cluster launcher: the two-step smoke tests are complete, agree with the production plan and were run with the current package versions."""
    import numpy as np
    plan = load_plan(True)
    full = load_plan(False) if plan_path(False).exists() else prepare(False)
    for key in ("implementation_sha256", "dataset_sha256", "pool_ids", "evaluation_ids", "acquisition", "reporting_seeds", "random_orders", "warmstart_ids"):
        if plan[key] != full[key]:
            raise ValueError(f"Smoke and production differ: {key}")
    by_model = {}
    for task in [spec["task"] for spec in plan["tasks"] if spec["path"] == 0]:
        i, phase, _ = progress(plan, task)
        if i != 1 or phase != "done":
            raise ValueError(f"Smoke task {task} is incomplete ({phase}); review logs")
        folder = case(plan, task)
        saved_environment = read(folder / "manifest.json")["environment"]
        if saved_environment["python"] != platform.python_version():
            raise ValueError("Python changed after integration smoke")
        for package, version in saved_environment["versions"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Dependency changed after smoke: {package}")
        for package, expected in saved_environment["vcs_packages"].items():
            current = json.loads(importlib.metadata.distribution(package).read_text("direct_url.json") or "null")
            if current != expected:
                raise ValueError(f"Pinned commit changed after smoke: {package}")
        first = read(folder / "iter00.json")
        values = [m["rmse_K"] for m in first["evaluation"]["members"]]
        model = plan["tasks"][task]["model"]
        if model in by_model:
            np.testing.assert_allclose(values, by_model[model], rtol=1e-8, atol=1e-6)
        else:
            by_model[model] = values
        for iteration in (0, 1):
            report = read(folder / f"iter{iteration:02d}.json")
            if report["evaluation"]["all_members"]["valid_count"] != 3:
                raise ValueError("Smoke has failed member evaluation")
        print(f"PASS task={task} {model}/{plan['tasks'][task]['method']}/{plan['tasks'][task]['criterion']}")
    print("All 14 two-step integration smokes passed, including paired N=1 member errors. Not a convergence study.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pending", "done", "progress", "check"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--group", type=int, choices=range(20))
    parser.add_argument("--task", type=int, choices=range(200))
    args = parser.parse_args()
    if args.action == "prepare":
        value = prepare(args.smoke)
        path = plan_path(args.smoke)
        if path.exists() and json.loads(path.read_text()) != value:
            raise SystemExit("Frozen plan differs; archive prior run rather than mixing configurations")
        atomic_json(path, value)
        print(f"Prepared {len(value['tasks'])} paths, max N={max(value['max_N'].values())}, K=3")
        print("Warmstart IDs:", json.dumps(value["warmstart_ids"], sort_keys=True))
    elif args.action == "check":
        check_smoke()
    else:
        plan = load_plan(args.smoke)
        if args.action == "done":
            if args.task is None:
                parser.error("--task required")
            raise SystemExit(0 if progress(plan, args.task)[1] == "done" else 1)
        elif args.action == "pending":
            if args.group is None:
                parser.error("--group required")
            indices = range(args.group*10, args.group*10+10)
            if args.smoke:
                indices = [args.group*10] if args.group < 14 else []
            print(",".join(str(task) for task in indices if progress(plan, task)[1] != "done"))
        else:
            complete = listed = 0
            for spec in plan["tasks"]:
                if args.smoke and spec["path"] != 0:
                    continue
                i, phase, _ = progress(plan, spec["task"])
                n = i+1 if phase in ("done", "acquire") else i
                complete += phase == "done"
                listed += 1
                print(f"task={spec['task']:03d} {spec['model']:5s} {spec['method']:6s} {spec['criterion']:12s} "
                      f"P={spec['path']:02d} reports={n}/{plan['max_N'][str(spec['path'])]} next={phase}")
            print(f"Complete: {complete}/{listed} paths")
