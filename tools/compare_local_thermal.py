"""Compare a path that was run locally (al_runs/run_local.sh) with the stored records of the thesis.

    python tools/compare_local_thermal.py 30     # task number as printed by `python control.py progress` (0-199)

For every training-set size that the local run has completed the script prints the member RMSEs (AP15 test set, K) and the acquired trajectory next to the stored values.
Tasks 140-199 (paths 10-14, the five additional starts) are stored in `small_extra5/run/` as paths 0-4.
Training on a different CPU is not bitwise reproducible: most RMSEs agree to a few percent, a single member can end in a different optimum (up to a factor of about 3 in the paths checked), the first acquisitions normally agree, and picks can differ later.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "experiments/03_thermal_food_processing/al_runs/study"
STORED = ROOT / "runs/03_AL_chicken/warmstart_path_AL"
ORIGINAL_PATHS = 10


def load(path):
    """JSON file as a dict, or None if it does not exist."""
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    """Print the member RMSEs and acquired trajectories of a local path next to the stored records."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", type=int)
    task = parser.parse_args().task
    spec = json.loads((LOCAL / "plan.json").read_text())["tasks"][task]
    additional = spec["path"] >= ORIGINAL_PATHS
    stored_root = STORED / "small_extra5/run" if additional else STORED / "small"
    stored_path = spec["path"] - ORIGINAL_PATHS if additional else spec["path"]
    relative = Path(spec["model"]) / spec["method"] / spec["criterion"]
    local, stored = LOCAL / "results" / relative / f"path{spec['path']:02d}", stored_root / "results" / relative / f"path{stored_path:02d}"
    print(f"task {task}: {spec['model']} / {spec['method']} / {spec['criterion']} / path {spec['path']}")
    print(f"{'n_D':>3}  {'member RMSE local [K]':<28}{'member RMSE stored [K]':<28}{'pick local':>11}{'pick stored':>12}")
    for iteration in range(15):
        report, reference = load(local / f"iter{iteration:02d}.json"), load(stored / f"iter{iteration:02d}.json")
        if report is None:
            break
        members = lambda r: " ".join(f"{m['rmse_K']:6.2f}" for m in r["evaluation"]["members"]) if r else "-"
        pick = lambda folder: (load(folder / f"selections/iter{iteration:02d}.json") or {}).get("picked_id", "-")
        print(f"{iteration + 1:>3}  {members(report):<28}{members(reference):<28}{pick(local):>11}{pick(stored):>12}")


if __name__ == "__main__":
    main()
