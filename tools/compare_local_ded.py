"""Compare a DED path that was run locally (experiments/04_ded/al_runs/start_study/run_local.sh) with the stored records of the thesis.

    python tools/compare_local_ded.py NODE/start12/LA

For every training-set size that the local run has completed the script prints the three member RMSEs (all 25 simulations, K) and the trajectory that was added to the training set
next to the stored values. Training on a different CPU is not bitwise reproducible, and single members of the DED ensembles can differ by a large factor; picks can differ as well.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNS = ROOT / "experiments/04_ded/al_runs/start_study/run"
STORED = ROOT / "runs/ded_start_study/run/main/paths"          # ten-start study; the five further starts are in runs/ded_final_extension/run/paths/<package>/...
STORED_EXTENSION = ROOT / "runs/ded_final_extension/run/paths"


def load(path):
    """JSON file as a dict, or None if it does not exist."""
    return json.loads(path.read_text()) if path.is_file() else None


def stored_folder(path_id):
    """Folder of a path in the stored records of the start study or of the extension."""
    if (STORED / path_id).is_dir():
        return STORED / path_id
    for package in sorted(STORED_EXTENSION.glob("*")):
        if (package / path_id).is_dir():
            return package / path_id
    return STORED / path_id


def main():
    """Print the member RMSEs and added trajectories of a local path next to the stored records."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path_id", help="model/startNN/stream, e.g. NODE/start12/LA")
    parser.add_argument("--run", type=Path, help="local run folder (default: the newest run/main_local_* that contains the path)")
    arguments = parser.parse_args()
    runs = [arguments.run] if arguments.run else sorted(LOCAL_RUNS.glob("main_local*"), key=lambda p: p.stat().st_mtime, reverse=True)
    local = next((r / "paths" / arguments.path_id for r in runs if (r / "paths" / arguments.path_id).is_dir()), None)
    if local is None:
        raise SystemExit(f"No local results for {arguments.path_id} in {LOCAL_RUNS}")
    stored = stored_folder(arguments.path_id)
    members = lambda r: " ".join(f"{m['rmse_K']:8.2f}" for m in r["evaluation"]["members"]) if r else "-"
    added = lambda r: (r["train_ids"][-1] if len(r["train_ids"]) > 1 else r["train_ids"][0]) if r else "-"
    print(f"{arguments.path_id}   (local: {local.parents[3].name})")
    print(f"{'n_D':>3}  {'member RMSE local [K]':<30}{'member RMSE stored [K]':<30}{'added local':>12}{'added stored':>13}")
    for iteration in range(5):
        report, reference = load(local / f"iter{iteration:02d}.json"), load(stored / f"iter{iteration:02d}.json")
        if report is None:
            break
        print(f"{iteration + 1:>3}  {members(report):<30}{members(reference):<30}{added(report):>12}{added(reference):>13}")


if __name__ == "__main__":
    main()
