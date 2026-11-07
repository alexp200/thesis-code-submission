"""Design of the DED extension: the Random paths 3-5 of the first five starts (package T) and the five further starts with LA, DE, MoLA and six Random paths (G6).
The plan also lists the packages F (start 0, Random paths), C (start 0, second-trajectory candidates) and H2 (extra first-step Random draws); they are not part of the thesis results and have no stored records."""

from collections import Counter
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "experiments/04_ded/al_runs/start_study"
sys.path.append(str(BASE))
import study_design as original

STARTS = [12, 4, 24, 22, 10, 20, 2, 14, 18, 6]
PACKAGES = ("F", "C", "T", "G6", "H2")


def plan():
    """All paths of the extension on top of the plan of the start study."""
    paths = []

    def add(package, model, start, stream, key, end, repeat=None, candidate=None, index=None):
        paths.append({
            "id": f"{package}/{model}/start{start:02d}/{stream}", "package": package,
            "model_class": model, "start": start, "stream": stream,
            "reporting_key_base": key,
            "acquisition_key_base": None if index is None else 9500000 + 100000 * index,
            "random_tail": original.tail(start, repeat) if repeat is not None else None,
            "repeat": repeat, "candidate": candidate, "max_N": end,
            "reuse_n1": package in ("F", "C", "T", "H2"), "reuse_existing": package == "C",
        })

    for model in original.MODELS:
        for r in range(6):
            add("F", model, 0, f"random{r}", 5000000 + 100000 * r, 5, repeat=r)
        for candidate in range(1, 25):
            add("C", model, 0, f"candidate{candidate:02d}", 5000000, 2, candidate=candidate)
        for i, start in enumerate(STARTS):
            key = 9000000 + 100000 * i
            if i < 5:
                for r in range(3, 6):
                    add("T", model, start, f"random{r}", key, 5, repeat=r, index=i)
            else:
                for stream in ("LA", "DE", "MoLA"):
                    add("G6", model, start, stream, key, 5, index=i)
                for r in range(6):
                    add("G6", model, start, f"random{r}", key, 5, repeat=r, index=i)
            for r in range(6, 10):
                add("H2", model, start, f"random{r}", key, 2, repeat=r, index=i)
    value = original.plan("main")
    value.update(schema_version=2, part="final", starts=STARTS, paths=paths,
                 packages=list(PACKAGES), max_N=5, min_N=1,
                 cohort_note="ten new starts N1..5; start0 separate paired reference; N5 extension approved 2026-10-21",
                 H2_sampling="ten IID seeded permutation draws, not ten distinct second IDs",
                 G1="reuse verified original gate; do not rerun reproduction training")
    return value


def active(path, N):
    """Whether the path has a record at size N."""
    return N <= path["max_N"]


def static_requests(value):
    """Training requests that do not depend on earlier trained models (Random prefixes, candidates, the first size of every path)."""
    requests = {}
    for path in value["paths"]:
        for N in range(1, path["max_N"] + 1):
            if path["stream"] in ("LA", "DE", "MoLA") and N > 1:
                continue
            ids = ([path["start"], path["candidate"]] if N == 2 and path["candidate"] is not None
                   else [path["start"], *(path["random_tail"] or [])[:N-1]])
            request = original.training_request(path, ids, N)
            requests[request["id"]] = request
    # Seed AL early; larger-N independent jobs subsequently fill otherwise idle slots.
    return sorted(requests.values(), key=lambda r: (r["dataset_size"], r["start"] not in STARTS[5:],
                                                   r["model_class"], r["id"]))


def counts(value):
    """Number of paths and unique trainings per package."""
    report = {}
    all_known = {}
    for package in PACKAGES:
        paths = [p for p in value["paths"] if p["package"] == package]
        unique = {}
        for path in paths:
            for N in range(1, path["max_N"] + 1):
                if path["stream"] in ("LA", "DE", "MoLA") and N > 1:
                    continue
                ids = ([path["start"], path["candidate"]] if N == 2 and path["candidate"] is not None
                       else [path["start"], *(path["random_tail"] or [])[:N-1]])
                request = original.training_request(path, ids, N)
                unique[request["id"]] = request
                all_known[request["id"]] = request
        report[package] = {"paths_or_draws": len(paths),
                           "published_cells_including_referenced_N1": sum(p["max_N"] for p in paths),
                           "deterministic_unique_requests_by_N": dict(sorted(Counter(
                               r["dataset_size"] for r in unique.values()).items())),
                           "AL_N2plus_requests": "selection-dependent" if package == "G6" else 0}
    report["all"] = {"deterministic_unique_requests": len(all_known),
                     "AL_N2plus_requests": "unknown until acquisition; no false exact forecast"}
    return report


def sources():
    """Code files whose hashes are frozen with the plan."""
    files = original.sources() + list(HERE.glob("*.py")) + list(HERE.glob("*.sbatch")) + list(HERE.glob("*.sh"))
    return sorted(set(files))
