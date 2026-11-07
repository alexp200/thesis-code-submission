"""Design of the DED start study: starts, path ids, training keys, Random tails, training requests, the plan, and the code files whose hashes are frozen with it (no JAX import)."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
# ------------------------------------------------------------------------------------------------ constants and local mode
STARTS = [12, 4, 24, 22, 10]
EXTENSION_STARTS = [20, 2, 14, 18, 6]      # added by ../final_extension; only part of the plan in DED_LOCAL mode
MODELS = ["NODE", "sPHNN"]
PACKAGES = ("jax", "jaxlib", "dynax", "klax", "diffrax", "equinox", "optax")
DATASET = ROOT / "data/directed_energy_deposition/max_simulation/dataset.nc"
OLD = ROOT / "experiments/shared/al_path_repeats/results_p10_k3/DED"
# DED_LOCAL=1 (set by run_local.sh): one machine without Slurm. The release gate, the cluster CPU checks and the reuse of earlier
# runs are switched off; DED_LOCAL_PATHS (comma separated path ids such as NODE/start12/LA) restricts the plan to these paths and
# the run folder is named after the selection. The local plan also contains the five further starts and the Random paths 3-5 of
# ../final_extension (same keys and request ids), so that every DED path of the thesis can be selected. The computational code is the same.
LOCAL = os.environ.get("DED_LOCAL") == "1"
SELECTION = sorted(s for s in os.environ.get("DED_LOCAL_PATHS", "").split(",") if s) if LOCAL else []
RUN_TAG = ("_local" + ("_" + hashlib.sha256(",".join(SELECTION).encode()).hexdigest()[:8] if SELECTION else "")) if LOCAL else ""
STORED_SHARED = [ROOT / "runs/ded_start_study/run/main/shared", ROOT / "runs/ded_final_extension/run/shared"]      # stored ensembles of the thesis runs (POD bases: large_assets/ded_pod_bases.tar.gz)
PRIMARY = ROOT / "experiments/04_ded/kernels"
MOLA = ROOT / "experiments/04_ded/kernels/mola_acquisition"      # executor of the MoLA components


# ------------------------------------------------------------------------------------------------ hashes and training sets
def digest(path):
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    """SHA-256 of the canonical JSON of a value (identity of a plan or a request)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical(ids):
    """Training ids in training order: the start first, then the others in increasing order."""
    if not ids or len(ids) != len(set(ids)) or any(i not in range(25) for i in ids):
        raise ValueError("Invalid DED training IDs")
    return [ids[0], *sorted(ids[1:])]


def tail(start, repeat):
    """Random order of the other 24 simulations for a start and a Random path r."""
    return np.random.default_rng([20264001, start, repeat]).permutation(
        [i for i in range(25) if i != start]).tolist()


def recipe(model):
    """Training recipe of an architecture (must equal the schedule in kernels/core/al_loop_ded.py)."""
    return {"procedure": "derivative_then_trajectory_finetune", "derivative_steps": 20000,
            "derivative_lr": 1e-3, "finetune_steps": 15000 if model == "NODE" else 10000,
            "finetune_lr": 1e-4 if model == "NODE" else 1e-5, "derivative_l2": 0.,
            "finetune_l2": 1e-5 if model == "NODE" else 1e-6,
            "finetune_objective": "MSE + 0.5 * finetune_l2 * ||theta_laplace||^2"}


# ------------------------------------------------------------------------------------------------ plan
def plan(part):
    """The frozen plan of a part: 'main' (the start study of the thesis); 'gate' (release gate against an earlier run), 'B' and 'C' belong to the cluster procedure."""
    paths = []
    starts = (STARTS + EXTENSION_STARTS if LOCAL else STARTS) if part == "main" else [0]
    for i, start in enumerate(starts):
        for model in MODELS:
            streams = ["LA", "DE", "MoLA", "random0", "random1", "random2"] + (["random3", "random4", "random5"] if LOCAL else [])
            if part == "gate":
                streams = ["LA"]
            elif part == "B":
                streams = ["random0", "random1", "random2"]
            elif part == "C":
                streams = [f"candidate{c:02d}" for c in range(1, 25)]
            for stream in streams:
                r = int(stream[-1]) if stream.startswith("random") else 0
                key = 9000000 + 100000 * i if part == "main" else 5000000
                if part == "B":
                    key += 100000 * r
                paths.append({"id": f"{model}/start{start:02d}/{stream}", "model_class": model,
                              "start": start, "stream": stream, "reporting_key_base": key,
                              "acquisition_key_base": 9500000 + 100000 * i if part == "main" else 1000000,
                              "random_tail": tail(start, r) if stream.startswith("random") else None,
                              "candidate": int(stream[9:]) if part == "C" else None,
                              "reuse_n1": part == "B", "reuse_existing": part == "C"})
    if SELECTION:
        unknown = set(SELECTION) - {p["id"] for p in paths}
        if unknown:
            raise ValueError(f"Unknown path ids: {sorted(unknown)}")
        paths = [p for p in paths if p["id"] in SELECTION]
    return {"schema_version": 1, "part": part, "starts": starts, "max_N": 2 if part in ("gate", "C") else 5,
            "min_N": 2 if part == "C" else 1, "reporting_ensemble_size": 3,
            "LA_acquisition_size": 1, "MoLA_indices": [0, 1, 2], "criterion": "sum_marginal",
            "ggn_backend": "exact_production_reverse_mode", "score_backend": "exact_eigen_jvp",
            "direction_chunk_size": 128, "sigma_noise": 0., "de_variance_ddof": 0,
            "evaluation_ids": list(range(25)), "evaluation_scope": "all25_including_training_not_holdout",
            "data_order": "start followed by sorted other IDs", "random_tail_seed": 20264001,
            "training_recipes": {m: recipe(m) for m in MODELS}, "paths": paths}


def training_request(path, ids, N, acquisition=False):
    """Request for one training (reporting ensemble of three members, or the single-member LA acquisition model); identical requests are trained once and shared."""
    if len(ids) != N:
        raise ValueError("Training-set size does not match N")
    value = {"model_class": path["model_class"], "start": path["start"], "train_ids": canonical(ids),
             "training_key": (path["acquisition_key_base"] if acquisition else path["reporting_key_base"]) + 1000 * (N - 1),
             "ensemble_size": 1 if acquisition else 3, "role": "acquisition" if acquisition else "reporting"}
    return dict(value, id=identity(value), dataset_size=N)


def sources():
    """Code files whose SHA-256 values are frozen in the manifest of a run."""
    files = list(HERE.glob("*.py")) + list(HERE.glob("*.sh")) + list(HERE.glob("*.sbatch"))
    files += list((ROOT / "src").rglob("*.py"))
    files += list((ROOT / "experiments/shared/al_path_repeats").glob("repeat_*.py"))
    files += [ROOT / "experiments/shared/cluster_recovery" / name for name in
              ("runtime.py", "pod_cache.py", "run_cpu.sh")]
    files += [PRIMARY / "core" / name for name in
              ("_ded_al_core.py", "al_loop_ded.py", "ded_data.py", "ded_split.py")]
    files += [PRIMARY / "la_de_runner" / name for name in ("run_al.py", "reporting.py")]
    files += list(MOLA.glob("*.py"))
    return sorted(set(files))
