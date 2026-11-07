"""Design of the thermal food processing AL study: tasks, seeds, starts, the frozen plan (with SHA-256 of data and code) and its verification."""

import hashlib
import json
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
HERE = CODE / "study"            # plan.json and the results of local runs
DATA = ROOT / "data" / "chicken_data"
PRIMARY = ROOT / "experiments/03_thermal_food_processing/kernels/core"
K = 3                            # ensemble members
ORIGINAL_PATHS, MAX_N = 10, 15   # ten original starts (paths 0-9), training-set sizes n_D = 1..15
EXTENSION_STARTS, MAX_N_EXTENSION = [745, 795, 861, 873, 905], 5   # five additional starts (paths 10-14), n_D = 1..5
PATHS = ORIGINAL_PATHS + len(EXTENSION_STARTS)
CRITERIA = ("sum_marginal", "sum_variance")
SMALL_POOL = [745, 795, 835, 851, 853, 861, 868, 873, 874, 888, 899, 902, 905, 906, 917]
SMALL_TEST = [313, 320, 344, 378, 383, 407, 412, 415, 461, 462, 466, 467, 474, 508, 528]
WARMSTART_SEED = 20261012


# ------------------------------------------------------------------------------------------------ files
def digest(path):
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    """Write a JSON file so that it appears completely or not at all."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


# ------------------------------------------------------------------------------------------------ design
def tasks():
    """The 200 tasks. 0-139: 2 architectures x 7 methods (Random; DE, LA, MoLA x MGE, VAR) x 10 original starts (paths 0-9).
    140-199: 2 architectures x 6 methods (DE, LA, MoLA x MGE, VAR; no Random) x 5 additional starts (paths 10-14)."""
    strategies = [(m, c) for m in ("DE", "LA", "MoLA") for c in CRITERIA]
    entries = []
    for model in ("NODE", "sPHNN"):
        for method, criterion in [("random", "random"), *strategies]:
            for path in range(ORIGINAL_PATHS):
                entries.append(dict(task=len(entries), model=model, method=method,
                                    criterion=criterion, path=path))
    for model in ("NODE", "sPHNN"):
        for method, criterion in strategies:
            for path in range(ORIGINAL_PATHS, PATHS):
                entries.append(dict(task=len(entries), model=model, method=method,
                                    criterion=criterion, path=path))
    return entries


def max_n(path):
    """Largest training-set size n_D of a path."""
    return MAX_N if path < ORIGINAL_PATHS else MAX_N_EXTENSION


def seeds(path, iteration):
    """Initialisation seeds of the K members of one path at one training-set size."""
    return [7_000_000 + 100_000 * path + 1000 * iteration + k for k in range(K)]


def warmstarts(pool):
    """Start trajectories: the ten original starts drawn without replacement from the selection pool (paths 0-9), then the five remaining trajectories of the pool (paths 10-14)."""
    import numpy as np
    draws = np.random.default_rng(WARMSTART_SEED).choice(sorted(pool), size=ORIGINAL_PATHS, replace=False)
    starts = [int(sid) for sid in draws] + EXTENSION_STARTS
    if sorted(starts) != sorted(pool):
        raise ValueError("The 15 starts must be the 15 trajectories of the selection pool")
    return {str(p): sid for p, sid in enumerate(starts)}


def random_order(pool, path, start):
    """Order in which the Random baseline of one path adds trajectories (the start comes first)."""
    import numpy as np
    remaining = sorted(sid for sid in pool if sid != start)
    return [start, *np.random.default_rng(20262012 + path).permutation(remaining).tolist()]


def source_paths():
    """Code files whose hashes are frozen in the plan."""
    paths = list(CODE.glob("*.py")) + list(CODE.glob("*.sh")) + list(CODE.glob("*.sbatch"))
    paths += list((ROOT / "src").rglob("*.py"))
    paths += [PRIMARY / "train_common.py", PRIMARY / "chicken_data.py",
              ROOT / "experiments/shared/cluster_recovery/run_cpu.sh"]
    return sorted(set(paths))


# ------------------------------------------------------------------------------------------------ data set and plan
def validate_dataset():
    """Check the 30-trajectory data set; returns selection pool, evaluation set (AP15), evaluation subsets and the bin edges of the surface median."""
    import numpy as np
    import xarray as xr
    with xr.open_dataset(DATA / "dataset.nc", engine="h5netcdf") as ds:
        ids = ds.simulation_id.values.astype(int).tolist()
        ys, us = ds.temperature.values, ds.oven_temperature.values
        n = 30
        if (len(ids) != n or len(set(ids)) != n or ys.shape != (n, 280, 2)
                or us.shape != (n, 280) or ds.probe_point.values.tolist() != ["A", "B"]):
            raise ValueError("Unexpected dataset dimensions/probe order")
        np.testing.assert_array_equal(ds.time.values, np.arange(280)*5.)
        np.testing.assert_allclose(ys[:, 0], 279.15, atol=1e-8, rtol=0)
        np.testing.assert_allclose(us[:, 0], 279.15, atol=1e-8, rtol=0)
        if not np.all(np.isfinite(ys)) or not np.all(np.isfinite(us)):
            raise ValueError("Nonfinite data")
        fingerprints = {hashlib.sha256(u.tobytes()+y.tobytes()).hexdigest() for u, y in zip(us, ys)}
        if len(fingerprints) != n:
            raise ValueError("Duplicate trajectory")
        if set(ids) != set(SMALL_POOL + SMALL_TEST) or set(SMALL_POOL) & set(SMALL_TEST):
            raise ValueError("Small-pool split changed")
        pool, evaluation = SMALL_POOL, SMALL_TEST
        ei = [ids.index(sid) for sid in evaluation]
        medians = np.median(ys[ei, :, 1], axis=1)
        edges = np.linspace(medians.min(), medians.max(), 7)
        bins = np.minimum(np.searchsorted(edges, medians, side="right")-1, 5)
        subsets = {f"surface_median_bin{i}": [sid for sid, b in zip(evaluation, bins) if b == i] for i in range(6)}
        subsets["AP15"] = SMALL_TEST
    return list(pool), list(evaluation), subsets, edges.tolist()


def prepare(smoke=False):
    """Build the plan: design, seeds, starts, random orders, recipes, acquisition settings and the SHA-256 of data and code."""
    pool, evaluation, subsets, edges = validate_dataset()
    starts = warmstarts(pool)
    return dict(schema_version=3, smoke=smoke, tasks=tasks(), paths=PATHS, ensemble_size=K,
                max_N={str(p): 2 if smoke else max_n(p) for p in range(PATHS)},
                warmstart_ids=starts, warmstart_selection_seed=WARMSTART_SEED,
                warmstart_policy="paths 0-9: drawn uniformly without replacement from the selection pool; paths 10-14: the five remaining trajectories of the pool",
                pool_ids=pool, evaluation_ids=evaluation,
                evaluation_scope="fixed AP15 holdout, never selected",
                subset_ids=subsets, surface_median_bin_edges_K=edges,
                reporting_seeds={str(p): [seeds(p, i) for i in range(max_n(p))] for p in range(PATHS)},
                random_orders={str(p): random_order(pool, p, starts[str(p)])[:MAX_N] for p in range(ORIGINAL_PATHS)},
                recipes={m: dict(steps=2 if smoke else (30000 if m == "NODE" else 20000), lr=1e-4, l2=1e-6,
                                 procedure="trajectory", augmented_dimensions=0, vmap_ensemble=True, full_batch=True)
                         for m in ("NODE", "sPHNN")},
                acquisition=dict(LA_member_index=0, MoLA_member_indices=[0, 1, 2],
                                 posterior="exact production reverse GGN/MLL, full spectrum JVP",
                                 direction_chunk=128, candidate_batch=4, evaluation_batch=8, cache_block=128,
                                 noise=0., filter="training latent RMSE < 5 * median; no test-based filtering"),
                dataset_sha256=digest(DATA / "dataset.nc"),
                implementation_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths()})


def plan_path(smoke):
    """Location of the plan file (the smoke plan has two training-set sizes)."""
    return HERE / ("plan_smoke.json" if smoke else "plan.json")


def load_plan(smoke):
    """Read the plan and verify data and code against its hashes."""
    spec = json.loads(plan_path(smoke).read_text())
    verify(spec)
    return spec


def verify(plan):
    """Raise if the data set or any hashed code file differs from the plan."""
    if digest(DATA / "dataset.nc") != plan["dataset_sha256"]:
        raise ValueError("Frozen dataset changed")
    for name, expected in plan["implementation_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def case(plan, task):
    """Result folder of one task."""
    spec = plan["tasks"][task]
    folder = HERE / ("smoke_results" if plan["smoke"] else "results")
    return folder / spec["model"] / spec["method"] / spec["criterion"] / f"path{spec['path']:02d}"
