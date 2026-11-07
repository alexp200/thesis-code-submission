"""Builds the provenance metadata of the FINAL DED comparison (ten starts, n_D = 1..5, six Random paths per start, MGE only) for the code submission. Read-only on all records; no training, no new random draws;
recorded keys / seeds are copied, recipe values are only CHECKED against them (never substituted).

Final cohort (key-index order i = 0..9): [12, 4, 24, 22, 10, 20, 2, 14, 18, 6] =
  package A  (start_study/run/main)          starts 12, 4, 24, 22, 10 (i = 0..4): LA / DE / MoLA (MGE) and Random 0..2
  package T  (final_extension/run)           the same five starts: Random 3..5
  package G6 (final_extension/run)           starts 20, 2, 14, 18, 6 (i = 5..9): LA / DE / MoLA and Random 0..5
NOT part of the final figures: start 0 (package F, C), the H2 first-step Random draws 6..9 and the candidate-only pilot studies.
Keys: reporting key = 9000000 + 100000 i + 1000 (n_D - 1); LA acquisition key = 9500000 + 100000 i + 1000 (n_D - 1); Random tail for start s and r: np.random.default_rng([20264001, s, r]).permutation([j for j in range(25) if j != s]).
Member seeds are NOT key + k: they are derived by the recorded JAX key splitting (la_de_runner/run_al.py::member_seeds, reached through repeat_adapter); the recorded `initialization_seeds` are copied and re-derived as a cross-check only.
Outputs (this folder): seed_manifest_paths.csv, seed_manifest_acquisition_models.csv, seed_manifest.json, plans/, random_tails.json, environment_recorded.json, inspected_settings.json.
Usage: python build_submission_metadata.py"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)        # the cluster runs used float64 / x64 (randint then draws int64: the recorded seeds are only reproduced in x64 mode)

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("SUBMISSION_METADATA_OUT", HERE))        # submission: outputs can be redirected (verification)
FE = HERE.parents[1]
REPO = FE.parent
CB = REPO                                                               # submission: the sources inspected for hashing are the files of the submission
RUN_A = REPO / "runs/ded_start_study/run/main"
RUN_X = REPO / "runs/ded_final_extension/run"
rel = lambda p: str(Path(p).relative_to(REPO))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
STARTS = [12, 4, 24, 22, 10, 20, 2, 14, 18, 6]
MODELS = ("NODE", "sPHNN")
SELECT = [("A", RUN_A / "paths", STARTS[:5], ["LA", "DE", "MoLA", "random0", "random1", "random2"]),
          ("T", RUN_X / "paths/T", STARTS[:5], ["random3", "random4", "random5"]),
          ("G6", RUN_X / "paths/G6", STARTS[5:], ["LA", "DE", "MoLA"] + [f"random{r}" for r in range(6)])]


def write_csv(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def resolve(base_file, relative):
    return (Path(base_file).parent / relative).resolve()


def member_seeds(key_value, size):
    """the recorded derivation (la_de_runner/run_al.py::member_seeds), used for a cross-check of the recorded seeds only"""
    import jax.random as jrd
    stage1_key, _ = jrd.split(jrd.key(key_value))
    ensemble_key, _ = jrd.split(stage1_key)
    return [int(s) for s in jrd.randint(ensemble_key, (size,), 0, 2**31 - 1)]


def main():
    summary = {}
    rows, acq_rows = [], []
    for pkg, root, starts, streams in SELECT:
        for model in MODELS:
            for s in starts:
                i = STARTS.index(s)
                for stream in streams:
                    for it in range(5):
                        f = root / model / f"start{s:02d}" / stream / f"iter{it:02d}.json"
                        d = json.loads(f.read_text())
                        n = d["dataset_size"]
                        rk = 9_000_000 + 100_000 * i + 1000 * (n - 1)
                        req = d["request"]
                        rec_ok = d["training_key"] == rk == req["training_key"]
                        seeds_ok = member_seeds(d["training_key"], 3) == d["initialization_seeds"]
                        art = d["artifacts"]
                        method = "Random" if stream.startswith("random") else stream
                        row = dict(package=pkg, key_index=i, start_id=s, arch=model, method=method, stream=stream, n_D=n,
                                   train_ids_acquisition_order=" ".join(map(str, d["train_ids"])), train_ids_training_order=" ".join(map(str, d["canonical_train_ids"])), picked_id=d.get("picked_id", ""),
                                   role="reporting (3 members)", shared_node_id=req["id"], recorded_training_key=d["training_key"], recipe_training_key=rk, recipe_key_match=rec_ok,
                                   recorded_initialization_seeds=" ".join(map(str, d["initialization_seeds"])), seeds_rederived_match=seeds_ok, effective_members=d["effective_ensemble_size"], dropped_indices=" ".join(map(str, d["dropped_indices"])),
                                   checkpoint=rel(resolve(f, art["checkpoint"]["path"])), checkpoint_sha256=art["checkpoint"]["sha256"], pod_basis=rel(resolve(f, art["pod"]["path"])), pod_sha256=art["pod"]["sha256"],
                                   training_record=rel(resolve(f, d["shared_training"]["path"])), training_record_sha256=d["shared_training"]["sha256"],
                                   host=d["execution"]["hostname"], slurm_job_id=d["execution"]["slurm_job_id"], timestamp_utc=d["execution"]["timestamp_utc"], train_seconds=round(d["train_seconds"], 1), run_manifest_sha256=d["manifest_sha256"], record=rel(f))
                        if stream == "LA":
                            a = d.get("acquisition")
                            if a:                                                                                   # iter k contains the selection that added `picked_id` (made with the LA model of size k)
                                sel = resolve(f, a["cache"]["path"])
                                sj = json.loads(sel.read_text())
                                tf = resolve(sel, sj["model_commit"]["path"])
                                tj = json.loads(tf.read_text())
                                m = tj["dataset_size"]
                                ak = 9_500_000 + 100_000 * i + 1000 * (m - 1)
                                acq_rows.append(dict(package=pkg, key_index=i, start_id=s, arch=model, selection_in_record_n_D=n, model_n_D=m, picked_id=sj["picked_id"], train_ids=" ".join(map(str, tj["train_ids"])),
                                                     role="acquisition (1 member, LA)", shared_node_id=sj["request"]["model_id"], recorded_training_key=tj["training_key"], recipe_training_key=ak, recipe_key_match=(tj["training_key"] == ak),
                                                     recorded_initialization_seeds=" ".join(map(str, tj["initialization_seeds"])), seeds_rederived_match=(member_seeds(tj["training_key"], 1) == tj["initialization_seeds"]),
                                                     is_reporting_ensemble_member="no (separate single model with its own key; shares the POD basis of the reporting node of the same training set)",
                                                     pod_sha256_acquisition_model=tj["pod_sha256"], pod_sha256_reporting_node=art["pod"]["sha256"], same_pod=(tj["pod_sha256"] == art["pod"]["sha256"]) if m == n else "n/a (different training-set size)",
                                                     checkpoint=rel(tf.parent / "model.eqx"), selection_cache=rel(sel), selection_cache_sha256=a["cache"]["sha256"], training_record=rel(tf), laplace_fit_seconds=sj.get("timings", {}).get("laplace_fit_seconds", "")))
                        rows.append(row)
    ids = Counter(r["shared_node_id"] for r in rows)
    for r in rows:
        r["records_sharing_this_node"] = ids[r["shared_node_id"]]
    write_csv("seed_manifest_paths.csv", rows)
    write_csv("seed_manifest_acquisition_models.csv", acq_rows)
    summary.update(path_records=len(rows), unique_reporting_nodes=len(ids), nodes_read_by_several_path_occurrences=sum(1 for v in ids.values() if v > 1), reporting_recipe_key_mismatches=[r["record"] for r in rows if not r["recipe_key_match"]],
                   reporting_seed_rederivation_mismatches=sum(not r["seeds_rederived_match"] for r in rows), acquisition_model_records=len(acq_rows), acquisition_recipe_key_mismatches=sum(not r["recipe_key_match"] for r in acq_rows),
                   acquisition_seed_rederivation_mismatches=sum(not r["seeds_rederived_match"] for r in acq_rows), local_jax_version=jax.__version__, recorded_jax_version="0.10.1")

    kc = 9_500_000                                                                                   # reporting key of i = 5, n_D = 1 and acquisition key of i = 0, n_D = 1
    summary["key_collision_example"] = dict(key=kc, reporting_member_seeds=member_seeds(kc, 3), acquisition_member_seeds=member_seeds(kc, 1))
    # ---- Random tails: plan versus recomputation
    plan_a, plan_x = json.loads((RUN_A / "plan.json").read_text()), json.loads((RUN_X / "plan.json").read_text())
    tails, ok = {}, True
    for p in plan_a["paths"] + plan_x["paths"]:
        if p["start"] in STARTS and p["stream"].startswith("random") and p.get("package", "A") in ("A", "T", "G6") and p["random_tail"] is not None:
            s, r = p["start"], int(p["stream"][6:])
            re = np.random.default_rng([20264001, s, r]).permutation([j for j in range(25) if j != s]).tolist()
            ok &= (re == p["random_tail"])
            tails[f"{p.get('package', 'A')}/{p['model_class']}/start{s}/random{r}"] = dict(package=p.get("package", "A"), start=s, r=r, plan_tail=p["random_tail"], recomputed_tail=re)
    by_sr = {}
    for k, v in tails.items():
        by_sr.setdefault((v["start"], v["r"]), []).append(v["plan_tail"])
    shared_ok = all(all(t == ts[0] for t in ts) for ts in by_sr.values())
    (OUT / "random_tails.json").write_text(json.dumps(dict(rng="np.random.default_rng([20264001, s, r]).permutation([j for j in range(25) if j != s]); training sets are prefixes, training order = start followed by the other selected IDs in increasing order; "
                                                              "draws shared across architectures", entries=tails), indent=1))
    summary.update(random_tails_in_plans=len(tails), random_tails_match_recomputation=bool(ok), random_tails_identical_across_architectures=bool(shared_ok), distinct_start_r_pairs=len(by_sr))

    # ---- plans, manifests, environment
    out = OUT / "plans"
    out.mkdir(exist_ok=True)
    copies = {"start_study_main_plan.json": RUN_A / "plan.json", "start_study_main_manifest.json": RUN_A / "manifest.json", "final_extension_plan.json": RUN_X / "plan.json", "final_extension_manifest.json": RUN_X / "manifest.json"}
    summary["plan_copies"] = {}
    for n, src in copies.items():
        shutil.copyfile(src, out / n)
        summary["plan_copies"][n] = dict(source=rel(src), sha256=sha(src))
    ma, mx = json.loads((RUN_A / "manifest.json").read_text()), json.loads((RUN_X / "manifest.json").read_text())
    (OUT / "environment_recorded.json").write_text(json.dumps(dict(
        start_study_main=dict(dataset_sha256=ma["dataset_sha256"], plan_sha256=ma["plan_sha256"], environment=ma["environment"], git_commit=ma["git_commit"] or "unknown (recorded empty)", implementation_sha256=ma["implementation_sha256"]),
        final_extension=dict(dataset_sha256=mx["dataset_sha256"], plan_sha256=mx["plan_sha256"], environment=mx["environment"], git_commit=mx["git_commit"] or "unknown (recorded empty)", implementation_sha256=mx["implementation_sha256"],
                             original_main_manifest_sha256=mx["original_main_manifest_sha256"]),
        note="Recorded at launch of the cluster runs. Current package versions are not substituted."), indent=1))
    summary["dataset_sha256_equal"] = ma["dataset_sha256"] == mx["dataset_sha256"]

    # ---- inspected sources and settings with local file hashes
    files = ["experiments/04_ded/al_runs/start_study/study_design.py", "experiments/04_ded/al_runs/final_extension/design.py", "experiments/04_ded/al_runs/start_study/study_numerical.py", "runs/ded_start_study/run/main/manifest.json",
             "experiments/shared/al_path_repeats/repeat_adapter.py", "experiments/04_ded/kernels/la_de_runner/run_al.py", "experiments/04_ded/kernels/core/_ded_al_core.py",
             "experiments/04_ded/kernels/core/ded_data.py", "src/models.py", "src/regularization.py", "src/active_learning.py"]
    (OUT / "inspected_settings.json").write_text(json.dumps(dict(
        active_parameter_counts=dict(NODE=2248, sPHNN=6425, note="source models constructed without training"),
        solver=dict(method="Tsit5", step_size_controller="PID", atol=1e-6, rtol=1e-6, max_steps=262144), dynax_commit_recorded="4cd21efe2675f9a875f6a628493ad7bcc7973f37",
        training=dict(NODE=dict(derivative_steps=20000, derivative_lr=1e-3, finetune_steps=15000, finetune_lr=1e-4, finetune_l2=1e-5), sPHNN=dict(derivative_steps=20000, derivative_lr=1e-3, finetune_steps=10000, finetune_lr=1e-5, finetune_l2=1e-6), derivative_batch=128,
                      note="two training stages with optimiser keys / state reset; the shared POD is used by a reporting node and by the separate LA model of the same training set"),
        inspected_sources_relative_to_submission_root={f: dict(exists=(CB / f).exists(), sha256=sha(CB / f) if (CB / f).exists() else None) for f in files}), indent=1))

    # ---- cohort check against the tidy loader used by the figures and tables
    sys.path.insert(0, str(HERE.parent / "figures"))
    spec = importlib.util.spec_from_file_location("ded_mrf_meta", HERE.parent / "figures/make_results_figures.py")
    mrf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mrf)
    d = mrf.IO.load_new()
    used = d[d.pkg.isin(["A", "T", "G6"]) & d.start.isin(STARTS)]
    s_loader = {(r.model, r.start, r.stream, r.N) for r in used.itertuples()}
    s_manifest = {(r["arch"], r["start_id"], r["stream"], r["n_D"]) for r in rows}
    PS, runs = mrf.cohort(d, 5)
    summary.update(loader_records_final_cohort=len(s_loader), manifest_records=len(s_manifest), manifest_equals_loader_set=(s_loader == s_manifest), only_in_loader=len(s_loader - s_manifest), only_in_manifest=len(s_manifest - s_loader),
                   figure_cohort_starts=sorted(int(x) for x in PS.start.unique()), figure_cohort_rows_AL=int(len(PS)), figure_cohort_random_rows=int(len(runs)), excluded_packages=sorted(set(d.pkg.unique()) - {"A", "T", "G6"}))
    (OUT / "seed_manifest.json").write_text(json.dumps(dict(constants=dict(starts_key_index_order=STARTS, packages={"A": "start study main: starts 12,4,24,22,10; LA/DE/MoLA, Random 0..2", "T": "final extension: Random 3..5 of the same starts", "G6": "final extension: starts 20,2,14,18,6, LA/DE/MoLA, Random 0..5"},
                                                           reporting_key="9000000 + 100000*i + 1000*(n_D-1)", la_acquisition_key="9500000 + 100000*i + 1000*(n_D-1)", random_tail="np.random.default_rng([20264001, s, r]).permutation([j for j in range(25) if j != s])",
                                                           key_collision_note="the numeric reporting keys of key indices i >= 5 (9500000 + ...) fall into the numeric range of the acquisition keys of i = 0 (9500000 + ...); the two roles are distinct (request role, ensemble size, training set, training record); see cross_checks.key_collision_example for the seeds",
                                                           training_order="start followed by the other selected IDs in increasing order (`train_ids_training_order`); the acquisition order is `train_ids_acquisition_order`",
                                                           member_seeds="recorded `initialization_seeds` (JAX key splitting, not key + k); re-derived as a cross-check"),
                                                cross_checks=summary), indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "plan_copies"}, indent=1))


if __name__ == "__main__":
    main()
