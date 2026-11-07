"""Builds the seed / provenance metadata of the thermal food processing experiment (read-only on all sources; no training, no new random draws).

Sources (all recorded files; relative to the repository root):
  AL / Random paths        runs/03_AL_chicken/warmstart_path_AL/small/{plan.json, results/**/iterNN_train.json, iterNN.json}   (ten original starts, n_D = 1..15)
                           .../warmstart_path_AL/small_extra5/run/{plan.json, manifest.json, results/**, shared_n1/**}                                                             (five additional starts, n_D = 1..5)
  exhaustive n_D = 2       experiments/03_thermal_food_processing/side_studies/n2_landscape/merged/{NODE,sPHNN}/start*/cand*_set{A,B}.json
  n_D = 1 initial models   experiments/03_thermal_food_processing/side_studies/n2_landscape/n1_models/{arch}/start*[_setB|_setC].json  (+ AL iter00 records for set A of the original starts)
  repeated Random          experiments/03_thermal_food_processing/side_studies/random_repeats/{plan.json, tasks.tsv}; trainings runs/03_AL_chicken/investigations/random_repeats/cluster/run/results/**
Key recipe (checked against the records, NEVER used to overwrite a recorded value): b = 7_000_000 + 100_000 p + 1000 (n_D - 1) + delta_X (original starts, key index p), 8_000_000 + 100_000 i + ... (additional starts,
key index i); delta_A = 0, delta_B = 50_000, delta_C = 70_000; member keys b + k (k = 0, 1, 2), optimiser key b + 1.
Outputs (this folder): seed_manifest_al.csv, seed_manifest_landscape.csv, seed_manifest_n1.csv, seed_manifest_random_repeats.csv, seed_manifest.json (constants, cohort-to-key mapping, summary of the cross-checks), plans/ (verbatim copies of the existing plan files),
random_permutations/ (stored Random extension of the original starts, repeated-Random tails, recomputation check), pick_audit.csv / pick_audit.md, environment_recorded.json.
Usage: python build_submission_metadata.py"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("SUBMISSION_METADATA_OUT", HERE))        # submission: outputs can be redirected (verification)
FE = HERE.parents[1]                                                    # experiments/
REPO = FE.parent                                                         # repository root
RES = REPO / "runs/03_AL_chicken"
WS, WX = RES / "warmstart_path_AL/small", RES / "warmstart_path_AL/small_extra5/run"
N2 = FE / "03_thermal_food_processing/side_studies/n2_landscape"
RR = FE / "03_thermal_food_processing/side_studies/random_repeats"
RR_RUN = RES / "investigations/random_repeats/cluster/run"
rel = lambda p: str(Path(p).relative_to(REPO))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
DELTA = {"A": 0, "B": 50_000, "C": 70_000}
POOL = [745, 795, 835, 851, 853, 861, 868, 873, 874, 888, 899, 902, 905, 906, 917]
TEST = [313, 320, 344, 378, 383, 407, 412, 415, 461, 462, 466, 467, 474, 508, 528]


def load_module(path, name):
    """Import a script by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def write_csv(name, rows):
    """Write a list of dicts as CSV into the output folder."""
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def expected_base(cohort, idx, n_d, seed_set="A"):
    """Training key from the key recipe."""
    return (7_000_000 if cohort == "original" else 8_000_000) + 100_000 * idx + 1000 * (n_d - 1) + DELTA[seed_set]


def al_rows(summary):
    """one row per recorded training of an AL / Random path (iteration = n_D - 1)"""
    rows = []
    for cohort, root, plan_file in (("original", WS, WS / "plan.json"), ("additional", WX, WX / "plan.json")):
        plan = json.loads(plan_file.read_text())
        for f in sorted((root / "results").glob("*/*/*/path??/iter??_train.json")):
            arch, method, crit, pdir = f.parts[-5:-1]
            path = int(pdir[4:])
            tr = json.loads(f.read_text())
            ev = json.loads(f.with_name(f.name.replace("_train", "")).read_text())
            n_d = int(tr["dataset_size"])
            start = int(tr["train_ids"][0])
            assert plan["warmstart_ids"][str(path)] == start, (f, start)
            b = expected_base(cohort, path, n_d)
            ck = f.with_name(f.name.replace("_train.json", ".eqx"))
            shared = not ck.exists()
            if shared and cohort == "additional" and n_d == 1:                                   # n_D = 1 of the additional starts is one shared fit per architecture / start (shared_n1/)
                cand = sorted((WX / "shared_n1").glob(f"*/{arch}/start{start}/iter00.eqx"))
                ck, shared = (cand[0], True) if cand else (None, True)
            rec_ok = tr["training_key"] == b and tr["initialization_seeds"] == [b, b + 1, b + 2] and tr["optimizer_key"] == b + 1
            rows.append(dict(cohort=cohort, key_index=path, start_id=start, arch=arch, method="Random" if method == "random" else method, criterion=crit, n_D=n_d, seed_set="A", train_ids_in_training_order=" ".join(map(str, tr["train_ids"])),
                             training_key=tr["training_key"], initialization_seeds=" ".join(map(str, tr["initialization_seeds"])), optimizer_key=tr["optimizer_key"],
                             recipe_key_match=rec_ok, recipe_training_key=b, fit_kind="shared n_D=1 fit (one training, read by every stream of the start)" if shared and n_d == 1 and cohort == "additional" else "path occurrence (own training, paired keys)",
                             checkpoint=rel(ck) if ck is not None and Path(ck).exists() else "", checkpoint_sha256_recorded=json.dumps(tr.get("artifact_sha256", {})) if tr.get("artifact_sha256") else "",
                             steps=tr["recipe"]["steps"], lr=tr["recipe"]["lr"], l2=tr["recipe"]["l2"], effective_members=ev.get("effective_ensemble_size"), dropped_indices=" ".join(map(str, ev.get("dropped_indices", []))),
                             host=tr.get("hostname", ""), slurm_job_id=tr.get("slurm_job_id", ""), train_seconds=round(tr.get("train_seconds", float("nan")), 1), record=rel(f)))
    for r in rows:                                                                               # a recorded checkpoint hash that occurs in several records = ONE fit that several path occurrences read
        try:
            r["fit_id"] = next(iter(json.loads(r["checkpoint_sha256_recorded"]).values()))[:12] if r["checkpoint_sha256_recorded"] else ""
        except (ValueError, StopIteration):
            r["fit_id"] = ""
    from collections import Counter
    cnt = Counter(r["fit_id"] for r in rows if r["fit_id"])
    for r in rows:
        r["records_sharing_this_fit"] = cnt[r["fit_id"]] if r["fit_id"] else ""
        if r["fit_id"] and cnt[r["fit_id"]] > 1:
            r["fit_kind"] = f"ONE fit read by {cnt[r['fit_id']]} path occurrences (identical recorded checkpoint hash)"
    summary["al_fits_shared_by_several_records"] = sum(1 for k, v in cnt.items() if v > 1)
    summary["al_records_in_shared_fits"] = sum(v for v in cnt.values() if v > 1)
    summary["al_unique_recorded_fits"] = len(cnt)
    summary["al_records"] = len(rows)
    summary["al_recipe_key_mismatches"] = [r["record"] for r in rows if not r["recipe_key_match"]]
    return rows


def landscape_rows(summary):
    """One row per exhaustive n_D = 2 fit (start, candidate, architecture, seed set)."""
    ls = load_module(N2 / "landscape_run.py", "landscape_run_meta")
    rows = []
    for f in sorted((N2 / "merged").glob("*/start*/cand*_set?.json")):
        arch, sdir, name = f.parts[-3:]
        d = json.loads(f.read_text())
        s, c, X = int(sdir[5:]), int(name.split("_")[0][4:]), name.split("_set")[1][0]
        base = ls.key_base(s, X)
        cohort, idx = ("original", ls.BLOCK[s]) if s in ls.BLOCK else ("additional", ls.EXTRA[s])
        rows.append(dict(cohort=cohort, key_index=idx, start_id=s, candidate_id=c, arch=arch, seed_set=X, n_D=2, train_ids_in_training_order=" ".join(map(str, d["train_ids"])), training_key=d["training_key"], optimizer_key=d["optimizer_key"],
                         recipe_training_key=base, recipe_key_match=(d["training_key"] == base and d["optimizer_key"] == base + 1 and base == expected_base(cohort, idx, 2, X)), fit_kind="exhaustive two-trajectory fit (also read by Random n_D = 2 when set A)",
                         steps=d["steps"], effective_members=d.get("effective_members"), dropped_indices=" ".join(map(str, d.get("dropped_indices", []))), host=d.get("host", ""), checkpoint=rel(f.with_suffix(".eqx")) if f.with_suffix(".eqx").exists() else "", record=rel(f)))
    summary["landscape_records"] = len(rows)
    summary["landscape_by_set"] = {X: sum(r["seed_set"] == X for r in rows) for X in "AB"}
    summary["landscape_recipe_key_mismatches"] = [r["record"] for r in rows if not r["recipe_key_match"]]
    return rows


def n1_rows(summary):
    """One row per n_D = 1 initial model (seed sets A, B, C)."""
    ls = load_module(N2 / "landscape_run.py", "landscape_run_meta2")
    rows = []
    for f in sorted((N2 / "n1_models").glob("*/start*.json")):
        arch, name = f.parts[-2:]
        d = json.loads(f.read_text())
        s = int(name[5:].split("_")[0].split(".")[0])
        X = d.get("seed_set", "A")
        base = ls.n1_key(s, X)
        cohort, idx = ("original", ls.BLOCK[s]) if s in ls.BLOCK else ("additional", ls.EXTRA[s])
        rows.append(dict(cohort=cohort, key_index=idx, start_id=s, arch=arch, seed_set=X, n_D=1, train_ids_in_training_order=str(s), training_key=d["training_key"], optimizer_key=d["optimizer_key"], recipe_training_key=base,
                         recipe_key_match=(d["training_key"] == base and d["optimizer_key"] == base + 1 and base == expected_base(cohort, idx, 1, X)), fit_kind="initial model for the acquisition-sensitivity analysis",
                         checkpoint=rel(f.with_suffix(".eqx")) if f.with_suffix(".eqx").exists() else "", record=rel(f)))
    have = {(r["arch"], r["start_id"], r["seed_set"]) for r in rows}
    for r in al_rows_cache:                                                                     # set A of the original starts: the AL n_D = 1 fit of the block (one record per stream; MoLA / MGE referenced)
        if r["cohort"] == "original" and r["n_D"] == 1 and r["method"] == "MoLA" and r["criterion"] == "sum_marginal" and (r["arch"], r["start_id"], "A") not in have:
            rows.append(dict(cohort="original", key_index=r["key_index"], start_id=r["start_id"], arch=r["arch"], seed_set="A", n_D=1, train_ids_in_training_order=str(r["start_id"]), training_key=r["training_key"], optimizer_key=r["optimizer_key"],
                             recipe_training_key=r["recipe_training_key"], recipe_key_match=r["recipe_key_match"], fit_kind="AL iter00 fit of the block (MoLA / MGE record referenced; every stream of the block has its own paired record)", checkpoint=r["checkpoint"], record=r["record"]))
    summary["n1_records"] = len(rows)
    summary["n1_recipe_key_mismatches"] = [r["record"] for r in rows if not r["recipe_key_match"]]
    return rows


def random_repeat_rows(summary):
    """One row per training of the repeated Random baseline, plus the n_D = 2 cells that reuse the landscape fits."""
    plan = json.loads((RR / "plan.json").read_text())
    ls = load_module(N2 / "landscape_run.py", "landscape_run_meta3")
    rows = []
    for t in plan["tasks"]:
        f = RR_RUN / "results" / t["arch"] / f"start{t['start']}" / f"N{t['N']}_task{t['task']}.json"
        rec = json.loads(f.read_text()) if f.exists() else {}
        s = t["start"]
        cohort, idx = ("original", ls.BLOCK[s]) if s in ls.BLOCK else ("additional", ls.EXTRA[s])
        b = expected_base(cohort, idx, t["N"])
        rows.append(dict(kind="new training (n_D >= 3)", cohort=cohort, key_index=idx, start_id=s, arch=t["arch"], n_D=t["N"], paths_r="/".join(map(str, t["paths"])), train_ids_in_training_order=" ".join(map(str, t["train_ids"])),
                         planned_training_key=t["training_key"], planned_optimizer_key=t["optimizer_key"], recorded_training_key=rec.get("training_key", ""), recorded_optimizer_key=rec.get("optimizer_key", ""), recipe_training_key=b,
                         recipe_key_match=(t["training_key"] == b and t["optimizer_key"] == b + 1 and (rec.get("training_key", b) == b)), steps=t["steps"], lr=t["lr"], l2=t["l2"], host=rec.get("host", ""), checkpoint=rel(f.with_suffix(".eqx")) if f.with_suffix(".eqx").exists() else "",
                         record=rel(f) if f.exists() else "missing", plan_task=t["task"]))
    for e in plan["landscape_n2"]:                                                              # n_D = 2: no new training, the exhaustive set-A fit is read
        s = e["start"]
        cohort, idx = ("original", ls.BLOCK[s]) if s in ls.BLOCK else ("additional", ls.EXTRA[s])
        rows.append(dict(kind="reused fit (n_D = 2 = exhaustive set A, no training)", cohort=cohort, key_index=idx, start_id=s, arch=e["arch"], n_D=2, paths_r=str(e["r"]), train_ids_in_training_order=f"{s} {e['candidate']}", planned_training_key=ls.key_base(s, "A"),
                         planned_optimizer_key=ls.key_base(s, "A") + 1, recorded_training_key="", recorded_optimizer_key="", recipe_training_key=expected_base(cohort, idx, 2), recipe_key_match=ls.key_base(s, "A") == expected_base(cohort, idx, 2), steps="", lr="", l2="", host="",
                         checkpoint=rel(N2 / "merged" / e["arch"] / f"start{s}" / f"cand{e['candidate']}_setA.eqx"), record=rel(N2 / "merged" / e["arch"] / f"start{s}" / f"cand{e['candidate']}_setA.json"), plan_task=""))
    summary["random_repeat_rows"] = len(rows)
    summary["random_repeat_trainings"] = sum(r["kind"].startswith("new") for r in rows)
    summary["random_repeat_missing_records"] = sum(r["record"] == "missing" for r in rows)
    summary["random_repeat_recipe_key_mismatches"] = [r["plan_task"] for r in rows if not r["recipe_key_match"]]
    return rows, plan


def permutations(summary, rr_plan):
    """Stored Random orders and repeated-Random tails, with the check against their recomputation from the RNG recipes."""
    out = OUT / "random_permutations"
    out.mkdir(exist_ok=True)
    plan = json.loads((WS / "plan.json").read_text())
    stored, ok = {}, True
    for p, start in plan["warmstart_ids"].items():
        remaining = sorted(i for i in POOL if i != start)
        recomputed = [start, *np.random.default_rng(20262012 + int(p)).permutation(remaining).tolist()]
        rec = plan["random_orders"][p]
        stored[p] = dict(start=start, plan_order_first_15=rec, recomputed_order=recomputed, identical=bool(rec == recomputed[:len(rec)]))
        ok &= stored[p]["identical"]
    (out / "stored_random_extension_original_starts.json").write_text(json.dumps(dict(rng="np.random.default_rng(20262012 + p).permutation(sorted(remaining_ids)), remaining = pool without the start; order = [start, *permutation]", per_key_index=stored), indent=1))
    tails, ok2 = {}, True
    for s in POOL:
        for r in range(10):
            rec = rr_plan["tails"][str(s)][r]
            re = np.random.default_rng([20263012, s, r]).permutation(sorted(i for i in POOL if i != s)).tolist()
            ok2 &= (rec == re)
            tails[f"{s}/{r}"] = dict(plan=rec, recomputed=re)
    (out / "repeated_random_tails.json").write_text(json.dumps(dict(rng="np.random.default_rng([20263012, s, r]).permutation(sorted(other pool ids)); prefixes define n_D = 2..5; shared across architectures", tails=tails), indent=1))
    summary["stored_random_extension_matches_recomputation"] = bool(ok)
    summary["repeated_random_tails_match_recomputation"] = bool(ok2)


def copy_plans(summary):
    """Verbatim copies of the existing plans."""
    out = OUT / "plans"
    out.mkdir(exist_ok=True)
    files = {"warmstart_small_plan.json": WS / "plan.json", "warmstart_extra5_plan.json": WX / "plan.json", "warmstart_extra5_manifest.json": WX / "manifest.json", "random_repeats_plan.json": RR / "plan.json", "random_repeats_tasks.tsv": RR / "tasks.tsv"}
    summary["plan_copies"] = {}
    for n, src in files.items():
        shutil.copyfile(src, out / n)
        summary["plan_copies"][n] = dict(source=rel(src), sha256=sha(src))


def pick_audit(summary):
    """first selection of every stored AL path (n_D = 2) against the argmax of the separately evaluated acquisition scores (offline scores, seed set A)"""
    sys.path.insert(0, str(FE / "03_thermal_food_processing/figures"))
    mrf = load_module(FE / "03_thermal_food_processing/figures/make_results_figures.py", "mrf_meta")
    S = mrf.S
    D0 = S.load_all()
    W = D0[D0.study.isin(["warm_small_K3", "warm_extra5_K3"])].rename(columns={"median": "ap15"})[["arch", "method", "criterion", "path", "N", "ap15", "train_ids", "members"]]
    W["start"] = W.train_ids.apply(lambda t: t[0])
    P = mrf.pick_table_offline(mrf.scores(), mrf.landscape(), mrf.oven_means())
    A = W[(W.N == 2) & (W.method != "Random")].copy()
    A["actual_pick"] = A.train_ids.apply(lambda t: t[1])
    m = A.merge(P.rename(columns={"crit": "criterion", "pick": "diagnostic_pick"}), on=["arch", "start", "method", "criterion"])
    rows = [dict(arch=r.arch, start_id=r.start, method=r.method, criterion=r.criterion, actual_AL_pick=r.actual_pick, diagnostic_pick=r.diagnostic_pick, agree=bool(r.actual_pick == r.diagnostic_pick)) for r in m.sort_values(["arch", "start", "method", "criterion"]).itertuples()]
    write_csv("pick_audit.csv", rows)
    diff = [r for r in rows if not r["agree"]]
    summary["pick_audit"] = dict(total=len(rows), agree=len(rows) - len(diff), differences=[f"{r['arch']}/{r['method']}/{r['criterion']} start {r['start_id']}: actual {r['actual_AL_pick']}, diagnostic {r['diagnostic_pick']}" for r in diff])
    (OUT / "pick_audit.md").write_text(
        "# Actual AL picks versus separately evaluated acquisition scores (first selection, n_D = 1 -> 2)\n\n"
        f"{len(rows) - len(diff)} of {len(rows)} first selections agree (2 architectures x 15 starts x 6 strategies). Differences:\n\n" + "\n".join(f"- {d}" for d in summary["pick_audit"]["differences"]) + "\n\n"
        "The first-acquisition figures use the ACTUAL AL-path picks (`actual_AL_pick`, from the stored paths); the error rankings and the seed-sensitivity analysis keep their diagnostic data (separately evaluated acquisition scores, exhaustive errors). "
        "The diagnostic models were evaluated locally, while the AL paths ran on the cluster; equal keys do not imply bitwise identical fits across machines. Nothing was recomputed or replaced to force agreement.\n\n`pick_audit.csv`: all comparisons.\n")


def environment(summary):
    """Recorded package versions and implementation hashes."""
    m = json.loads((WX / "manifest.json").read_text())
    pl_s, pl_x = json.loads((WS / "plan.json").read_text()), json.loads((WX / "plan.json").read_text())
    env = dict(additional_starts_run=dict(source="small_extra5/run/manifest.json (recorded at launch)", python=m["dependencies"]["python"], package_versions=m["dependencies"]["versions"], vcs_installs=m["dependencies"]["direct_urls"], git_commit=m["git_commit"] or "unknown (recorded empty)",
                                           source_tree=m["source"], dataset_sha256=m["dataset_sha256"]),
               original_starts_run=dict(source="small/plan.json (recorded); no dependency manifest was recorded for this run", package_versions="unknown", git_commit="unknown", dataset_sha256=pl_s["dataset_sha256"]),
               implementation_sha256_recorded=dict(original=pl_s["implementation_sha256"], additional=pl_x["implementation_sha256"]),
               note="Versions are those recorded for the five additional starts; the original ten-start run recorded implementation hashes and the dataset hash only. Current package versions are NOT substituted.")
    (OUT / "environment_recorded.json").write_text(json.dumps(env, indent=1))
    summary["dataset_sha256_original_equals_additional"] = pl_s["dataset_sha256"] == pl_x["dataset_sha256"]


def main():
    """Build all tables and the summary of the cross-checks."""
    global al_rows_cache
    summary = {}
    al_rows_cache = al_rows(summary)
    write_csv("seed_manifest_al.csv", al_rows_cache)
    write_csv("seed_manifest_landscape.csv", landscape_rows(summary))
    write_csv("seed_manifest_n1.csv", n1_rows(summary))
    rr_rows, rr_plan = random_repeat_rows(summary)
    write_csv("seed_manifest_random_repeats.csv", rr_rows)
    permutations(summary, rr_plan)
    copy_plans(summary)
    pick_audit(summary)
    environment(summary)
    ws, wx = json.loads((WS / "plan.json").read_text()), json.loads((WX / "plan.json").read_text())
    const = dict(selection_pool_sorted=sorted(POOL), ap15_test_set=TEST,
                 original_starts_key_index_order={int(k): v for k, v in ws["warmstart_ids"].items()}, original_start_draw="np.random.default_rng(20261012).choice(sorted(pool), size=10, replace=False)",
                 additional_starts_key_index_order={int(k): v for k, v in wx["warmstart_ids"].items()}, seed_set_offsets=DELTA,
                 key_recipe="b = 7000000 + 100000*p + 1000*(n_D-1) + delta_X (original starts, key index p); 8000000 + 100000*i + 1000*(n_D-1) + delta_X (additional starts, key index i); member keys b + k (k = 0, 1, 2); optimiser key b + 1",
                 roles="A: main training schedule of all AL / Random paths and the exhaustive n_D = 2 errors; B: second exhaustive n_D = 2 set; A, B, C initial models: acquisition-sensitivity analysis (C is not a full AL-path repetition)",
                 random_rng=dict(stored_extension="np.random.default_rng(20262012 + p).permutation(sorted(remaining_ids)), original starts only", repeated="np.random.default_rng([20263012, s, r]).permutation(sorted(remaining_ids)), r = 0..9, all 15 starts, prefixes n_D = 2..5, shared across architectures"),
                 tie_tolerance_K=1e-12)
    (OUT / "seed_manifest.json").write_text(json.dumps(dict(constants=const, cross_checks=summary), indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "plan_copies"}, indent=1))


if __name__ == "__main__":
    main()
