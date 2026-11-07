"""Offline acquisition scores at n_D = 1 for LA, DE and MoLA (the criteria of the AL runs, sigma_noise = 0) with the feature-matrix formulation of acq_criteria.py.
Scores the n_D = 1 ensemble (K = 3) of one start on the 14 remaining pool trajectories; no training. Columns per candidate: {LA,DE,MoLA}_{sum_variance,sum_marginal} (VAR, MGE).
LA = member 0, as in the AL runs. The n_D = 1 ensemble of seed set A comes from the warm-start study (iter00.eqx) or, for the five extra starts, from n2_landscape/n1_models (n1_run.py); sets B and C from n2_landscape/n1_models.
Usage (repo root, CPU / x64 / one native thread, see run_criteria.sh):  python criteria_offline.py --arch sPHNN --start 853 [--set B]
Output: results/v2_{arch}_start{S}[_set{B|C}].csv. Resumable."""
import argparse
import importlib
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "n2_landscape"))
import acq_criteria as AC
from landscape_run import BLOCK, DATA, POOL, WARM, warm_result


def main():
    """Score all candidates of one start with one n_D = 1 ensemble and write the CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["NODE", "sPHNN"])
    ap.add_argument("--start", required=True, type=int)
    ap.add_argument("--set", default="A", choices=["A", "B", "C"], help="seed set of the n_D = 1 ensemble")
    ap.add_argument("--outdir", type=Path, default=HERE / "results")
    a = ap.parse_args()
    out = a.outdir / f"v2_{a.arch}_start{a.start}{'' if a.set == 'A' else '_set' + a.set}.csv"
    if out.exists():
        print("exists", out); return
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    if any(os.environ.get(n) != "1" for n in names) or "--xla_cpu_parallel_codegen_split_count=1" not in os.environ.get("XLA_FLAGS", "").split():
        raise SystemExit("use run_criteria.sh (single native thread, XLA flag)")
    sys.path.insert(0, str(WARM))
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pandas as pd
    numerical = importlib.import_module("numerical"); numerical.DATA = DATA
    tc, _ = numerical.helpers()
    from src.models import Laplace_NeuralODESolver, Laplace_sPHNNSolver
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("CPU / x64 required")
    a.outdir.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter()

    # ---- n_D = 1 ensemble and the Laplace fit of every member ---------------------------
    if a.set == "A":
        ckpt = warm_result(a.arch, "MoLA/sum_marginal", f"path{BLOCK[a.start]:02d}", "iter00.eqx") if a.start in BLOCK else HERE.parent / "n2_landscape/n1_models" / a.arch / f"start{a.start}.eqx"
    else:
        ckpt = HERE.parent / "n2_landscape/n1_models" / a.arch / f"start{a.start}_set{a.set}.eqx"
    data = numerical.prepare_data([a.start]); t = data["t"]
    members = tc.split_ensemble(eqx.tree_deserialise_leaves(ckpt, numerical.skeleton(a.arch, [0, 1, 2])), 3)
    cls = Laplace_NeuralODESolver if a.arch == "NODE" else Laplace_sPHNNSolver
    comps = []
    for m in members:
        c = cls(m, t, data["x0s_train"], data["ys_train"], us_train=data["us_train"]); c.fit_laplace_mll(); comps.append(c)
    UW = []
    for c in comps:
        U, w = c._posterior_directions_and_weights(); UW.append(U * jnp.sqrt(w))                 # (D, r): Sigma_theta = UW UW^T
    idx_of = lambda i: data["all_ids"].index(i)
    jac = lambda c, x0, u: jax.jacobian(lambda p: c.f_solve_flat_general(p, t, x0, u))(c.p_map)

    # ---- feature matrices of one candidate -----------------------------------------------
    def features(x0, u):
        phis, mus = [], []
        for c, uw in zip(comps, UW):
            phis.append(np.asarray(jac(c, x0, u) @ uw)); mus.append(np.asarray(c.f_solve_flat_general(c.p_map, t, x0, u)))
        mus = np.stack(mus)
        return {"LA": phis[0], "DE": AC.de_features(mus), "MoLA": AC.mola_features(phis, mus)}

    cands = [i for i in POOL if i != a.start]
    F = {i: features(data["x"][idx_of(i)], data["u"][idx_of(i)]) for i in cands}
    Sc = F[cands[0]]["LA"]; ref = comps[0].std_linearized(t, data["x"][idx_of(cands[0])], data["u"][idx_of(cands[0])], cov_type="dense")
    print("self-check LA covariance, max |Phi Phi^T - std_linearized dense| / max:", float(np.abs(Sc @ Sc.T - np.asarray(ref)).max() / np.abs(np.asarray(ref)).max()), flush=True)

    # ---- scores ---------------------------------------------------------------------------
    rows = []
    for c in cands:
        row = dict(arch=a.arch, start=a.start, cand=c)
        for m in ("LA", "DE", "MoLA"):
            row[f"{m}_sum_variance"] = AC.sum_variance(F[c][m]); row[f"{m}_sum_marginal"] = AC.sum_marginal(F[c][m])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{a.arch} start {a.start}: {len(rows)} candidates, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
