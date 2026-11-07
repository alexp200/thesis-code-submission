# Reproduction

All commands are run with the environment of the README and `PYTHONPATH` set to the repository root. Paths of generators and entry points in the tables are relative to `experiments/`.

## Level 1: figures and tables

Run the script that generates a figure or table (the generator) from its own folder, for example `cd experiments/01_cubic_regression && python ffnn_uq_comparison.py`. Outputs are written to `thesis_figures/final/` (figures and tables) and `thesis_figures/final_audit/` (underlying numbers) of the experiment folder; `cost_analysis` writes to `results/`.

| generator | produces | raw data |
|---|---|---|
| `01_cubic_regression/verify_analytic_cubic_regression_mll.py` | Fig. 4.1, Tab. B.1, B.3, B.4 | no |
| `01_cubic_regression/ffnn_uq_comparison.py` | Fig. 4.2, Tab. B.2 | no |
| `02_harmonic_oscillator/figures/make_figures.py` | Figs. 4.3-4.5, B.2, B.3 | no |
| `02_harmonic_oscillator/figures/make_envelope_and_frequencies.py` | Figs. 4.6, B.4, Tab. B.10 | no |
| `02_harmonic_oscillator/figures/make_tables.py` | Tabs. B.5-B.9 | no |
| `02_harmonic_oscillator/ott_check/make_ott_figure.py` | Fig. B.1 (trains the network, about 1 min) | no |
| `03_thermal_food_processing/figures/make_results_figures.py` | Figs. 4.8-4.11, B.7, Tabs. B.11-B.21 | no |
| `03_thermal_food_processing/figures/make_dataset_figure.py` | Fig. 4.7 | yes |
| `04_ded/figures/make_results_figures.py` | Figs. 4.13-4.15, B.9, Tabs. B.22-B.26 | no |
| `04_ded/figures/make_dataset_figures.py` | Fig. 4.12 | yes |
| `cost_analysis/make_cost_figures.py` | Figs. B.8, B.10 | no |
| `03_thermal_food_processing/figures/experiment_graph/{al_flowchart,aggregation_graph,schematic_general}.py` | Figs. 3.1, B.6, B.5 | no |


## Level 2: recomputation from checkpoints

| tool | recomputes |
|---|---|
| `tools/recompute_thermal_nodes.py` | member RMSEs and node error of stored thermal nodes on the 15 test trajectories |
| `tools/recompute_ded_nodes.py` | member RMSEs of stored DED nodes over the 25 simulations, using the POD basis of the node; `--refit-pod` refits the basis from the raw data |
| `tools/recompute_ho.py` | long-horizon moments of LA, MoLA and DE with the member frequencies, and the short-horizon data of all five methods of the harmonic oscillator (about 4 min per model and dataset) |

The POD bases of the DED nodes are in `large_assets/ded_pod_bases.tar.gz`, which is delivered separately; how to check and unpack it is described in [DATA_ACCESS.md](DATA_ACCESS.md#large-assets).
A recomputed member RMSE is accepted if it deviates from the stored value by at most 1e-6 K + 1e-3 times the stored value. The original runs used x86-64 Linux; on other CPUs the adaptive ODE solver gives small differences (relative 1e-5 to 4e-4 in the nodes checked). `recompute_ho.py` compares every array with a relative tolerance of 1e-6 (`--rtol`); on the Apple-silicon laptop that produced the stored results, the recomputation is bit-identical.

## Level 3: training

The stored models were trained on a CPU cluster (Slurm) with the code in this repository. Slurm only distributes independent tasks; the computation of a task is done by the scripts listed below. The stored training times add up to about 2000 hours for the thermal food processing studies and about 2000 hours for DED, so on a single machine only slices (single paths, the first training-set sizes) are practical.

| experiment                                             | entry points                                                                                                      | how to run                                                                                                                                                                                                      |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| thermal food processing, all 15 starts (200 paths)     | `03_thermal_food_processing/al_runs/{control.py,run.py}`, plan `study/plan.json`                                  | one machine: `run_local.sh TASK [PHASES]` (README of the folder); `launch.sh` is the Slurm launcher of the thesis runs for the 140 tasks of the original starts (needs the cluster configuration)               |
| harmonic oscillator, ensembles                         | `02_harmonic_oscillator/ensembles/train_ensembles.py`                                                             | one machine (`--force` retrains an existing checkpoint)                                                                                                                                                         |
| harmonic oscillator, SWAG study and short-rollout data | `02_harmonic_oscillator/swag_dependence/{run_sweep.py,aggregate.py}`, `short_horizon/compute_short_rollout.py`    | one machine, about 15 min per model and dataset (README of the folder)                                                                                                                                          |
| first-acquisition landscape, n_D = 1 models            | `03_thermal_food_processing/side_studies/n2_landscape/{landscape_run.py,n1_run.py,run_landscape.sh}`              | launcher for one machine (header of the script)                                                                                                                                                                 |
| repeated Random paths                                  | `03_thermal_food_processing/side_studies/random_repeats/{random_repeat_run.py,run_random_repeats.sh}`, `cluster/` | launcher for one machine; `cluster/` for Slurm                                                                                                                                                                  |
| offline acquisition scores                             | `03_thermal_food_processing/side_studies/criteria_offline/`                                                       | launcher `run_criteria.sh`                                                                                                                                                                                      |
| DED, five starts (60 paths)                            | `start_study/{control.py,worker.py}`                                                                              | one machine: `run_local.sh PATHS [N]` (README of the folder); cluster: `submit.sh` (Slurm, 16 CPUs per task, 32 for MoLA acquisition; the release gate needs earlier runs that are not part of this repository) |
| DED, extension (five further starts, Random paths 3-5) | `final_extension/{submit.sh,control.py,worker.py}`                                                                | cluster; the same paths run on one machine with `start_study/run_local.sh`                                                                                                                                      |

Tested on an Apple-silicon laptop (single paths, the Python runners; the shell launchers only loop over them):

| what | result |
|---|---|
| thermal food processing, task 30 (original start) up to n_D = 3 and task 150 (additional start) up to n_D = 2 | picks identical to the stored ones; most member RMSEs within a few percent; a single member can end in a different optimum on another CPU (up to a factor of about 3 in the paths checked) |
| DED, NODE / LA, DE, MoLA and Random 0 up to n_D = 2; sPHNN / LA and DE with the acquisition at n_D = 1 | picks identical to the stored ones; member RMSEs of the sPHNN at n_D = 1 within 1 %, of the NODE within a few percent (POD bases fitted locally; with the stored POD bases within 1 %) |
| n_D = 2 landscape, one ensemble (`landscape_run.py`) | member RMSEs agree with the stored ones to three digits |
| offline criteria (`criteria_offline.py`): sPHNN / start 853 (seed set A) and NODE / start 745 (seed set B) | byte-identical to the stored scores |
| harmonic oscillator, NODE ensemble (`train_ensembles.py`, about 2 minutes) | checkpoint bit-identical to the stored one |
| harmonic oscillator, `tools/recompute_ho.py` (sPHNN / left half, both stages) | long-horizon and short-horizon arrays bit-identical to the stored ones |
| harmonic oscillator, `compute_methods.py` (all four combinations), SWAG study (all 133 configurations, comparison curves), `compute_short_rollout.py` (all four combinations) | identical to the stored results; in one of 6000 entries of the MoLA Std of NODE / left half the last bit differs between runs |

Training of the AL studies on a different CPU is not bitwise reproducible (the thesis runs used x86-64 Linux), and single members of the DED ensembles can differ noticeably. `tools/compare_local_thermal.py` and `tools/compare_local_ded.py` print a local path next to the stored records. For DED, unpack `large_assets/ded_pod_bases.tar.gz` first to reuse the stored POD bases (the sign convention of an SVD depends on the platform).
