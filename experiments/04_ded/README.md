# Directed energy deposition (Section 4.2.2, Appendix B.4)

Active learning with NODE and sPHNN surrogates on 25 thermal simulations of directed energy deposition (DED). The models work on POD coefficients, with a node-specific POD basis that is refitted for every training set.

| folder | content |
|---|---|
| `al_runs/start_study/` | the AL workflow of the five first starts (plan, coordinator, worker, `run_local.sh`, Slurm launchers); it also runs the paths of the extension on one machine |
| `al_runs/final_extension/` | the five further starts and the remaining Random paths (coordinator for the cluster) |
| `kernels/` | the computational kernels the workflows import: `core/` (data and POD, training, Laplace fit and scoring), `la_de_runner/` (reporting, entry module), `mola_acquisition/` (MoLA mixture and component workers); see its README |
| `figures/` | `make_results_figures.py` (result figures and tables, loader `ded_results_io.py`), `make_dataset_figures.py` (Fig. 4.12), `authored_tables/` (hand-written table fragments of the thesis) |
| `common/` | output locations of the generators |
| `metadata/` | seed manifests: which training used which keys and data (see its README) |
| `cluster/config.example.sh` | template for the Slurm settings that the launch scripts source |

The final comparison (ten starts, n_D = 1..5, six Random paths per start) was produced by `start_study` (five starts) and `final_extension` (five further starts and the remaining Random paths). Run records and checkpoints are in `runs/ded_start_study` and `runs/ded_final_extension`; the node-specific POD bases are in `large_assets/ded_pod_bases.tar.gz`. Training needs `data/directed_energy_deposition/max_simulation/dataset.nc`.
