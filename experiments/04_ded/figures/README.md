# DED evaluation

* `make_dataset_figures.py`: dataset figures (Fig. 4.12: temperature snapshots, operating grid, peak temperature); needs the raw data.
* `make_results_figures.py`: learning curves (Fig. 4.13), ECDFs of the path errors (Fig. 4.14), pick map (Fig. 4.15), per-simulation error ratio (Fig. B.9) and the result tables E.1-E.5; reads the run records through `ded_results_io.py`.

Outputs go to `../thesis_figures/final/` and `../thesis_figures/final_audit/` (numbers behind each figure in `figure_data/`). Paths are defined in `../common/export_paths.py`.
