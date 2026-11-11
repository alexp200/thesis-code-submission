# Data access

## Raw data

The raw simulation data are not part of this repository. Place the files at the paths below to run Level 2 and 3.

| file | bytes | SHA-256 | used for |
|---|---|---|---|
| `data/chicken_data/dataset.nc` | 214,352 | `2ff82f83226396abe62c8b9662ad828a9c8db3abcbf06f8b005bc6e71d8590c8` | thermal food processing: 30 simulations (15 selection pool, 15 test) |
| `data/chicken_data_full_clean/dataset.nc` | 5,428,552 | `cb7b31cd7204895f0d34186a4cc359e5c5a157b85741377417ca8b93620b9b32` | catalogue of 805 simulations, Figure 4.7 |
| `data/chicken_data_full_clean/manifest.json`, `EXCITATION_CLASSES.md`, `extra_samples.npz` | | | catalogue description |
| `data/directed_energy_deposition/max_simulation/dataset.nc` | 441,804,120 | `e04bd527a80b6b45a14a7b65f76fbe1c6b9e71071b72da92f57b6ac14331986e` | DED: 25 simulations; all DED training and evaluation, Figure 4.12 |
| `data/directed_energy_deposition/max_simulation/mesh.vtk` | 139,246 | `903f3b349e4dbb84609ec38d2a7bb25ef24e7cbe930787304e1967c5b4015bdd` | DED mesh, Figure 4.12 |

Without the raw data, Level 1 regenerates 46 of the 50 figure and table files of the thesis (all except Figure 4.7 and the three panels of Figure 4.12).

## Large assets

Two archives (about 0.9 GB) are not part of the git repository. They are handed in separately; only `large_assets/MANIFEST.sha256` (the SHA-256 values of the archives) is in the repository.

| archive | size | content | needed for |
|---|---|---|---|
| `ded_pod_bases.tar.gz` | 762 MB | `pod.npz`, the POD basis of each of the 716 stored DED ensembles (in `runs/ded_start_study/` and `runs/ded_final_extension/`) | Level 2 of DED (`tools/recompute_ded_nodes.py`); Level 3 of DED, to reuse the stored bases |
| `thermal_acquisition_evaluation_caches.tar.gz` | 152 MB | cached evaluation and acquisition blocks of the stored thermal food processing paths (10,560 files in `runs/03_AL_chicken/`) | no tool; completes the run folders |

To insert them, run the following from the repository root:

1. Copy the received archives into the folder `large_assets/`, next to `MANIFEST.sha256`.
2. Check them: `shasum -a 256 -c large_assets/MANIFEST.sha256` (Linux: `sha256sum -c large_assets/MANIFEST.sha256`). Every line must end with `OK`.
3. Unpack them: `tar -xzf large_assets/ded_pod_bases.tar.gz` and, if wanted, `tar -xzf large_assets/thermal_acquisition_evaluation_caches.tar.gz`. The paths inside the archives are relative to the repository root, so the files are placed in `runs/` next to the records they belong to. No file of the repository is overwritten.
