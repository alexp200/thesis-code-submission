# DED extension

Adds to the start study the Random paths 3-5 of the first five starts (package T) and the five further starts 20, 2, 14, 18, 6 with LA, DE, MoLA and six Random paths each (package G6). Starts in key-index order: `[12, 4, 24, 22, 10, 20, 2, 14, 18, 6]`; reporting and acquisition keys as in `../start_study`. The other packages of the original plan (start 0, additional first-step Random draws) are not part of the thesis results.

Files: `design.py` (packages, keys), `control.py`, `worker.py`, `submit.sh`, `*_cpu.sbatch`. Stored results: `runs/ded_final_extension/`.

The coordinator of this folder is written for the cluster and checks the validated runs of the start study. The paths of the extension (including the Random paths 3-5 of the first five starts) can be run on a single machine with `../start_study/run_local.sh`, e.g. `NODE/start20/LA`; see the README of the start study.
