#!/usr/bin/env bash
# Runs selected paths of the DED start study on this machine, without Slurm.
#
#   bash experiments/04_ded/al_runs/start_study/run_local.sh PATHS [THROUGH_N]
#
#   PATHS       comma separated path ids, e.g. NODE/start12/LA or NODE/start12/LA,NODE/start12/random0
#               (model / start / stream; streams LA, DE, MoLA, random0, random1, random2; starts 12, 4, 24, 22, 10)
#   THROUGH_N   largest training-set size (n_D), default 2
#
# DED_LOCAL=1 switches off the release gate (a comparison with earlier runs of the thesis that are not part of this
# repository), the Slurm submission and the CPU checks of the cluster; the computational code is the same. The plan and the
# results of a selection are written to run/main_local_<hash of the selection>/; calling the script again with the same
# arguments continues where it stopped. Every task runs in a fresh process through cluster_recovery/run_cpu.sh.
# Stored POD bases (large_assets/ded_pod_bases.tar.gz, unpacked into runs/) are used if present.
set -euo pipefail
export DED_LOCAL=1 DED_LOCAL_PATHS="${1:?Usage: run_local.sh PATHS [THROUGH_N]}"
THROUGH_N=${2:-2}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$ROOT"
FOLDER=experiments/04_ded/al_runs/start_study
.venv/bin/python "$FOLDER/control.py" freeze --part main
.venv/bin/python "$FOLDER/control.py" advance --part main --through-N "$THROUGH_N"
.venv/bin/python "$FOLDER/control.py" progress --part main
