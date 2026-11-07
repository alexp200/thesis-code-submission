#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$ROOT"
source experiments/04_ded/cluster/config.local.sh
export REPO_DIR="$ROOT" PROJECT_ACCOUNT TOOLCHAIN_MODULE PYTHON_MODULE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
case "${1:-}" in
  prepare) .venv/bin/python experiments/04_ded/al_runs/final_extension/control.py freeze ;;
  upgrade-preparation) .venv/bin/python experiments/04_ded/al_runs/final_extension/control.py upgrade-preparation ;;
  check) .venv/bin/python experiments/04_ded/al_runs/final_extension/control.py check ;;
  start|retry)
    [[ "${FINAL_LAUNCH_APPROVED:-}" == yes && -n "${FINAL_RESERVATION:-}" ]] || {
      echo 'Set FINAL_LAUNCH_APPROVED=yes and the HPC-confirmed FINAL_RESERVATION first.' >&2; exit 1;
    }
    export FINAL_LAUNCH_APPROVED FINAL_RESERVATION FINAL_PARTITION
    retry=()
    [[ "$1" != retry ]] || retry=(--retry-failed)
    .venv/bin/python experiments/04_ded/al_runs/final_extension/control.py check
    .venv/bin/python experiments/04_ded/al_runs/final_extension/control.py advance \
      --through-N "${2:-5}" --throttle "${3:-12}" "${retry[@]}"
    ;;
  *) echo 'Usage: submit.sh prepare|upgrade-preparation|check|start|retry [through-N=5] [throttle=12]' >&2; exit 2 ;;
esac
