#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$ROOT"
source experiments/04_ded/cluster/config.local.sh
export REPO_DIR="$ROOT" PROJECT_ACCOUNT TOOLCHAIN_MODULE PYTHON_MODULE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
FOLDER=experiments/04_ded/al_runs/start_study
case "${1:-}" in
  start)
    .venv/bin/python "$FOLDER/control.py" freeze --part gate
    .venv/bin/python "$FOLDER/control.py" freeze --part main
    .venv/bin/python "$FOLDER/control.py" queue-main --through-N "${2:-5}" --throttle "${3:-4}"
    .venv/bin/python "$FOLDER/control.py" advance --part gate --through-N 2 --throttle 2
    ;;
  gate)
    .venv/bin/python "$FOLDER/control.py" freeze --part gate
    .venv/bin/python "$FOLDER/control.py" advance --part gate --through-N 2 --throttle 2
    ;;
  main|B|C)
    part="$1"
    limit="${2:-5}"
    if [[ "$part" == C ]]; then limit=2; fi
    .venv/bin/python "$FOLDER/control.py" freeze --part "$part"
    .venv/bin/python "$FOLDER/control.py" advance --part "$part" --through-N "$limit" --throttle "${3:-4}"
    ;;
  *) echo "Usage: bash $FOLDER/submit.sh {start|gate|main|B|C} [through-N] [throttle]" >&2; exit 2 ;;
esac
