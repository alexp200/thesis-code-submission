#!/usr/bin/env bash
#SBATCH --job-name=ded-start-next
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:15:00
#SBATCH --mail-type=NONE
set -euo pipefail
: "${REPO_DIR:?Export REPO_DIR}"
module purge
module load "$TOOLCHAIN_MODULE"
module load "$PYTHON_MODULE"
cd "$REPO_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -u experiments/04_ded/al_runs/start_study/control.py advance \
  --part "$1" --through-N "$2" --throttle "$3"
