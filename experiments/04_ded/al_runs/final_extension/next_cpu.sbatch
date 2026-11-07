#!/usr/bin/env bash
#SBATCH --job-name=ded-final-next
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:15:00
set -euo pipefail
module purge
module load "${TOOLCHAIN_MODULE:?}" "${PYTHON_MODULE:?}"
cd "${REPO_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -u experiments/04_ded/al_runs/final_extension/control.py advance \
  --through-N "$1" --throttle "$2"
