#!/usr/bin/env bash
#SBATCH --job-name=ded-final
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=08:00:00
set -euo pipefail
module purge
module load "${TOOLCHAIN_MODULE:?}" "${PYTHON_MODULE:?}"
cd "${REPO_DIR:?}"
ulimit -v unlimited
srun --ntasks=1 --cpu-bind=cores bash experiments/shared/cluster_recovery/run_cpu.sh -u \
  experiments/04_ded/al_runs/final_extension/worker.py \
  --run "$1" --batch "$2" --task "${SLURM_ARRAY_TASK_ID:?}"
