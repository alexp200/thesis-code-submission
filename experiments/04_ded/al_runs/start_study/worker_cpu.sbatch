#!/usr/bin/env bash
#SBATCH --job-name=ded-start
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --mail-type=NONE
set -euo pipefail
: "${REPO_DIR:?Export REPO_DIR}"
: "${TOOLCHAIN_MODULE:?Export TOOLCHAIN_MODULE}"
: "${PYTHON_MODULE:?Export PYTHON_MODULE}"
module purge
module load "$TOOLCHAIN_MODULE"
module load "$PYTHON_MODULE"
cd "$REPO_DIR"
ulimit -v unlimited
srun --ntasks=1 --cpu-bind=cores bash experiments/shared/cluster_recovery/run_cpu.sh -u \
  experiments/04_ded/al_runs/start_study/worker.py \
  --run "$1" --batch "$2" --task "$SLURM_ARRAY_TASK_ID"
