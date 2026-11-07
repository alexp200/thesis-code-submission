#!/usr/bin/env bash
#SBATCH --job-name=chicken-warmstart
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:50:00
#SBATCH --mail-type=NONE
set -euo pipefail
module purge
module load "${TOOLCHAIN_MODULE:?}" "${PYTHON_MODULE:?}"
cd "${REPO_DIR:?}"
FOLDER=experiments/03_thermal_food_processing/al_runs
options=()
if [[ "${CHICKEN_WARMSTART_MODE:-full}" == smoke ]]; then
  options=(--smoke)
else
  .venv/bin/python "$FOLDER/control.py" check
fi
# Every phase has a fresh process; only committed artifacts are resumed.
for ((phase=0; phase<45; phase++)); do
  srun --ntasks=1 --cpu-bind=cores bash experiments/shared/cluster_recovery/run_cpu.sh -u \
    "$FOLDER/run.py" --task "$SLURM_ARRAY_TASK_ID" ${options[@]+"${options[@]}"}
  if .venv/bin/python "$FOLDER/control.py" done --task "$SLURM_ARRAY_TASK_ID" ${options[@]+"${options[@]}"}; then
    exit 0
  fi
done
echo "Task incomplete after 45 phases; inspect before resubmission" >&2
exit 1
