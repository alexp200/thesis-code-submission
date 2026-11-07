#!/usr/bin/env bash
# Runs one path (task) of the study on this machine, without Slurm.
#
#   bash experiments/03_thermal_food_processing/al_runs/run_local.sh TASK [MAX_PHASES]
#
#   TASK        task number 0-199 (`python control.py progress` lists them with model, method, criterion and path)
#   MAX_PHASES  maximum number of phases run by this call; a training-set size takes three phases (train,
#               evaluate, acquire), so 9 phases from a fresh start give n_D = 1..3. Default 45 = the complete path
#               of an original start (n_D = 1..15); the additional starts (tasks 140-199) finish after 14 phases (n_D = 1..5).
#
# Every phase runs in a fresh process through cluster_recovery/run_cpu.sh (CPU, float64, one native thread).
# Finished phases are never repeated: the script continues where the previous call stopped, and it also
# continues after an interruption.
# Results are written to study/results/; the stored records of the thesis are in runs/.
set -euo pipefail
TASK=${1:?Usage: run_local.sh TASK [MAX_PHASES]}
MAX_PHASES=${2:-45}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
FOLDER=experiments/03_thermal_food_processing/al_runs
for ((phase=0; phase<MAX_PHASES; phase++)); do
  bash experiments/shared/cluster_recovery/run_cpu.sh -u "$FOLDER/run.py" --task "$TASK"
  if .venv/bin/python "$FOLDER/control.py" done --task "$TASK"; then
    echo "Task $TASK is complete."
    exit 0
  fi
done
echo "Stopped after $MAX_PHASES phases; call the script again to continue with the next phases." >&2
