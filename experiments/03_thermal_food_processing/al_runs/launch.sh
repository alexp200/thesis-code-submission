#!/usr/bin/env bash
# Submit the smoke-gated job arrays of the 140 tasks of the ten original starts (groups 0-13) without waiting in the terminal.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
source experiments/04_ded/cluster/config.local.sh
export REPO_DIR="$ROOT"
FOLDER=experiments/03_thermal_food_processing/al_runs
LOGS="$ROOT/$FOLDER/study/logs"
mkdir -p "$LOGS"
exec 9>"$LOGS/.submission.lock"
flock -n 9 || { echo "Another submission is in progress" >&2; exit 1; }
name="chicken-warmstart"
smoke_name="chicken-warmstart-smoke"
queued=$(squeue -h -u "$USER" -n "$name,$smoke_name" -o '%i')
if [[ -n "$queued" ]]; then echo "Study already queued/running: $queued" >&2; exit 1; fi
.venv/bin/python "$FOLDER/control.py" prepare --smoke
.venv/bin/python "$FOLDER/control.py" prepare
COMMON=(--account="$PROJECT_ACCOUNT" --output="$LOGS/%x-%A_%a.out" --error="$LOGS/%x-%A_%a.err")
FULL_EXPORT="ALL,TOOLCHAIN_MODULE=$TOOLCHAIN_MODULE,PYTHON_MODULE=$PYTHON_MODULE,REPO_DIR=$REPO_DIR,CHICKEN_WARMSTART_MODE=full"
SMOKE_EXPORT="${FULL_EXPORT%full}smoke"
indices=()
for group in {0..13}; do
  indices[$group]=$(.venv/bin/python "$FOLDER/control.py" pending --group "$group")
done
pending="${indices[*]}"
if [[ -z "${pending// /}" ]]; then echo "All paths complete; no jobs submitted"; exit 0; fi
dependency=()
# Checking all 14 smokes also verifies dependency versions and seed pairing.
if .venv/bin/python "$FOLDER/control.py" check > "$LOGS/smoke_release_check.txt" 2>&1; then
  echo "Existing smoke release valid"
else
  smokes=()
  for group in {0..13}; do
    s=$(.venv/bin/python "$FOLDER/control.py" pending --smoke --group "$group")
    if [[ -n "$s" ]]; then smokes+=("$s"); fi
  done
  if (( ${#smokes[@]} == 0 )); then
    cat "$LOGS/smoke_release_check.txt" >&2
    echo "Completed smokes failed validation; inspect instead of bypassing release" >&2
    exit 1
  fi
  smoke_array=$(IFS=,; echo "${smokes[*]}")
  sbatch --test-only "${COMMON[@]}" --job-name="$smoke_name" --time=02:00:00 \
    --export="$SMOKE_EXPORT" --array="$smoke_array%14" "$FOLDER/run_cpu.sbatch"
fi
# Validate all requested allocations before submitting any real job.
for group in {0..13}; do
  if [[ -n "${indices[$group]}" ]]; then
    sbatch --test-only "${COMMON[@]}" --job-name="$name" --time=23:50:00 \
      --export="$FULL_EXPORT" --array="${indices[$group]}%10" "$FOLDER/run_cpu.sbatch"
  fi
done
journal="$LOGS/submission_$(date +%Y%m%d_%H%M%S).tsv"
submit() {
  local output id
  output=$(sbatch --parsable "$@")
  printf '%s\n' "$output" >&2
  id=${output##*$'\n'}
  id=${id%%;*}
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "Cannot parse Slurm job ID; inspect submission" >&2; return 1; }
  printf '%s\n' "$id"
}
if [[ -n "${smoke_array:-}" ]]; then
  job=$(submit "${COMMON[@]}" --job-name="$smoke_name" --time=02:00:00 \
    --export="$SMOKE_EXPORT" --array="$smoke_array%14" "$FOLDER/run_cpu.sbatch")
  printf 'smoke\t%s\n' "$job" >> "$journal"
  dependency=(--dependency="afterok:$job" --kill-on-invalid-dep=yes)
fi
for group in {0..13}; do
  if [[ -n "${indices[$group]}" ]]; then
    job=$(submit "${COMMON[@]}" "${dependency[@]}" --job-name="$name" --time=23:50:00 \
      --export="$FULL_EXPORT" --array="${indices[$group]}%10" "$FOLDER/run_cpu.sbatch")
    printf 'group%s\t%s\n' "$group" "$job" >> "$journal"
  fi
done
echo "Submitted the study; production waits for successful smokes. Journal: $journal"
