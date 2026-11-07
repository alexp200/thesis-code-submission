#!/usr/bin/env bash
# Pilot first; all/resume require completed reproduction and Landscape A.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
cd "$ROOT"
source experiments/04_ded/cluster/config.local.sh
export REPO_DIR="$ROOT"
FOLDER=experiments/03_thermal_food_processing/side_studies/random_repeats/cluster
export RR_RUN_DIR="${RR_RUN_DIR:-$ROOT/$FOLDER/run}"
PY="$ROOT/.venv/bin/python"
MODE=${1:-pilot}
[[ "$MODE" == pilot || "$MODE" == all ]] || { echo 'Usage: submit.sh pilot|all' >&2; exit 1; }
workers=${RR_MAX_PARALLEL:-64}
[[ "$workers" =~ ^[0-9]+$ ]] && (( workers >= 1 && workers <= 200 )) || { echo 'RR_MAX_PARALLEL must be 1..200' >&2; exit 1; }
mkdir -p "$RR_RUN_DIR/logs"
exec 9>"$RR_RUN_DIR/.submission.lock"
flock -n 9 || { echo 'Another submission is in progress' >&2; exit 1; }
queued=$(squeue -r -h -u "$USER" -n chicken-rr-smoke,chicken-rr-pilot,chicken-rr-full -o '%i')
[[ -z "$queued" ]] || { echo "Random repeats already queued/running: $queued" >&2; exit 1; }
"$PY" "$FOLDER/control.py" prepare
pending() { "$PY" "$FOLDER/control.py" pending --mode "$1"; }
COMMON=(--account="$PROJECT_ACCOUNT" --output="$RR_RUN_DIR/logs/%x-%A_%a.out" --error="$RR_RUN_DIR/logs/%x-%A_%a.err")
EXPORT="ALL,TOOLCHAIN_MODULE=$TOOLCHAIN_MODULE,PYTHON_MODULE=$PYTHON_MODULE,REPO_DIR=$ROOT,RR_RUN_DIR=$RR_RUN_DIR"
journal="$RR_RUN_DIR/logs/submission_$(date +%Y%m%d_%H%M%S).tsv"
submit() {
  local output id
  output=$(sbatch --parsable "$@")
  printf '%s\n' "$output" >&2
  id=${output##*$'\n'}; id=${id%%;*}
  [[ "$id" =~ ^[0-9]+$ ]] || { echo 'Cannot parse sbatch output; check squeue before retry' >&2; exit 1; }
  printf '%s\n' "$id"
}
if [[ "$MODE" == all ]]; then
  "$PY" "$FOLDER/control.py" check --mode smoke
  "$PY" "$FOLDER/control.py" check --mode pilot
  "$PY" "$FOLDER/control.py" check --mode n2
  indices=$(pending full)
  [[ -n "$indices" ]] || { echo 'All 880 ensembles already complete'; exit 0; }
  sbatch --test-only "${COMMON[@]}" --job-name=chicken-rr-full --array="$indices%$workers" \
    --time="${RR_TIME_LIMIT:-02:00:00}" --export="$EXPORT,RR_MODE=full" "$FOLDER/run_cpu.sbatch"
  job=$(submit "${COMMON[@]}" --job-name=chicken-rr-full --array="$indices%$workers" \
    --time="${RR_TIME_LIMIT:-02:00:00}" --export="$EXPORT,RR_MODE=full" "$FOLDER/run_cpu.sbatch")
  printf 'full\t%s\n' "$job" >> "$journal"
else
  smoke=$(pending smoke); pilot=$(pending pilot)
  for mode in smoke pilot; do
    indices=$smoke; limit=00:15:00
    [[ "$mode" == smoke ]] || { indices=$pilot; limit=01:00:00; }
    if [[ -n "$indices" ]]; then
      sbatch --test-only "${COMMON[@]}" --job-name="chicken-rr-$mode" --array="$indices%6" \
        --time="$limit" --export="$EXPORT,RR_MODE=$mode" "$FOLDER/run_cpu.sbatch"
    fi
  done
  dependency=()
  if [[ -n "$smoke" ]]; then
    job=$(submit "${COMMON[@]}" --job-name=chicken-rr-smoke --array="$smoke%4" \
      --time=00:15:00 --export="$EXPORT,RR_MODE=smoke" "$FOLDER/run_cpu.sbatch")
    printf 'smoke\t%s\n' "$job" >> "$journal"
    dependency=(--dependency="afterok:$job" --kill-on-invalid-dep=yes)
  else
    "$PY" "$FOLDER/control.py" check --mode smoke
  fi
  if [[ -n "$pilot" ]]; then
    job=$(submit "${COMMON[@]}" "${dependency[@]}" --job-name=chicken-rr-pilot --array="$pilot%6" \
      --time=01:00:00 --export="$EXPORT,RR_MODE=pilot" "$FOLDER/run_cpu.sbatch")
    printf 'pilot\t%s\n' "$job" >> "$journal"
  else
    "$PY" "$FOLDER/control.py" check --mode pilot
  fi
fi
echo "Submission journal: $journal"
