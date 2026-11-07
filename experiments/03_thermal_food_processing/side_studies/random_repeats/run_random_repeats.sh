#!/usr/bin/env bash
# Random-repeats trainings (880 unique tasks of plan.json) on a single machine. Usage: bash experiments/03_thermal_food_processing/side_studies/random_repeats/run_random_repeats.sh [N_PARALLEL]   (resumable; the cluster runs use their own array layer, see cluster/)
set -u
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"; cd "$REPO"
export PY="${PYTHON:-$REPO/.venv/bin/python}"; N=${1:-3}; F=experiments/03_thermal_food_processing/side_studies/random_repeats
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_parallel_codegen_split_count=1
mkdir -p $F/logs $F/results
"$PY" -c "import json; print('\n'.join(str(t['task']) for t in json.load(open('$F/plan.json'))['tasks']))" > $F/logs/task_ids.txt
cat $F/logs/task_ids.txt | xargs -P "$N" -L1 bash -c '"$PY" -u '$F'/random_repeat_run.py --task $0 > '$F'/logs/task$0.log 2>&1 || echo FAILED $0 >> '$F'/logs/failed.txt'
echo "done: $(ls $F/results/*/start*/N*_task*.json 2>/dev/null | wc -l) results, failures: $(cat $F/logs/failed.txt 2>/dev/null | wc -l)"
