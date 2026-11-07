#!/usr/bin/env bash
# Offline scoring of the n_D = 1 ensembles (LA / DE / MoLA, no training) for the 15 starts and both architectures.
#   bash experiments/03_thermal_food_processing/side_studies/criteria_offline/run_criteria.sh [N_PARALLEL]
# STARTS="853 899" restricts the starts (the five starts without a warm-start block need n2_landscape/n1_run.py first). Resumable.
# SET=B (or C) scores the n_D = 1 ensembles of that seed set (n2_landscape/n1_run.py --set X). Default A (the stored n_D = 1 ensembles).
set -u
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"; cd "$REPO"
export PY="${PYTHON:-$REPO/.venv/bin/python}"; [ -x "$PY" ] || { echo "python not found: $PY"; exit 1; }
N=${1:-3}; export SET=${SET:-A}; TAG=""
if [ "$SET" != "A" ]; then TAG="_set$SET"; fi
export TAG; F=experiments/03_thermal_food_processing/side_studies/criteria_offline; mkdir -p $F/results $F/logs
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_parallel_codegen_split_count=1
for st in ${STARTS:-853 899 851 906 888 835 868 917 874 902 745 795 861 873 905}; do for a in sPHNN NODE; do echo "$a $st"; done; done > $F/logs/task_list.txt
cat $F/logs/task_list.txt | xargs -P "$N" -L1 bash -c 'o='$F'/results/v2_$0_start$1${TAG}.csv; [ -f $o ] || "$PY" -u '$F'/criteria_offline.py --arch $0 --start $1 --set $SET > '$F'/logs/v2_$0_$1${TAG}.log 2>&1 || echo FAILED $0 $1 ${TAG} >> '$F'/logs/failed.txt'
echo "done: $(ls $F/results/v2_*_start*${TAG}.csv 2>/dev/null | wc -l) results, failures: $(cat $F/logs/failed.txt 2>/dev/null | wc -l)"
