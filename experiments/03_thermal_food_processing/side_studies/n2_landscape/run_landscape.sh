#!/usr/bin/env bash
# n_D = 2 landscape on a single machine (CPU, one native thread per process, resumable: existing results are skipped).
# Usage (any directory):  bash experiments/03_thermal_food_processing/side_studies/n2_landscape/run_landscape.sh [N_PARALLEL] [TIER]
#   TIER  all = all 15 pool trajectories as start, seed sets A and B (840 ensembles)   [default]
#         min = starts 853, 899, 851, seed set A only (84 ensembles)
# Order: seed set A before B, start by start, candidate by candidate, both architectures, so that partial results are usable. 8 parallel processes need ~8-10 GB of memory.
set -u
# works from any directory and without an activated virtual environment: uses <repo>/.venv/bin/python (override: PYTHON=/path/to/python)
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"; cd "$REPO"
export PY="${PYTHON:-$REPO/.venv/bin/python}"; [ -x "$PY" ] || { echo "python not found: $PY"; exit 1; }
N=${1:-8}; TIER=${2:-all}
F=experiments/03_thermal_food_processing/side_studies/n2_landscape
mkdir -p $F/results $F/logs
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_parallel_codegen_split_count=1
case $TIER in
  min) STARTS="853 899 851"; SETS="A" ;;
  all) STARTS="853 899 851 906 888 835 868 917 874 902 745 795 861 873 905"; SETS="A B" ;;
  *) echo "unknown tier $TIER"; exit 1 ;;
esac
POOL="745 795 835 851 853 861 868 873 874 888 899 902 905 906 917"
for s in $SETS; do for st in $STARTS; do for c in $POOL; do [ $c = $st ] && continue; for a in NODE sPHNN; do echo "$a $st $c $s"; done; done; done; done > $F/logs/task_list_$TIER.txt
echo "$(wc -l < $F/logs/task_list_$TIER.txt) ensembles, $N parallel"
cat $F/logs/task_list_$TIER.txt | xargs -P "$N" -L1 bash -c 'o='$F'/results/$0/start$1/cand$2_set$3.json; [ -f $o ] || "$PY" -u '$F'/landscape_run.py --arch $0 --start $1 --cand $2 --set $3 > '$F'/logs/$0_$1_$2_$3.log 2>&1 || echo FAILED $0 $1 $2 $3 >> '$F'/logs/failed.txt'
echo "done: $(ls $F/results/*/start*/cand*_set?.json 2>/dev/null | wc -l) results, failures: $(cat $F/logs/failed.txt 2>/dev/null | wc -l)"
