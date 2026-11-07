#!/usr/bin/env bash
# Set the tested environment at the actual Python-process boundary.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ ! -f "$ROOT/restart_snapshot.json" || ! -f "$ROOT/phase_integration.json" || -e "$ROOT/.restart_preparation_incomplete" ]]; then
  echo "This launcher requires a completely prepared isolated repository." >&2
  exit 1
fi
found=0
for flag in ${XLA_FLAGS:-}; do
  case "$flag" in
    --xla_cpu_parallel_codegen_split_count=1) found=1 ;;
    *xla_cpu_parallel_codegen_split_count*)
      echo "Conflicting CPU codegen setting: $flag" >&2; exit 1 ;;
  esac
done
flags="${XLA_FLAGS:-}"
if [[ "$found" == 0 ]]; then
  flags="$flags --xla_cpu_parallel_codegen_split_count=1"
fi
cd "$ROOT"
exec env JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  XLA_FLAGS="$flags" "$ROOT/.venv/bin/python" "$@"
