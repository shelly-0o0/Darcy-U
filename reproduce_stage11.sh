#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
run_dir="reproductions/stage11-$(date -u +%Y%m%dT%H%M%SZ)"
"$PYTHON_BIN" -m unittest test_stage11_adaptive test_stage11_experiment -v
"$PYTHON_BIN" stage11_experiment.py --output "$run_dir"
"$PYTHON_BIN" report_stage11.py --source "$run_dir" --output "$run_dir/report"
