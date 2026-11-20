#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
run_dir="reproductions/stage13-$(date -u +%Y%m%dT%H%M%SZ)"
"$PYTHON_BIN" -m unittest test_stage13_local_fem test_stage13_experiment -v
"$PYTHON_BIN" stage13_experiment.py --output "$run_dir"
"$PYTHON_BIN" report_stage13.py --source "$run_dir" --output "$run_dir/report"
"$PYTHON_BIN" stage13_selection_diagnostic.py --source "$run_dir" --output "$run_dir/selection_diagnostic.json"
