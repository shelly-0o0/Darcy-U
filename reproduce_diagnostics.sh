#!/usr/bin/env bash
# Uses the committed, frozen model checkpoints; no additional model training.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
DARCY_PYTHON="${DARCY_PYTHON:-python}"
DARCY_RUN_NAME="${DARCY_RUN_NAME:-reproduced_diagnostic_cpu}"
"$DARCY_PYTHON" -m unittest discover -s . -p '*test*.py' -v
"$DARCY_PYTHON" diagnose.py --run-name "$DARCY_RUN_NAME"
"$DARCY_PYTHON" report_diagnostics.py "results/$DARCY_RUN_NAME"
