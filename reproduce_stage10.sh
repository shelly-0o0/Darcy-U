#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
STAGE10_PYTHON="${PYTHON_BIN:-python3}"
STAGE10_DEST="${1:-reproductions/stage10-$(date +%Y%m%d-%H%M%S)}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
"$STAGE10_PYTHON" -m unittest test_stage10_goal_numerics test_stage10_error_model test_stage10_experiment -v
"$STAGE10_PYTHON" -u stage10_experiment.py --output "$STAGE10_DEST/pilot"
"$STAGE10_PYTHON" report_stage10.py --source "$STAGE10_DEST/pilot" --output "$STAGE10_DEST/report"
"$STAGE10_PYTHON" stage10_projection_probe.py --self-test
"$STAGE10_PYTHON" stage10_projection_probe.py --source "$STAGE10_DEST/pilot" --output "$STAGE10_DEST/projection"
"$STAGE10_PYTHON" report_stage10_projection.py --source "$STAGE10_DEST/projection" --output "$STAGE10_DEST/projection_report"
