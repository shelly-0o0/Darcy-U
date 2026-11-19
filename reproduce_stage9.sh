#!/usr/bin/env bash
# Reproduce the bounded nonlinear opportunity screen in a fresh directory.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
DARCY_PYTHON="${DARCY_PYTHON:-python}"
DARCY_STAGE9_ROOT="${DARCY_STAGE9_ROOT:-$PWD/reproductions/stage9_$(date +%Y%m%d_%H%M%S)}"
if [ -e "$DARCY_STAGE9_ROOT" ] || [ -L "$DARCY_STAGE9_ROOT" ]; then
    echo "Refusing to overwrite $DARCY_STAGE9_ROOT" >&2
    exit 1
fi
mkdir -p -- "$(dirname "$DARCY_STAGE9_ROOT")"
mkdir -- "$DARCY_STAGE9_ROOT"
mkdir -- "$DARCY_STAGE9_ROOT/docs" "$DARCY_STAGE9_ROOT/logs"
cp nonlinear_budget.py nonlinear_screen_experiment.py report_stage9.py test_nonlinear_budget.py test_nonlinear_screen.py requirements.txt "$DARCY_STAGE9_ROOT/"
cp docs/stage9*.md "$DARCY_STAGE9_ROOT/docs/"
cd "$DARCY_STAGE9_ROOT"
"$DARCY_PYTHON" -m unittest -v test_nonlinear_budget test_nonlinear_screen 2>&1 | tee logs/tests.log
"$DARCY_PYTHON" -u nonlinear_screen_experiment.py run 2>&1 | tee logs/screen.log
"$DARCY_PYTHON" report_stage9.py 2>&1 | tee logs/report.log
