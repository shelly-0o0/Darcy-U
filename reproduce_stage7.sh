#!/usr/bin/env bash
# Reproduce the fixed persistent-space hypothesis in a fresh CPU directory.
set -euo pipefail
cd "$(dirname "$0")"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
DARCY_PYTHON="${DARCY_PYTHON:-python}"
DARCY_STAGE7_ROOT="${DARCY_STAGE7_ROOT:-$PWD/reproductions/stage7_$(date +%Y%m%d_%H%M%S)}"
if ! command -v "$DARCY_PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $DARCY_PYTHON; set DARCY_PYTHON." >&2
    exit 1
fi
for required in persistent_recycling_experiment.py report_stage7.py requirements.txt \
    data/stage6_splits.json docs/stage7_protocol_zh.md; do
    if [ ! -f "$required" ]; then
        echo "Required reproduction input is missing: $required" >&2
        exit 1
    fi
done
if [ -e "$DARCY_STAGE7_ROOT" ] || [ -L "$DARCY_STAGE7_ROOT" ]; then
    echo "Refusing to overwrite $DARCY_STAGE7_ROOT" >&2
    exit 1
fi
mkdir -p -- "$(dirname "$DARCY_STAGE7_ROOT")"
mkdir -- "$DARCY_STAGE7_ROOT"
mkdir -- "$DARCY_STAGE7_ROOT/data" "$DARCY_STAGE7_ROOT/docs" "$DARCY_STAGE7_ROOT/logs"
cp ./*.py requirements.txt "$DARCY_STAGE7_ROOT/"
cp data/stage6_splits.json "$DARCY_STAGE7_ROOT/data/"
cp docs/stage7*.md "$DARCY_STAGE7_ROOT/docs/"
if [ -d "$PWD/vendor" ]; then
    ln -s "$PWD/vendor" "$DARCY_STAGE7_ROOT/vendor"
fi
cd "$DARCY_STAGE7_ROOT"
echo "Stage-7 reproduction directory: $PWD"
echo "CPU only. Run without concurrent training or benchmark workloads."
"$DARCY_PYTHON" -m unittest discover -s . -p '*test*.py' -v 2>&1 | tee logs/tests.log
if ! "$DARCY_PYTHON" -u persistent_recycling_experiment.py development 2>&1 | tee logs/development.log; then
    if [ -f results/stage7_development/complete.json ]; then
        "$DARCY_PYTHON" report_stage7.py 2>&1 | tee logs/report.log
    fi
    exit 1
fi
"$DARCY_PYTHON" -u persistent_recycling_experiment.py confirm 2>&1 | tee logs/confirm.log
"$DARCY_PYTHON" report_stage7.py 2>&1 | tee logs/report.log
