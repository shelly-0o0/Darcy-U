#!/usr/bin/env bash
# Reproduce causal event-space exploration in an empty directory, on CPU.
set -euo pipefail
cd "$(dirname "$0")"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
DARCY_PYTHON="${DARCY_PYTHON:-python}"
DARCY_STAGE6_ROOT="${DARCY_STAGE6_ROOT:-$PWD/reproductions/stage6_$(date +%Y%m%d_%H%M%S)}"

if ! command -v "$DARCY_PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $DARCY_PYTHON; set DARCY_PYTHON." >&2
    exit 1
fi
for required in event_recycling_experiment.py report_stage6.py requirements.txt \
    data/stage4_splits.json data/stage5_splits.json docs/stage6_protocol_zh.md; do
    if [ ! -f "$required" ]; then
        echo "Required reproduction input is missing: $required" >&2
        exit 1
    fi
done
if [ -e "$DARCY_STAGE6_ROOT" ] || [ -L "$DARCY_STAGE6_ROOT" ]; then
    echo "Refusing to overwrite $DARCY_STAGE6_ROOT" >&2
    exit 1
fi
mkdir -p -- "$(dirname "$DARCY_STAGE6_ROOT")"
mkdir -- "$DARCY_STAGE6_ROOT"
mkdir -- "$DARCY_STAGE6_ROOT/data" "$DARCY_STAGE6_ROOT/docs" "$DARCY_STAGE6_ROOT/logs"
cp ./*.py requirements.txt "$DARCY_STAGE6_ROOT/"
cp data/stage4_splits.json data/stage5_splits.json "$DARCY_STAGE6_ROOT/data/"
cp docs/stage6*.md "$DARCY_STAGE6_ROOT/docs/"
if [ -d "$PWD/vendor" ]; then
    ln -s "$PWD/vendor" "$DARCY_STAGE6_ROOT/vendor"
fi

cd "$DARCY_STAGE6_ROOT"
echo "Stage-6 reproduction directory: $PWD"
echo "CPU only. Run without concurrent training or benchmark workloads."
"$DARCY_PYTHON" -m unittest discover -s . -p '*test*.py' -v 2>&1 | tee logs/tests.log
"$DARCY_PYTHON" -u event_recycling_experiment.py probe 2>&1 | tee logs/probe.log
if ! "$DARCY_PYTHON" -u event_recycling_experiment.py screen 2>&1 | tee logs/screen.log; then
    if [ -f results/stage6_screen/complete.json ]; then
        "$DARCY_PYTHON" report_stage6.py 2>&1 | tee logs/report.log
    fi
    exit 1
fi
"$DARCY_PYTHON" -u event_recycling_experiment.py confirm 2>&1 | tee logs/confirm.log
"$DARCY_PYTHON" report_stage6.py 2>&1 | tee logs/report.log
