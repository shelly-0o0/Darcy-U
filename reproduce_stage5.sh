#!/usr/bin/env bash
# Reproduce stage 5 in a fresh directory. CPU only; no model weights are used.
set -euo pipefail
cd "$(dirname "$0")"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
DARCY_PYTHON="${DARCY_PYTHON:-python}"
DARCY_STAGE5_ROOT="${DARCY_STAGE5_ROOT:-$PWD/reproductions/stage5_$(date +%Y%m%d_%H%M%S)}"

if ! command -v "$DARCY_PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $DARCY_PYTHON; set DARCY_PYTHON." >&2
    exit 1
fi
for required in cached_pcg.py cached_recycling_experiment.py report_stage5.py \
    requirements.txt data/stage3_splits.json data/stage4_splits.json docs/stage5_protocol_zh.md; do
    if [ ! -f "$required" ]; then
        echo "Required reproduction input is missing: $required" >&2
        exit 1
    fi
done
if [ -e "$DARCY_STAGE5_ROOT" ] || [ -L "$DARCY_STAGE5_ROOT" ]; then
    echo "Refusing to overwrite $DARCY_STAGE5_ROOT" >&2
    exit 1
fi
mkdir -p -- "$(dirname "$DARCY_STAGE5_ROOT")"
mkdir -- "$DARCY_STAGE5_ROOT"
mkdir -- "$DARCY_STAGE5_ROOT/data" "$DARCY_STAGE5_ROOT/docs" "$DARCY_STAGE5_ROOT/logs"
cp ./*.py requirements.txt "$DARCY_STAGE5_ROOT/"
cp data/stage3_splits.json data/stage4_splits.json "$DARCY_STAGE5_ROOT/data/"
cp docs/stage5*.md "$DARCY_STAGE5_ROOT/docs/"
if [ -d "$PWD/vendor" ]; then
    ln -s "$PWD/vendor" "$DARCY_STAGE5_ROOT/vendor"
fi

cd "$DARCY_STAGE5_ROOT"
echo "Stage-5 reproduction directory: $PWD"
echo "CPU only. Run without concurrent training or benchmark workloads."
"$DARCY_PYTHON" -m unittest discover -s . -p '*test*.py' -v 2>&1 | tee logs/tests.log
# A failed prespecified audit or screen remains a failure. Generate its report
# if enough records exist, then stop; do not change thresholds or enter confirm.
if ! "$DARCY_PYTHON" -u cached_recycling_experiment.py audit 2>&1 | tee logs/audit.log; then
    if [ -f results/stage5_cache_audit/complete.json ]; then
        "$DARCY_PYTHON" report_stage5.py 2>&1 | tee logs/report.log
    fi
    exit 1
fi
if ! "$DARCY_PYTHON" -u cached_recycling_experiment.py screen 2>&1 | tee logs/screen.log; then
    if [ -f results/stage5_screen/complete.json ]; then
        "$DARCY_PYTHON" report_stage5.py 2>&1 | tee logs/report.log
    fi
    exit 1
fi
"$DARCY_PYTHON" -u cached_recycling_experiment.py confirm 2>&1 | tee logs/confirm.log
"$DARCY_PYTHON" report_stage5.py 2>&1 | tee logs/report.log
