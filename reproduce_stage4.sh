#!/usr/bin/env bash
# Reproduce stage 4 with the frozen stage-3 model in a fresh directory.
# No existing results, weights, or stage-4 split files are overwritten.
set -euo pipefail
cd "$(dirname "$0")"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
DARCY_PYTHON="${DARCY_PYTHON:-python}"
DARCY_STAGE4_ROOT="${DARCY_STAGE4_ROOT:-$PWD/reproductions/stage4_$(date +%Y%m%d_%H%M%S)}"

if ! command -v "$DARCY_PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $DARCY_PYTHON; set DARCY_PYTHON." >&2
    exit 1
fi
for required in mechanism_experiment.py report_stage4.py \
    docs/stage3_protocol_zh.md docs/stage4_protocol_zh.md \
    data/stage3_splits.json checkpoints/correction_mixed.pt; do
    if [ ! -f "$required" ]; then
        echo "Required reproduction input is missing: $required" >&2
        exit 1
    fi
done
if [ -e "$DARCY_STAGE4_ROOT" ] || [ -L "$DARCY_STAGE4_ROOT" ]; then
    echo "Refusing to overwrite $DARCY_STAGE4_ROOT" >&2
    exit 1
fi

# The final mkdir also rejects a destination created after the check above.
mkdir -p -- "$(dirname "$DARCY_STAGE4_ROOT")"
mkdir -- "$DARCY_STAGE4_ROOT"
mkdir -- "$DARCY_STAGE4_ROOT/data" "$DARCY_STAGE4_ROOT/docs" \
    "$DARCY_STAGE4_ROOT/checkpoints" "$DARCY_STAGE4_ROOT/logs"
cp ./*.py requirements.txt "$DARCY_STAGE4_ROOT/"
cp data/stage3_splits.json "$DARCY_STAGE4_ROOT/data/"
cp docs/stage3*.md docs/stage4*.md docs/sequence_analysis_zh.md "$DARCY_STAGE4_ROOT/docs/"
cp checkpoints/correction_mixed.pt "$DARCY_STAGE4_ROOT/checkpoints/"
if [ -d "$PWD/vendor" ]; then
    ln -s "$PWD/vendor" "$DARCY_STAGE4_ROOT/vendor"
fi

cd "$DARCY_STAGE4_ROOT"
echo "Stage-4 reproduction directory: $PWD"
echo "Using the frozen model; CUDA is required for the prescribed inference."
# Discover the complete numerical suite (59 tests at the stage-4 freeze).
"$DARCY_PYTHON" -m unittest discover -s . -p '*test*.py' -v 2>&1 | tee logs/tests.log
for stage in diagnose screen confirm; do
    "$DARCY_PYTHON" -u mechanism_experiment.py "$stage" 2>&1 | tee "logs/$stage.log"
done
"$DARCY_PYTHON" report_stage4.py 2>&1 | tee logs/report.log
