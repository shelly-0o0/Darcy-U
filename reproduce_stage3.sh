#!/usr/bin/env bash
# Reproduce in a fresh directory so committed results/weights stay unchanged.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
DARCY_PYTHON="${DARCY_PYTHON:-python}"
DARCY_STAGE3_ROOT="${DARCY_STAGE3_ROOT:-$PWD/reproductions/stage3_$(date +%Y%m%d_%H%M%S)}"
if [ -e "$DARCY_STAGE3_ROOT" ]; then
    echo "Refusing to overwrite $DARCY_STAGE3_ROOT" >&2
    exit 1
fi
mkdir -p "$DARCY_STAGE3_ROOT/data" "$DARCY_STAGE3_ROOT/docs" "$DARCY_STAGE3_ROOT/checkpoints"
cp ./*.py requirements.txt "$DARCY_STAGE3_ROOT/"
cp data/pilot.manifest.json "$DARCY_STAGE3_ROOT/data/"
cp docs/stage3*.md docs/sequence_analysis_zh.md "$DARCY_STAGE3_ROOT/docs/"
if [ -d "$PWD/vendor" ]; then
    ln -s "$PWD/vendor" "$DARCY_STAGE3_ROOT/vendor"
fi
cd "$DARCY_STAGE3_ROOT"
"$DARCY_PYTHON" -m unittest discover -s . -p '*test*.py' -v
for stage in screen data train_energy calibrate_initial train_energy_control train_mixed calibrate_optimized; do
    "$DARCY_PYTHON" -u sequence_experiment.py "$stage"
done
"$DARCY_PYTHON" freeze_stage3.py
"$DARCY_PYTHON" -u sequence_experiment.py test
"$DARCY_PYTHON" -u audit_stage3_references.py
"$DARCY_PYTHON" report_stage3.py
