#!/usr/bin/env bash
# Run the prespecified final budget screen, stopping when its gate fails.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
DARCY_PYTHON="${DARCY_PYTHON:-python}"
DARCY_STAGE8_ROOT="${DARCY_STAGE8_ROOT:-$PWD/reproductions/stage8_$(date +%Y%m%d_%H%M%S)}"
if ! command -v "$DARCY_PYTHON" >/dev/null 2>&1; then
    echo "Set DARCY_PYTHON to an interpreter with requirements.txt installed." >&2
    exit 1
fi
if [ -e "$DARCY_STAGE8_ROOT" ] || [ -L "$DARCY_STAGE8_ROOT" ]; then
    echo "Refusing to overwrite $DARCY_STAGE8_ROOT" >&2
    exit 1
fi
mkdir -p -- "$(dirname "$DARCY_STAGE8_ROOT")"
mkdir -- "$DARCY_STAGE8_ROOT"
mkdir -- "$DARCY_STAGE8_ROOT/data" "$DARCY_STAGE8_ROOT/docs" "$DARCY_STAGE8_ROOT/logs"
cp ./*.py requirements.txt "$DARCY_STAGE8_ROOT/"
cp data/stage7_splits.json "$DARCY_STAGE8_ROOT/data/"
cp docs/stage8*.md "$DARCY_STAGE8_ROOT/docs/"
cp docs/route_retrospective_zh.md "$DARCY_STAGE8_ROOT/docs/"
if [ -d "$PWD/vendor" ]; then
    ln -s "$PWD/vendor" "$DARCY_STAGE8_ROOT/vendor"
fi
cd "$DARCY_STAGE8_ROOT"
echo "Stage-8 directory: $PWD; CPU only, no concurrent benchmark workloads."
"$DARCY_PYTHON" -m unittest discover -v 2>&1 | tee logs/tests.log
"$DARCY_PYTHON" -u oracle_budget_experiment.py development 2>&1 | tee logs/development.log
if "$DARCY_PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open("results/stage8_decision.json"))["proceed_to_confirmation"] else 1)'; then
    "$DARCY_PYTHON" -u oracle_budget_experiment.py confirm 2>&1 | tee logs/confirm.log
else
    echo "Prespecified development gate failed; confirmation and training stopped."
fi
"$DARCY_PYTHON" report_stage8.py 2>&1 | tee logs/report.log
