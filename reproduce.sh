#!/usr/bin/env bash
# Run from any directory, in a Python environment with requirements installed.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
DARCY_PYTHON="${DARCY_PYTHON:-python}"
"$DARCY_PYTHON" -m unittest discover -s . -p '*test*.py' -v
"$DARCY_PYTHON" run_experiment.py data --n 128 --steps 20 --train-sequences 48 --val-sequences 12
"$DARCY_PYTHON" run_experiment.py train --model-seeds 7 17 27 --epochs 50 --device cpu
"$DARCY_PYTHON" run_experiment.py benchmark --n 128 --steps 20 \
  --test-seeds 90001 90002 90003 --model-seeds 7 17 27 --repeats 3 \
  --device cpu --run-name reproduced_cpu
"$DARCY_PYTHON" report_results.py results/reproduced_cpu
