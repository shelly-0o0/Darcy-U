#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
stage15_python="${DARCY_PYTHON:-python3}"
stage15_destination="${1:-reproductions/stage15-$(date -u +%Y%m%dT%H%M%SZ)}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
if [[ ! -f vendor/skfem/quadrature.py ]]; then
  echo 'Stage 15 records library source hashes from vendor/skfem. Install this local dependency first:' >&2
  echo '  <your-python> -m pip install --no-deps --target vendor scikit-fem==12.0.2' >&2
  exit 2
fi
"$stage15_python" -m unittest test_stage15_numerics -v
"$stage15_python" stage15_experiment.py --output "$stage15_destination/raw"
"$stage15_python" report_stage15.py --input "$stage15_destination/raw" --output "$stage15_destination/report"
"$stage15_python" verify_stage15.py --input "$stage15_destination/raw" --report "$stage15_destination/report" --output "$stage15_destination/verification.json"
