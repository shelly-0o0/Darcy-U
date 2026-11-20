#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
stage15b_python="${DARCY_PYTHON:-python3}"
stage15b_destination="${1:-reproductions/stage15b-$(date -u +%Y%m%dT%H%M%SZ)}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
if [[ ! -f vendor/skfem/quadrature.py ]]; then
  echo 'Install the pinned local dependency first:' >&2
  echo '  <your-python> -m pip install --no-deps --target vendor scikit-fem==12.0.2' >&2
  exit 2
fi
"$stage15b_python" -m unittest test_stage15b -v
"$stage15b_python" stage15b_experiment.py --output "$stage15b_destination/raw"
"$stage15b_python" report_stage15b.py --input "$stage15b_destination/raw" --output "$stage15b_destination/report"
"$stage15b_python" verify_stage15b.py --input "$stage15b_destination/raw" --report "$stage15b_destination/report" --output "$stage15b_destination/verification.json"
