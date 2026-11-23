#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
stage15c_python="${DARCY_PYTHON:-python3}"
stage15c_destination="${1:-reproductions/stage15c-$(date -u +%Y%m%dT%H%M%SZ)}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/darcy-u-mpl}"
if [[ ! -f vendor/skfem/quadrature.py ]]; then
  echo 'Install the pinned local dependency with this Python:' >&2
  echo '  <your-python> -m pip install --no-deps --target vendor scikit-fem==12.0.2' >&2
  exit 2
fi
"$stage15c_python" -m unittest discover -p 'test*.py' -v
"$stage15c_python" stage15c_experiment.py --output "$stage15c_destination/raw"
"$stage15c_python" report_stage15c.py --input "$stage15c_destination/raw" --output "$stage15c_destination/report"
"$stage15c_python" verify_stage15c.py --input "$stage15c_destination/raw" --report "$stage15c_destination/report" --output "$stage15c_destination/verification.json"
"$stage15c_python" analyze_stage15c_mechanisms.py --input "$stage15c_destination/raw" --summary "$stage15c_destination/report/summary.json" --output "$stage15c_destination/mechanisms.json"
"$stage15c_python" plot_stage15c_geometries.py --input "$stage15c_destination/raw" --output "$stage15c_destination/geometries.png"
