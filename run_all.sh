#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python}
"$PYTHON" -m pip install -r requirements.txt
"$PYTHON" -m src.check_api
"$PYTHON" -m pytest -q
"$PYTHON" -m src.memory_plot
"$PYTHON" -m src.ghz_benchmark --max-qubits 30
"$PYTHON" -m src.h2_vqe
"$PYTHON" -m src.build_readme
