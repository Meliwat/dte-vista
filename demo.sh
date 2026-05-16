#!/usr/bin/env bash
# VISTA - one-command offline demo.
# Deterministic, zero network, zero API keys. Produces:
#   output/dashboard.png   (the single map-based dashboard)
#   output/summary.json    (machine-readable metrics + provenance)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3.11}"
command -v "$PY" >/dev/null 2>&1 || PY="python3"

echo ">> VISTA demo - imagery-led predictive utility-pole risk (DTE)"
echo ">> interpreter: $($PY --version 2>&1)"

# Optional: create an isolated venv if VISTA_VENV=1 (default uses current env).
if [ "${VISTA_VENV:-0}" = "1" ]; then
  "$PY" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=python
  pip -q install -r requirements.txt
fi

"$PY" -m vista

echo ">> done. open output/app.html (interactive) or output/dashboard.png"
