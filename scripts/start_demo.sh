#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VIMARSH_DEMO_VENV:-$ROOT_DIR/.venv-demo}"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--prepare-only" ) ]]; then
  echo "Usage: bash scripts/start_demo.sh [--prepare-only]" >&2
  exit 2
fi
BUNDLE="$ROOT_DIR/research/models/reliability/wine_quality_reliability_bundle.joblib"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3 is required. Install it or set PYTHON_BIN to its executable." >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/research/requirements-demo.txt"

echo "Preparing the trusted Wine Quality reliability bundle..."
(
  cd "$ROOT_DIR/research"
  "$VENV_DIR/bin/python" experiments/run_reliability_experiments.py
)

if [[ ! -s "$BUNDLE" ]]; then
  echo "Training finished without creating the expected bundle: $BUNDLE" >&2
  exit 1
fi

echo "Model bundle ready: $BUNDLE"

if [[ "${1:-}" == "--prepare-only" ]]; then
  echo "Preparation complete. Start the local API/dashboard using the demo runbook."
  exit 0
fi

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose is required to launch the containerized demo." >&2
  echo "The model bundle is ready. Install Docker Desktop, then run:" >&2
  echo "  docker compose -f compose.yaml up --build" >&2
  exit 1
fi

cd "$ROOT_DIR"
docker compose -f compose.yaml up --build
