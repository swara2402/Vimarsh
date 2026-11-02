#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE=(docker compose -f "$ROOT/compose.yaml")
MODEL_DIR="${VIMARSH_MODEL_DIR:-}"

# Docker Compose reads .env itself. Mirror its simple MODEL_DIR setting for the
# preflight check without sourcing/evaluating arbitrary shell from that file.
if [[ -z "$MODEL_DIR" && -f "$ROOT/.env" ]]; then
  MODEL_DIR="$(sed -n 's/^VIMARSH_MODEL_DIR=//p' "$ROOT/.env" | tail -n 1)"
fi
MODEL_DIR="${MODEL_DIR:-./research/models/reliability}"
if [[ "$MODEL_DIR" != /* ]]; then MODEL_DIR="$ROOT/$MODEL_DIR"; fi
BUNDLE="$MODEL_DIR/wine_quality_reliability_bundle.joblib"

usage() {
  echo "Usage: bash scripts/vimarsh-demo.sh {up|down|status|logs}"
}

command_name="${1:-up}"
case "$command_name" in
  up)
    command -v docker >/dev/null 2>&1 || { echo "Docker is required. Install/start Docker Desktop first." >&2; exit 1; }
    docker compose version >/dev/null 2>&1 || { echo "Docker Compose v2 is required." >&2; exit 1; }
    [[ -f "$BUNDLE" ]] || {
      echo "Trusted model bundle not found: $BUNDLE" >&2
      echo "Generate it using the preparation steps in docs/demo-day-runbook.md." >&2
      exit 1
    }
    if [[ ! -f "$ROOT/.env" ]]; then
      cp "$ROOT/deploy/.env.example" "$ROOT/.env"
      echo "Created .env from deploy/.env.example. Review it before enabling protected research routes."
    fi
    cd "$ROOT"
    "${COMPOSE[@]}" config --quiet
    "${COMPOSE[@]}" up --build -d
    echo "Waiting for API and model schema..."
    ready=false
    for attempt in $(seq 1 45); do
      if curl --silent --show-error --fail http://localhost:8000/health >/dev/null 2>&1 \
        && curl --silent --show-error --fail http://localhost:8000/v1/reliability/schema >/dev/null 2>&1 \
        && curl --silent --show-error --fail http://localhost:8080/ >/dev/null 2>&1; then
        ready=true
        break
      fi
      sleep 2
    done
    if [[ "$ready" != true ]]; then
      echo "Demo stack did not become ready. Recent container logs:" >&2
      "${COMPOSE[@]}" logs --tail=80 >&2
      exit 1
    fi
    echo "VIMARSH demo is ready:"
    echo "  Dashboard: http://localhost:8080"
    echo "  API docs:  http://localhost:8000/docs"
    ;;
  down)
    cd "$ROOT"
    "${COMPOSE[@]}" down
    ;;
  status)
    cd "$ROOT"
    "${COMPOSE[@]}" ps
    ;;
  logs)
    cd "$ROOT"
    "${COMPOSE[@]}" logs -f
    ;;
  *)
    usage
    exit 2
    ;;
esac
