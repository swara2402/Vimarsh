# Containerized local VIMARSH demo

This deployment runs the FastAPI service and production-built React dashboard locally. It is a reproducible demo stack, not a public production deployment.

For a one-command macOS/Linux preparation and launch, run `bash scripts/vimarsh-demo.sh demo` from the repository root. The helper trains and verifies the local model bundle, validates Compose, starts both containers, and checks readiness. The `prepare` and `up` subcommands can also be run separately; see the [demo-day runbook](../docs/demo-day-runbook.md).

## 1. Generate the trusted model bundle

Follow [the demo-day runbook](../docs/demo-day-runbook.md) to install research dependencies and run:

```bash
cd research
python experiments/run_reliability_experiments.py
```

Confirm that this file exists on the host:

```text
research/models/reliability/wine_quality_reliability_bundle.joblib
```

The model directory is mounted read-only into the API container. The image does not contain model artifacts or benchmark datasets.

## 2. Configure optional research access

From the repository root:

```bash
cp deploy/.env.example .env
```

For the ordinary demo, leave `VIMARSH_RESEARCH_API_TOKEN` blank. If protected research endpoints are required, set a strong random token in this server-side environment file. Never add it to Vite variables or browser code. JSONL remains the default outcome ledger. To use transactional local SQLite storage in the container, set `VIMARSH_OUTCOME_LEDGER_BACKEND=sqlite` and `VIMARSH_OUTCOME_LEDGER=/app/research/data/action_outcomes.sqlite3` in `.env` before startup.

## 3. Build and start

For a guided preflight, build, startup, and readiness check, run:

```bash
bash scripts/vimarsh-demo.sh up
```

The helper requires the trusted model bundle to exist and prints service URLs only after the API schema and dashboard respond. Use `bash scripts/vimarsh-demo.sh status`, `logs`, or `down` to manage the local stack.

For manual startup, use:

```bash
docker compose -f compose.yaml up --build
```

Open:

- Dashboard: http://localhost:8080
- API health: http://localhost:8000/health
- API documentation: http://localhost:8000/docs

The dashboard container starts after the API health check succeeds. The API will return a clear 503 for model-backed routes if the trusted bundle is missing or invalid.

## 4. Stop

```bash
docker compose -f compose.yaml down
```

The named `vimarsh-outcomes` volume retains local research ledger data. To remove that local data as well:

```bash
docker compose -f compose.yaml down -v
```

## Operational boundaries

- The stack binds ports on the local host; do not expose it publicly without a security review.
- No TLS, identity provider, role management, rate limiting, or production secret manager is included.
- The JSONL research ledger remains single-writer local research storage.
- Human review and additional-evidence providers remain unavailable.
- The recommendation API does not execute actions. Alternate-model inference requires an explicit user action.
