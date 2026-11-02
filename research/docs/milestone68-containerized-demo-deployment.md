# Milestone 68: Reproducible containerized demo deployment

## Delivered implementation

- Added a FastAPI container image that includes backend code and the research modules required by the protected API.
- Added a production-built React dashboard image served by Nginx with SPA fallback routing.
- Added a Compose stack with API health gating, explicit local ports, read-only trusted model-bundle mount, and a persistent named volume for the outcome ledger.
- Added a safe environment template for optional server-side research token configuration.
- Excluded local datasets, trained artifacts, secrets, and frontend build output from Docker build contexts.
- Added a CI workflow that validates Compose configuration and builds both images on relevant pull requests.
- Added operator instructions for model preparation, startup, shutdown, and limitations.

## Run locally

1. Generate `research/models/reliability/wine_quality_reliability_bundle.joblib` using the existing research runbook.
2. Copy `deploy/.env.example` to `.env`; configure the protected research token only if needed.
3. Run `docker compose -f compose.yaml up --build`.
4. Open the dashboard at `http://localhost:8080` and the API health endpoint at `http://localhost:8000/health`.

## Boundaries

This is a local reproducible demo deployment, not a public production deployment. It does not include TLS, user identity, role-based access control, rate limiting, a production secret manager, transactional multi-worker ledger storage, or operational human-review/evidence providers. Do not expose the ports publicly without security review.
