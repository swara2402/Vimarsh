# VIMARSH FastAPI backend

The repository includes a research-stage FastAPI service in `backend/app/`.

## Run locally

From the repository root:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export VIMARSH_RELIABILITY_BUNDLE="$(cd ../research && pwd)/models/reliability/wine_quality_reliability_bundle.joblib"
uvicorn app.main:app --reload
```

Generate the bundle first by following [the demo runbook](../docs/demo-day-runbook.md). The API listens on `http://localhost:8000`; OpenAPI docs are at `/docs`.

## Endpoints

- `GET /health`: service status and recommendation-only mode.
- `POST /v1/decisions`: accepts an externally supplied prediction, confidence, estimated error probability and decision constraints.
- `GET /v1/reliability/schema`: returns the feature contract from the configured version-2 bundle.
- `POST /v1/reliability/decisions`: runs the trusted local bundle on raw feature values, estimates error probability and returns a constrained recommendation.

## Current behavior and limitations

- The decision policy is a transparent baseline: estimated risk is `estimated_error_probability * harm`.
- The default acceptance threshold is configurable with `VIMARSH_MAX_ACCEPTABLE_RISK`.
- The only candidate action currently eligible for recommendation is an alternate model, and only when it is listed as available and its supplied estimated cost fits the remaining budget.
- Human review and additional evidence are explicitly excluded until real operational integrations exist.
- The service returns recommendations only. It does not execute actions, collect ground truth, or establish deployment safety.
- The bundle is a trusted local joblib artifact. Never load arbitrary user-uploaded joblib files.

## Configuration

- `VIMARSH_RELIABILITY_BUNDLE`: path to the trusted inference bundle.
- `VIMARSH_MAX_ACCEPTABLE_RISK`: non-negative risk threshold (default `0.5`).
- `VIMARSH_CORS_ORIGINS`: comma-separated exact browser origins (default `http://localhost:5173`). Wildcard origins are rejected.

## Tests

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The API/runtime tests are also run by the Application checks GitHub Actions workflow.
