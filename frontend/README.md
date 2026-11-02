# VIMARSH dashboard

React + TypeScript + Vite dashboard for the research-stage reliability decision API.

## Run

```bash
npm install
npm run dev
```

By default, the UI calls `http://localhost:8000`. Set `VITE_API_BASE_URL` to override the API origin.

## API prerequisites

Start the FastAPI service and set `VIMARSH_RELIABILITY_BUNDLE` to a trusted version-2 Wine Quality reliability bundle. The UI retrieves the feature schema from `GET /v1/reliability/schema`, renders one numeric field per trained input feature, and submits raw feature values to `POST /v1/reliability/decisions`.

The dashboard also reads `GET /v1/actions/capabilities` and displays which action paths are configured. The alternate-model option is disabled when the trusted bundle contains no non-primary model. Human review and additional evidence remain clearly marked unavailable. An explicit user action can run alternate models already present in the trusted bundle; their outputs are displayed as evidence and never silently replace the primary prediction. The dashboard does not accept ground-truth labels or precomputed reliability scores. Recommendations are not action execution. This is a research preview, not a validated decision-support product.

See [Milestone 67 capability-aware dashboard notes](../research/docs/milestone67-capability-aware-dashboard.md).
