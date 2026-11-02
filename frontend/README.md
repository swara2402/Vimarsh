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

The dashboard also reads `GET /v1/actions/capabilities` and displays which action paths are configured. The alternate-model option is disabled when the trusted bundle contains no non-primary model. When reviewer and evidence tokens are configured, the dashboard exposes a separate local operator workspace for review queue management, claim/completion/cancellation, audit history, manual evidence intake, Crossref DOI metadata lookup, and evidence digest inspection. A decision result can be prepared as a review case with its prediction, rationale, confidence, estimated error probability, and baseline risk prefilled; the operator must still confirm and submit it. These are shared-token local workflows, not individual reviewer accounts or independent scientific-claim verification. An explicit user action can run alternate models already present in the trusted bundle; their outputs are displayed as evidence and never silently replace the primary prediction. The dashboard does not accept ground-truth labels or precomputed reliability scores. Recommendations are not action execution. This is a research preview, not a validated decision-support product.

See [Milestone 67 capability-aware dashboard notes](../research/docs/milestone67-capability-aware-dashboard.md).


## Data Analysis & Visualisation

The dashboard includes a browser-local CSV explorer. Upload a comma-separated UTF-8 file (header row, unique column names, up to 15 MB) in the **Data Analysis & Visualisation** section to inspect dataset dimensions, missingness, numeric histograms, category/class balance, a Pearson correlation matrix, and descriptive statistics. Feature and category selectors update the charts interactively. The file is analyzed in browser memory and is not uploaded to the API or used to train a model.

This is exploratory data analysis, not model evaluation. It does not produce confusion matrices, classification scores, ROC curves, calibration plots, or drift metrics without aligned predictions, labels, or reference data. See [Milestone 74 documentation](../docs/milestone74-data-analysis-visualisation.md).
