# VIMARSH demonstration runbook

This guide is for a reproducible local research demonstration. It uses the Wine Quality benchmark and a locally generated, trusted model bundle. It does not represent a production deployment or real-world validation.

## What to demonstrate

1. Explain the pipeline: base prediction -> reliability estimate -> risk = estimated error probability x harm -> hard action/budget constraints -> recommendation or abstention.
2. Show the React dashboard loading the feature schema from the running API.
3. Point out the live capability panel and explain that unavailable human-review/evidence paths are deliberately disabled.
4. Click **Load example wine sample** to populate a real red-wine feature row from the benchmark, then set the harm score and remaining budget and evaluate.
5. Explain the prediction, confidence, estimated error probability, baseline risk, recommended action, and rationale.
6. Change the budget or disable alternate-model availability to show how feasibility constraints change the recommendation.
7. Point out that recommendations are not executed. Human review and evidence acquisition are not operational integrations.
8. Use the [research evidence summary](research-evidence-summary-2026-11-02.md) for benchmark results and limitations.

## Optional containerized startup

If Docker is available, use the [containerized demo guide](../deploy/container-demo.md) to build and start the API and production dashboard with one Compose command. The trusted model bundle must still be generated locally first. This is a local demo stack, not a public production deployment.

## One-command container demo

After the trusted model bundle has been generated, start the local demo stack from the repository root:

```bash
bash scripts/vimarsh-demo.sh up
```

The helper checks Docker Compose and the bundle path, validates the Compose configuration, builds and starts both containers, then waits for the API model schema and dashboard to respond. It prints recent container logs if startup fails.

Useful follow-up commands:

```bash
bash scripts/vimarsh-demo.sh status
bash scripts/vimarsh-demo.sh logs
bash scripts/vimarsh-demo.sh down
```

The equivalent manual steps are documented in [the container demo guide](../deploy/container-demo.md).

## Prepare the model bundle

From the repository root:

```bash
cd research
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/run_reliability_experiments.py
```

The data loader downloads the red and white Wine Quality files from UCI if they are not already present under `research/datasets/raw/`. The reliability experiment writes the inference bundle to:

```
research/models/reliability/wine_quality_reliability_bundle.joblib
```

Allow the training experiment to finish before starting the API. Keep the generated bundle local and trusted; do not accept uploaded joblib artifacts from users.

## Start the API

Open a second terminal from the repository root:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set the bundle path to the absolute path of the artifact created above.

macOS/Linux:

```bash
export VIMARSH_RELIABILITY_BUNDLE="$(cd ../research && pwd)/models/reliability/wine_quality_reliability_bundle.joblib"
uvicorn app.main:app --reload
```

The API health check is `http://localhost:8000/health`; interactive API documentation is `http://localhost:8000/docs`.

## Start the dashboard

Open a third terminal from the repository root:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173`. The default API origin is `http://localhost:8000`; override it with `VITE_API_BASE_URL` if needed.

## Demo checks

- API health endpoint returns `status: ok`.
- `GET /v1/reliability/schema` returns the fitted feature names and base-model count.
- Dashboard status changes to `MODEL CONNECTED`.
- Submit all numeric features and verify that the response includes prediction, confidence, estimated error probability, baseline risk and rationale.
- Reduce remaining budget below the alternate-model cost. If risk is above the acceptance threshold, the policy should abstain rather than recommend an infeasible action.
- Confirm the UI labels the output as recommendation-only and not executed.

## Troubleshooting

- **Model offline / HTTP 503:** confirm the reliability experiment completed and `VIMARSH_RELIABILITY_BUNDLE` points to the exact bundle file.
- **Feature schema mismatch:** enter every field rendered by the dashboard; do not substitute wine-quality labels or omit the wine-type feature.
- **CORS error:** keep the dashboard on `http://localhost:5173`, or set `VIMARSH_CORS_ORIGINS` to the exact dashboard origin before starting the API. Wildcard origins are rejected.
- **Missing packages:** activate the virtual environment for the relevant service and install its requirements.

## Claims to make accurately

- The Wine Quality benchmark is an offline tabular classification experiment.
- The reliability estimator is trained from out-of-fold model prediction features.
- The current API uses a transparent baseline policy for recommendations and does not execute actions.
- The action-outcome, contextual-benefit and constrained-allocation work is research-stage. Any simulated outcomes or offline paired benchmark results must be described as such.
- Do not claim causal real-world risk reduction, operational human review, evidence-provider integration, production readiness, or patentability from this prototype.
