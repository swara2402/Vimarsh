# VIMARSH

**Question. Evaluate. Verify.**

VIMARSH is a research-oriented framework for assessing the reliability of machine-learning predictions and selecting an appropriate verification action before a prediction is accepted.

> A prediction should earn the right to be trusted.

## Faculty demo: start here

**Use the dedicated [faculty-demo branch](https://github.com/swara2402/Vimarsh/tree/faculty-demo) for the integrated presentation build.** It combines the integrated research stack, operator workflows, and Data Analysis & Visualisation (DAV) workspace in one branch.

- [Faculty demo guide](docs/DEMO_START_HERE.md)
- [DAV sample CSV](demo-data/vimarsh_dav_sample.csv)
- [Demo runbook](docs/demo-day-runbook.md)
- [Five-minute presentation script](docs/demo-presentation-script.md)
- [Research evidence summary](docs/research-evidence-summary-2026-11-02.md)

### Quick start

Requirements: macOS/Linux, Python 3.10+, Docker Desktop with Docker Compose v2, and internet access on first run.

~~~bash
git clone https://github.com/swara2402/Vimarsh.git
cd Vimarsh
git switch faculty-demo
bash scripts/vimarsh-demo.sh demo
~~~

Open the dashboard at http://localhost:8080. API docs: http://localhost:8000/docs.

The demo helper prepares the Wine Quality reliability bundle and starts the local API/dashboard containers. The first run downloads the benchmark data and may take several minutes. Keep the terminal running during the presentation.

## What the integrated demo contains

1. **Decision workspace:** model prediction, reliability estimate, baseline risk, capability discovery, budget-aware recommendation and explicit alternate-model inference.
2. **Human-in-the-loop operations:** protected review queue, claim/complete/cancel lifecycle, hash-chained audit history and append-only evidence register.
3. **Evidence lookup:** Crossref DOI metadata resolution. Metadata lookup is not scientific-claim verification.
4. **Data Analysis & Visualisation:** browser-local CSV upload, profiling, histograms, category balance, missingness, descriptive statistics, Pearson correlations and saved policy comparison.
5. **Research evidence:** historical offline policy snapshots and multi-dataset research notes.

### Datasets

- **Wine Quality:** used by the one-command local demo training workflow; raw benchmark files are downloaded from UCI on first preparation if not already present.
- **Banknote Authentication and Ionosphere:** independent UCI benchmark adapters download their source files when those research experiments are run.
- **DAV sample CSV:** a small, explicitly synthetic teaching/demo file included in this branch so the visual analysis workspace can be explored immediately without downloading a dataset. It is not a benchmark result and must not be cited as one.
- **Saved policy snapshots:** committed historical JSON evaluation tables power the policy-comparison panel. They are offline results, not live retraining.

## Research direction

**Working title:** *VIMARSH: An Evidence-Driven Framework for Risk-Aware Verification of Machine Learning Predictions*

**Research question:** Can reliability signals guide safer verification choices while accounting for action cost, availability and risk?

The current serving policy is a transparent baseline. The contextual estimator and constrained allocator remain offline research components. VIMARSH is a research prototype, not a production safety system.

## Important interpretation boundaries

- Report coverage and error together. Covered-case error is not a full-population classifier error rate.
- Coverage-oriented actions include alternate-model and additional-evidence actions, not simply accepting every primary prediction.
- Historical evaluation snapshots are not fresh training, causal evidence, or real-world validation.
- Reviewer IDs are labels behind a shared token, not production-authenticated identities.
- Crossref returns bibliographic metadata; it does not validate scientific claims.
- CSVs uploaded to DAV are processed in browser memory and are not sent to FastAPI or used for training.
- CI/integration checks must be green before describing this branch as fully verified. The local runtime must be launched and smoke-tested on the presentation machine.

## Technology

Python, FastAPI, scikit-learn, Pandas, NumPy, React, TypeScript, Vite, Docker Compose, SQLite and GitHub Actions.

## Repository guide

- **backend/**: API, policy runtime, review queue and evidence register.
- **frontend/**: decision dashboard, operator workflows and DAV workspace.
- **research/**: benchmark adapters, training/evaluation experiments and model preparation.
- **results/verification/**: saved offline policy evaluation snapshots.
- **docs/**: architecture, evidence, runbook and presentation material.
- **demo-data/**: small synthetic CSV for DAV walkthrough.

## Current status

The repository contains baseline classification, reliability analysis, verification-action research code, a FastAPI recommendation service, and a React research dashboard. Milestone 3.2 adds outcome and execution-cost auditing. Milestone 4 adds a reliability-to-verification inference bridge, serializable model artifacts, duplicate-grouped train/validation/test and out-of-fold splitting, and an integrated policy-comparison runner.

The integrated verification-engine workflow remains research-stage and runs on Wine Quality, with separate grouped multi-seed policy-comparison tracks for MAGIC Gamma Telescope and HTRU2. These benchmarks use different scenario assumptions and evaluation designs, so their simulated net-benefit and weighted-loss values are reported separately rather than pooled. The experiments use simulated human review and assumed action-effectiveness factors; they do not establish causal real-world risk reduction or broad domain generalization. Wine and MAGIC now share a validation-selected nominal 75% acceptance diagnostic, while HTRU2 reports separate hard-cap, matched-capacity, and validation-selected review-rate diagnostics. Milestone 48 adds paired cluster-bootstrap 95% intervals for validation-selected coverage and accepted-error differences against confidence-only on Wine and MAGIC; these intervals are conditional on each fitted model and split. The API and React dashboard are implemented as a local research preview. The API runs a trusted local reliability bundle and returns recommendations only; the dashboard displays the prediction, estimated error probability, risk, action recommendation, and rationale. Milestone 74 adds a browser-local CSV exploratory-analysis workspace with dataset profiling, missingness, histograms, category balance, Pearson correlations, and descriptive statistics; it does not upload datasets or replace model evaluation. See [Milestone 74 DAV documentation](docs/milestone74-data-analysis-visualisation.md). Milestone 73 adds optional protected local operator workflows: a persistent human-review queue and append-only manual evidence intake. These are local operator workflows, not external reviewer identity integrations. Reviewer IDs are not authenticated accounts. Manual evidence is operator-submitted; a limited Crossref DOI connector retrieves bibliographic metadata only and does not verify scientific claims.

## Application implementation status

The FastAPI service in `backend/` exposes health, baseline decision, reliability schema, and reliability-driven decision endpoints. The React + TypeScript dashboard in `frontend/` connects to the API and renders the model feature schema and recommendation results. The reliability decision endpoint remains recommendation-only. The dashboard also supports an explicit local alternate-model inference action, showing each alternate prediction and measured inference latency without automatically replacing the primary prediction. The protected research API records validated action outcomes, estimates context-conditioned action benefit from completed propensity-logged records, and recommends batch allocations under shared budgets and action capacities. It is disabled until a server-side research token is configured; its allocation output never executes selected actions.

Milestone 63 hardens the protected API boundary with finite-number checks, bounded feature/context maps, non-negative costs and capacities, and duplicate/malformed batch rejection. Milestone 66 adds append-only linked completion events for pending action outcomes, preserving the original execution record and inherited propensity. Milestone 70 adds an optional transactional SQLite ledger and a verified, resumable migration from JSONL. See [Milestone 66 outcome lifecycle](research/docs/milestone66-linked-outcome-completion.md). See [Milestone 62 implementation notes](research/docs/milestone62-protected-outcome-allocation-api.md) and [Milestone 63 validation notes](research/docs/milestone63-strict-api-validation.md). Milestone 73 adds a protected local operator queue with claim, completion and cancellation, hash-chained review audit history, append-only evidence storage, evidence digest checks, and a credential-free Crossref DOI metadata connector. See [Milestone 73 operator workflow documentation](docs/milestone73-operator-workflows.md). External reviewer and evidence-acquisition providers, identity/roles, and real outcome collection remain future work. The contextual benefit lower-bound is a diagnostic, not a guaranteed confidence bound or causal estimate. Milestone 68 adds a local Docker Compose deployment with API health gating and production-built dashboard containers. See [container demo instructions](deploy/container-demo.md), [Backend API](backend/README.md), [Dashboard](frontend/README.md), and [Demo-day runbook](docs/demo-day-runbook.md). Milestone 67 adds live capability discovery to the dashboard and disables unavailable action paths; Milestone 69 adds diagnostics that explain estimator abstentions without weakening the decision gate. The [research evidence summary](docs/research-evidence-summary-2026-11-02.md) consolidates the multi-dataset, multi-seed findings, including negative and mixed results, and the [five-minute presentation script](docs/demo-presentation-script.md) gives a defensible demo narrative.

## Quick local demo

On macOS/Linux with Python 3.10+ and Docker Desktop installed, prepare the Wine Quality model bundle and launch the containerized dashboard with:

```bash
bash scripts/vimarsh-demo.sh demo
```

The first run downloads the benchmark data and trains the trusted bundle, so allow several minutes. The dashboard opens at http://localhost:8080. See the [demo-day runbook](docs/demo-day-runbook.md) for the manual setup, demo flow, troubleshooting, and research-claim boundaries.

## Getting started

1. Install the dependencies listed in [research/requirements.txt](research/requirements.txt).
2. Obtain the Wine Quality dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality) and place the source files under `research/datasets/raw/` using the filenames expected by the data loader.
3. From the `research/` directory, run `python experiments/run_reliability_experiments.py` to train the reliability estimator and write its inference bundle.
4. Run `python experiments/run_integrated_verification.py` to use estimated error probabilities for verification action selection and offline outcome evaluation.

See [Controlled shift robustness protocol](research/docs/controlled-shift-robustness.md), [Milestone 4 integration notes](research/docs/milestone4-1-integration.md), the [Cross-dataset evidence audit](research/docs/cross-dataset-evidence-audit.md), [Paired bootstrap uncertainty protocol](research/docs/paired-bootstrap-uncertainty.md), [Research Plan](docs/research-plan.md), and [Architecture](docs/architecture.md) for assumptions, limitations, and the proposed sequence of work.

## License

No license has been selected yet. Until one is added, the default copyright rules apply.