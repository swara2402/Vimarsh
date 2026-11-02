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
