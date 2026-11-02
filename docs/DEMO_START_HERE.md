# VIMARSH faculty demo guide

This page is the presentation entry point for the integrated faculty-demo branch.

## What is included

| Area | What to show |
|---|---|
| Decision workspace | Wine Quality example, prediction, estimated error, risk, capability panel, budget-aware recommendation |
| Alternate model | Explicitly run the available alternate model and show that the primary prediction is not silently replaced |
| Human review | Prepare a case, submit it, claim it, complete/cancel it and inspect event history |
| Evidence register | Add manual evidence or resolve a DOI; explain digest verification and metadata-only Crossref behavior |
| DAV workspace | Upload the bundled CSV, inspect profile, missing values, histogram, category distribution, correlation and descriptive statistics |
| Policy comparison | Switch among conservative, balanced and coverage-oriented historical snapshots |

## 1. Start the application

Requirements: macOS/Linux, Python 3.10+, Docker Desktop with Docker Compose v2, internet connection on first setup.

~~~bash
git clone https://github.com/swara2402/Vimarsh.git
cd Vimarsh
git switch faculty-demo
bash scripts/vimarsh-demo.sh demo
~~~

Wait for the script to report that the API and dashboard are ready.

- Dashboard: http://localhost:8080
- API health: http://localhost:8000/health
- Interactive API docs: http://localhost:8000/docs

The first run creates an isolated .venv-demo, installs minimal training dependencies, downloads the Wine Quality benchmark if needed, trains the local reliability bundle, builds the containers and waits for health checks. This can take several minutes. Do not close the terminal while presenting.

## 2. Recommended 7-minute walkthrough

1. **Problem (45 sec):** confidence alone is not a verification strategy. VIMARSH asks whether to accept, verify further, request review or abstain.
2. **Decision workspace (2 min):** click **Load example wine sample**, set harm to 8 and budget to 2, evaluate, and explain prediction, confidence, estimated error, baseline risk and recommendation.
3. **Feasibility (45 sec):** reduce the budget and evaluate again. Explain why an unavailable/unaffordable action must not be recommended.
4. **Alternate model (45 sec):** if the capability panel reports availability, explicitly run alternate models. The primary prediction is not automatically replaced.
5. **DAV (90 sec):** open Data Analysis & Visualisation and upload demo-data/vimarsh_dav_sample.csv. Explore row/column counts, missingness, numeric histograms, category balance, descriptive statistics and correlation matrix. The file is synthetic and intended only to demonstrate the explorer.
6. **Policy comparison (45 sec):** switch the three saved historical policy snapshots. Explain coverage and error among covered cases together.
7. **Human review (1 min):** if tokens are configured, prepare and submit a case, claim it, resolve it and inspect the audit history. If tokens are not configured, show the UI and explain that the operator workspace is protected; do not invent credentials.

## 3. Dataset inventory and provenance

- **Wine Quality:** real UCI benchmark used by the local demo training pipeline. Raw files are downloaded on first preparation by the existing data loader; they are not bundled in Git.
- **Banknote Authentication / Ionosphere:** real UCI datasets supported by independent benchmark adapters; downloaded when those research experiments run.
- **DAV teaching sample:** demo-data/vimarsh_dav_sample.csv is synthetic, includes a few intentional missing cells and categorical/numeric columns, and is safe to upload to DAV. It is not a real benchmark dataset.
- **Policy snapshots:** committed JSON tables under results/verification/; historical offline evaluation data, not fresh model training.

## 4. Optional operator credentials

The review and evidence APIs are token-protected. Configure distinct local values for the reviewer and evidence tokens in the root .env file before the presentation if you intend to submit cases or evidence. Never commit real tokens. See backend/.env.example and docs/milestone73-operator-workflows.md for the supported variables and workflow.

If you do not configure these optional tokens, demonstrate the decision workspace and DAV without attempting protected write operations.

## 5. Honest research boundaries

- The live recommendation path uses a transparent baseline policy. Contextual action-benefit estimation and constrained allocation remain offline research components.
- The historical policy table is retrospective and not causal evidence. Covered-case error is calculated only over cases covered by each policy.
- Coverage-oriented actions include alternate-model and additional-evidence actions, not just accepted primary predictions.
- Reviewer labels are not individual authenticated accounts. Crossref supplies DOI metadata only.
- DAV parses CSV files in browser memory; it does not upload them to the API or use them for model training.
- This local Compose demo is not a production deployment or real-world validation.

## 6. Shutdown

~~~bash
bash scripts/vimarsh-demo.sh status
bash scripts/vimarsh-demo.sh logs
bash scripts/vimarsh-demo.sh down
~~~

See [the full runbook](demo-day-runbook.md) for manual startup and troubleshooting, and [the presentation script](demo-presentation-script.md) for a concise spoken walkthrough.
