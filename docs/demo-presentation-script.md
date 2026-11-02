# VIMARSH: five-minute project demonstration script

## 0:00–0:40 | Problem

Machine-learning systems often return a prediction with a confidence score, but confidence alone does not tell us whether a prediction is reliable enough for a consequential decision. VIMARSH explores a second question: when should a prediction be accepted, checked further, sent into a controlled review workflow, or withheld?

## 0:40–1:20 | Proposed approach

Walk through the architecture:

1. A base model produces a prediction.
2. A reliability estimator analyzes prediction-derived signals and estimates the probability of error.
3. The estimated error probability is combined with a supplied harm score to form a baseline risk estimate.
4. A transparent policy applies hard availability and budget constraints.
5. The system recommends acceptance, an eligible alternate-model check, or abstention.
6. Where an operator explicitly chooses to do so, a decision can be prepared as a human-review case. The handoff is not automatic.

Emphasize that the main decision API is recommendation-only. It does not execute the recommended action.

## 1:20–3:30 | Live dashboard

1. Open the React dashboard and wait for the model schema to load.
2. Explain that the feature fields are retrieved from the trained bundle, not hard-coded into the page.
3. Point out the live capability panel and explain that unavailable actions fail closed rather than being presented as active.
4. Click **Load example wine sample** to populate a complete red-wine benchmark row.
5. Set harm to 8, keep the remaining budget at 2, and leave the alternate-model option enabled with its default estimated cost of 1.
6. Select **Evaluate reliability**. Explain the primary prediction, confidence, estimated error probability, baseline risk (error probability multiplied by harm), recommendation, and rationale.
7. If the capability panel reports an alternate model as available, click **Run alternate models**. Explain that this is an explicit local inference action; the primary prediction is not automatically replaced.
8. Change the remaining budget so the alternate-model action is no longer affordable, then evaluate again. If risk is above the acceptance threshold, the policy should abstain rather than recommend an infeasible action.
9. Show **Prepare this decision for human review**. Explain that it pre-fills a case with the prediction, rationale, confidence, estimated error probability, and baseline risk. The operator must explicitly submit it.
10. If reviewer and evidence tokens have been configured, open the operator workspace. Enqueue the case, claim it using a reviewer label, complete or cancel it, and inspect the hash-chained event history. Add a short manual evidence record or resolve a DOI through Crossref, then inspect the stored digest status. If tokens are not configured, explain that the workspace is protected and show the documented setup rather than entering credentials during the presentation.

## 3:30–4:20 | Research and engineering contributions

- Reliability estimation, calibration, and held-out evaluation.
- Group-aware data splitting and out-of-fold reliability features.
- Append-only linked outcome records and a transactional SQLite ledger.
- Context-conditioned, propensity-adjusted action-benefit estimation.
- A constrained allocator that respects total budget, per-action capacity, and one-action-per-case rules.
- A protected operator review queue with atomic claim, completion, and cancellation transitions.
- Hash-chained review audit history and append-only evidence storage with SHA-256 digest checks.
- A Crossref DOI metadata adapter that records bibliographic metadata without claiming to validate scientific findings.

The contextual estimator and constrained allocator remain research-stage components, not the live serving policy. The local operator queue is a controlled prototype, not a production identity or reviewer-provider integration.

## 4:20–5:00 | Limitations and next steps

Be explicit:

- The live API uses a transparent baseline policy, not the learned contextual allocator.
- Reviewer IDs are operator-provided labels behind a shared token, not individually authenticated identities or production RBAC.
- Crossref lookup retrieves bibliographic metadata only; it does not fetch full text or verify scientific claims.
- Evidence submitted manually is stored and digest-checked, not independently validated.
- Offline benchmark outcomes are not real-world causal evidence.
- The system is not validated for clinical, financial, or other safety-critical deployment.
- GitHub Actions checks for the current integration PRs are pending; do not describe the code as CI-verified or merged until those checks pass.

Next steps are to complete CI and integration review, connect the allocator behind a versioned policy interface, establish real identity and provider integrations, collect genuine action outcomes where ethically and operationally possible, and validate on additional domains.

## Earlier paired alternate-model benchmark (Milestone 53)

A separate held-out Wine Quality experiment compared which 15% of cases should receive an alternate Random Forest prediction instead of the primary Logistic Regression prediction. Across three seeds (17, 42, 73), model-disagreement routing produced 23, 25, and 41 net corrections (mean 29.7); low-primary-confidence routing produced 7, 11, and 31 (mean 16.3); and a random matched-budget baseline produced 4, 6, and 11 (mean 7.0). Each policy selected 139 of 929 test cases per seed, but predictions from both models were generated for every test row before applying the selection mask. This is an offline paired substitution comparison, not a compute-saving router, live verification service, human review, or causal deployment result. The assumed action-cost units are experimental rather than measured monetary or latency costs.

## Consolidated research evidence

Use [the 2 November evidence summary](research-evidence-summary-2026-11-02.md) for exact tables and caveats. The key message is mixed evidence, not universal superiority:

- On Wine Quality at a validation-selected nominal 75% acceptance point, reliability-driven selection had 17.75% accepted error versus 16.72% for confidence-only.
- On MAGIC at the same nominal operating point, reliability-driven had 7.24% accepted error versus 7.04% for confidence-only.
- On HTRU2, under hypothetical review-cost and harm assumptions, confidence and entropy outperformed reliability at exact matched capacity.
- In a separate 15% paired alternate-model substitution experiment, model-disagreement routing produced a mean 29.7 net corrections across three seeds, versus 16.3 for low-confidence routing and 7.0 for random routing. This is offline benchmark evidence, not human review or causal deployment evidence.

## Milestone 60 first-run result

The three-seed experiment recommended **zero** alternate-model actions across all 929 test cases in each seed. Follow-up diagnostics showed that effective sample sizes averaged about 427–441, but weighted mean benefit was near zero (about -0.006 to 0.002) against an uncertainty penalty of about 0.117–0.119. The conservative lower-bound diagnostic therefore remained negative for every query. Matched-budget confidence-only and random baselines also selected zero, so that comparison is non-informative. This is a valid negative result, not evidence that contextual selection outperformed the baselines. Do not weaken the gate just to force non-zero actions.

## Likely questions

**Is VIMARSH another classifier?**  
No. It is a reliability and verification-decision layer designed to sit around an existing classifier. The base model predicts; VIMARSH estimates reliability and recommends what to do next.

**How is risk calculated in the current API?**  
The current baseline computes estimated risk as the reliability estimator's error probability multiplied by a supplied harm score.

**Does the system actually send a case to a human reviewer?**  
It can create a protected local review case after an operator explicitly submits the handoff. A reviewer can claim and complete or cancel the case, and the system records an auditable event history. Reviewer labels are not authenticated identities, and this is not an external reviewer service or production RBAC.

**Does the evidence connector verify sources?**  
No. Manual evidence is stored with a digest, and Crossref resolves DOI bibliographic metadata. Neither independently validates source claims.

**How do you prevent label leakage?**  
Inference endpoints reject unknown fields, including labels. Test labels are reserved for retrospective evaluation after policy decisions have been fixed. The research pipeline uses grouped splits and out-of-fold predictions for reliability training.

**What is the novelty?**  
Describe the research hypothesis, not a proven novelty claim: contextual action-benefit estimation combined with explicit safety, budget, and capacity constraints. Patentability or novelty requires a completed prior-art and legal review.

**What is the current limitation?**  
The live serving policy remains a transparent baseline. The contextual estimator and constrained allocator are validated offline before integration into the serving path, while the new operator workflows still need production identity, provider integrations, and operational controls.
