# VIMARSH: five-minute project demonstration script

## 0:00–0:40 | Problem

Machine-learning systems often return a prediction with a confidence score, but confidence alone does not tell us whether a prediction is reliable enough for a consequential decision. VIMARSH explores a second question: when should a prediction be accepted, checked further, or withheld?

## 0:40–1:30 | Proposed approach

Walk through the architecture:

1. A base model produces a prediction.
2. A reliability estimator analyzes prediction-derived signals and estimates the probability of error.
3. The estimated error probability is combined with a supplied harm score to form a baseline risk estimate.
4. A transparent policy applies hard availability and budget constraints.
5. The system recommends acceptance, an eligible alternate-model check, or abstention.

Emphasize that the current API is recommendation-only. It does not execute the recommended action.

## 1:30–3:30 | Live dashboard

1. Open the React dashboard and wait for the model schema to load.
2. Explain that the feature fields are retrieved from the trained bundle, not hard-coded into the page.
3. Point out the live capability panel: the dashboard reports which action paths are configured and marks human review/evidence acquisition unavailable rather than pretending they are integrated.
4. Click **Load example wine sample** to populate a complete red-wine benchmark row. Point out that the UI maps the sample by the feature names returned from the trained bundle.
5. Set harm to 8, keep the remaining budget at 2, and leave the alternate-model option enabled with its default estimated cost of 1. This illustrates how the supplied harm weight affects baseline risk and how the budget constrains the recommendation.
6. Select **Evaluate reliability**.
7. Explain the output in this order:
   - primary-model prediction and confidence;
   - estimated error probability from the reliability estimator;
   - baseline risk, calculated as estimated error probability multiplied by harm;
   - recommended action and the policy rationale.
8. If the capability panel reports an alternate model as available, click **Run alternate models**. Explain that this is an explicit local inference action and its predictions/latency are shown as additional evidence; VIMARSH does not automatically replace the primary prediction.
9. Change the remaining budget so the alternate-model action is no longer affordable, then evaluate again. If estimated risk is above the acceptance threshold, the policy should abstain rather than recommend an infeasible action.
10. Disable alternate-model availability and repeat to show the same hard-constraint behavior.

## 3:30–4:20 | Research contributions

Summarize the research progression without presenting unfinished work as production functionality:

- Reliability estimation and calibration experiments.
- Group-aware data splitting and held-out evaluation.
- An append-only, hash-chained ledger for paired action outcomes.
- Context-conditioned, propensity-adjusted action-benefit estimation.
- A constrained allocator that respects total budget, per-action capacity and one-action-per-case rules.
- An offline paired benchmark comparing contextual selection with matched-budget baselines.

The action-outcome ledger, contextual benefit estimator and constrained allocator are implemented as protected research APIs and offline research components. The allocator is not yet the live serving policy.

## 4:20–5:00 | Limitations and next steps

Be explicit:

- Current live API policy is a transparent baseline, not yet the learned contextual allocator.
- Human review and additional-evidence providers are not operational integrations.
- Offline benchmark outcomes are not real-world causal evidence.
- Cost values in the live demo are supplied estimates, not measured monetary or latency costs.
- The system is not validated for clinical, financial, or other safety-critical deployment.

Next steps are to complete CI and integration review, connect the allocator behind a versioned policy interface, collect genuine action outcomes where ethically and operationally possible, and validate on additional domains.

## Earlier paired alternate-model benchmark (Milestone 53)

A separate held-out Wine Quality experiment compared which 15% of cases should receive an alternate Random Forest prediction instead of the primary Logistic Regression prediction. Across three seeds (17, 42, 73), model-disagreement routing produced 23, 25, and 41 net corrections (mean 29.7); low-primary-confidence routing produced 7, 11, and 31 (mean 16.3); and a random matched-budget baseline produced 4, 6, and 11 (mean 7.0). Each policy selected 139 of 929 test cases per seed, but predictions from both models were generated for every test row before applying the selection mask. This is an offline paired substitution comparison, not a compute-saving router, live verification service, human review, or causal deployment result. The assumed action-cost units are experimental rather than measured monetary or latency costs.

## Consolidated research evidence

Use [the 2 November evidence summary](research-evidence-summary-2026-11-02.md) for exact tables and caveats. The key message is mixed evidence, not universal superiority:

- On Wine Quality at a validation-selected nominal 75% acceptance point, reliability-driven selection had 17.75% accepted error versus 16.72% for confidence-only.
- On MAGIC at the same nominal operating point, reliability-driven had 7.24% accepted error versus 7.04% for confidence-only.
- On HTRU2, under hypothetical review-cost and harm assumptions, confidence and entropy outperformed reliability at exact matched capacity.
- In a separate 15% paired alternate-model substitution experiment, model-disagreement routing produced a mean 29.7 net corrections across three seeds, versus 16.3 for low-confidence routing and 7.0 for random routing. This is offline benchmark evidence, not human review or causal deployment evidence.

## Milestone 60 first-run result (important)

The three-seed CI artifact recommended **zero** alternate-model actions across all 929 test cases in each seed. Follow-up diagnostics show the reason was not insufficient sample support: effective sample sizes averaged about 427–441, but weighted mean benefit was near zero (about -0.006 to 0.002) against an uncertainty penalty of about 0.117–0.119. The conservative lower-bound diagnostic therefore remained negative for every query. Matched-budget confidence-only and random baselines also selected zero, so that comparison is non-informative. This is a valid negative result, not evidence that contextual selection outperformed the baselines. Do not weaken the gate just to force non-zero actions.

## Likely questions

**Is VIMARSH another classifier?**  
No. It is a reliability and verification-decision layer designed to sit around an existing classifier. The base model predicts; VIMARSH estimates reliability and recommends what to do next.

**How is risk calculated in the current API?**  
The current baseline computes estimated risk as the reliability estimator's error probability multiplied by a supplied harm score.

**Does the system actually send a case to a human reviewer?**  
No. Human review is explicitly excluded until a real reviewer workflow exists. The API returns a recommendation only.

**How do you prevent label leakage?**  
Inference endpoints reject unknown fields, including labels. Test labels are reserved for retrospective evaluation after policy decisions have been fixed. The research pipeline uses grouped splits and out-of-fold predictions for reliability training.

**What is the novelty?**  
Describe the research hypothesis, not a proven novelty claim: contextual action-benefit estimation combined with explicit safety, budget and capacity constraints. Patentability or novelty requires a completed prior-art and legal review.

**What is the current limitation?**  
The live API still uses a transparent baseline decision policy. The contextual estimator and constrained allocator are being validated offline before they are integrated into the serving path.
