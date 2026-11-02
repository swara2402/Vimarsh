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
3. Click **Load example wine sample** to populate a complete red-wine benchmark row. Point out that the UI maps the sample by the feature names returned from the trained bundle.
4. Set a harm score and leave the alternate model enabled with an affordable estimated cost.
5. Select **Evaluate reliability**.
6. Explain the output in this order:
   - primary-model prediction and confidence;
   - estimated error probability from the reliability estimator;
   - baseline risk, calculated as estimated error probability multiplied by harm;
   - recommended action and the policy rationale.
7. Change the remaining budget so the alternate-model action is no longer affordable, then evaluate again. If estimated risk is above the acceptance threshold, the policy should abstain rather than recommend an infeasible action.
8. Disable alternate-model availability and repeat to show the same hard-constraint behavior.

## 3:30–4:20 | Research contributions

Summarize the research progression without presenting unfinished work as production functionality:

- Reliability estimation and calibration experiments.
- Group-aware data splitting and held-out evaluation.
- An append-only, hash-chained ledger for paired action outcomes.
- Context-conditioned, propensity-adjusted action-benefit estimation.
- A constrained allocator that respects total budget, per-action capacity and one-action-per-case rules.
- An offline paired benchmark comparing contextual selection with matched-budget baselines.

The latest action-benefit and allocator milestones are subject to their pull-request CI and merge review. If they are not merged before the presentation, describe them as ongoing research implementation rather than completed main-branch features.

## 4:20–5:00 | Limitations and next steps

Be explicit:

- Current live API policy is a transparent baseline, not yet the learned contextual allocator.
- Human review and additional-evidence providers are not operational integrations.
- Offline benchmark outcomes are not real-world causal evidence.
- Cost values in the live demo are supplied estimates, not measured monetary or latency costs.
- The system is not validated for clinical, financial, or other safety-critical deployment.

Next steps are to complete CI and integration review, connect the allocator behind a versioned policy interface, collect genuine action outcomes where ethically and operationally possible, and validate on additional domains.

## Milestone 60 first-run result (important)

The first three-seed CI artifact completed, but the contextual estimator recommended **zero** alternate-model actions across all 929 test cases in each seed. Therefore the matched-budget confidence-only and random baselines also selected zero cases. The conservative lower-bound/support gate abstained throughout; this is a valid negative/abstention result, not evidence that contextual selection outperformed the baselines. Present it honestly as a current limitation and say that estimator-support and threshold debugging is the next research task. Do not quote a positive action-benefit result from this run.

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
