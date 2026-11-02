# Milestone 57: Operational action-outcome data foundation

## Goal
Provide an auditable ingestion boundary for action-specific outcomes: strict record validation, propensity metadata, an append-only JSONL ledger, and tamper-evident hash chaining. This is infrastructure, not real outcome data and not a causal-effect claim.

## Lifecycle
1. Record a pseudonymous case ID, selected action, policy/model versions, baseline prediction, policy-time context, estimated cost, and selection propensity when available.
2. Record the actual action result, observed cost, latency, and provider version.
3. Once an eventual label is legitimately available, append a completed event with the final prediction and retrospective ground truth. Do not mutate the pending event.
4. Only completed events with known non-zero propensity are exposed by analysis_ready_records().

## Integrity and privacy
- Ground truth is rejected from policy context and accepted only for completed retrospective records.
- Pending, failed, and not-applicable actions are not successful outcomes.
- Randomized assignment requires an assignment identifier.
- Estimated and observed costs remain separate.
- Hash chaining detects edits to prior ledger entries, but is not a replacement for authenticated storage, access control, backups, or database transactions.
- Case IDs must be pseudonymous. Do not store names, emails, phone numbers, raw prompts, or unnecessary personal data.

## Limitations
This ledger does not integrate a real human-review or evidence provider. Existing offline alternate-model experiments remain benchmark measurements, not operational human/evidence outcomes. Real paired data is required before training an action-benefit model or making causal claims.

## Local use
Run from research/ and instantiate ActionOutcomeLedger("results/operational/action_outcomes.jsonl"). Append an ActionOutcomeRecord with status="pending" when an action is selected; append a separate completed record after execution and retrospective label availability.
