# Milestone 64: Append-only pending-outcome completion

## Delivered implementation

- Added `parent_event_id` to outcome-ledger records so later observations can be linked to an earlier execution without editing the original event.
- Added a protected completion request contract requiring a final prediction and retrospectively observed ground truth, with optional measured cost, latency, and provider version.
- Added `POST /v1/research/outcomes/{event_id}/complete`.
- The service accepts only an existing pending event, rejects missing/non-pending parents, and prevents more than one completion child for the same pending event.
- Completion appends a new hash-chained record. The original pending record remains unchanged.
- The completed child inherits the original case, action, policy/model versions, propensity, randomized-assignment metadata, and policy context. It cannot silently invent a propensity.
- The response exposes event linkage and analysis eligibility, not the retrospective label.
- Added API regression tests for successful completion, parent preservation, analysis eligibility, duplicate completion, missing event, and non-pending event rejection.

## Lifecycle

1. A protected execution or operator workflow appends a `pending` event.
2. After the action outcome and ground truth are genuinely observed and quality-checked, the operator submits the completion payload against the pending event ID.
3. VIMARSH appends a separate `completed` child event linked by `parent_event_id`.
4. The original pending event is retained as the audit trail. The completed child is eligible for benefit estimation only when the inherited propensity is present.
5. Aggregate summary counts update from the verified append-only ledger.

## Safety and interpretation

- This endpoint does not verify the truthfulness of submitted ground truth; it records operator-supplied retrospective observations.
- Do not complete an event until the final result and ground truth are genuinely available.
- A manually executed action without a defensible assignment propensity remains ineligible for propensity-adjusted benefit estimation.
- Parent linkage improves traceability but does not provide transactional multi-process locking. The local JSONL ledger remains single-writer research storage.
- The benefit estimator remains observational/research-stage; linkage does not establish causal benefit.
- Human review and additional-evidence providers remain unavailable until real integrations exist.
