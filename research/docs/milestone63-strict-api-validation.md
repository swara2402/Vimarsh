# Milestone 63: Strict research API input validation

## Delivered implementation

- Applied finite-number validation across research request models so NaN and infinity cannot enter estimates, budgets, costs, contexts, or inference inputs.
- Bounded feature/context maps and per-case action/cost maps to the supported research dimensions.
- Rejected negative or non-finite estimated action costs and negative/non-integer action capacities before the allocator runs.
- Rejected duplicate case IDs in a batch, duplicate available actions, and costs attached to actions not declared available.
- Added API regression tests for invalid numeric values and malformed allocation requests.

## Why this matters

The allocator and estimator already validate many values internally. This milestone moves important checks to the API boundary, returning consistent HTTP 422 responses instead of relying on downstream failures or optimizer behavior. It also prevents malformed batch payloads from ambiguously referring to the same case more than once.

## Scope and limitations

These checks improve input integrity; they do not authenticate callers beyond the existing research-token mechanism, validate the truthfulness of submitted outcomes, establish causal action benefit, or make the JSONL ledger safe for multi-process writers. The protected research API remains a local research-preview surface.
