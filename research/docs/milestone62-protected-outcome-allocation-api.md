# Milestone 62: Protected outcome-learning and allocation API

## Delivered implementation

- Added strict request contracts for action outcomes, contextual benefit estimation, and batch allocation.
- Exposed protected research endpoints for append-only outcome capture, aggregate ledger status, context-conditioned action-benefit estimation, and constrained allocation.
- Reused the existing tested research estimator, hash-chained outcome ledger, and SciPy MILP allocator rather than duplicating their algorithms.
- Required a server-side research token; endpoints are disabled by default.
- Kept retrospective ground truth inside the outcome ledger and out of policy context.
- Added integration tests and operator documentation.

## Data flow

1. An operator submits a real action outcome to the protected ledger after it is completed.
2. The ledger validates action, status, propensity, assignment, context, costs, and retrospective truth rules, then appends a hash-chained record.
3. The estimator reads only completed records with known propensities and compares outcomes near the requested policy-time context.
4. The batch endpoint generates per-case/action diagnostics and passes only positive diagnostic candidates to the constrained allocator.
5. The allocator applies one-action-per-case, shared-budget, and optional capacity constraints.
6. The response is explicitly a research recommendation. No external action is executed.

## Safety and interpretation

- Never invent randomized propensities or mark observational data as randomized.
- Do not put target/label/ground-truth values in query context.
- The estimator's conservative lower-bound diagnostic is not a guaranteed confidence bound and does not prove causal benefit.
- The allocator's output is not an execution receipt.
- The local JSONL file is suitable only for a single-process research prototype. Multi-worker operation requires transactional storage and concurrency controls.
- Keep the research token server-side. Do not expose it in the React application.

## Remaining operational work

Real alternate-model execution with recorded measured latency/cost, human reviewer workflow, and evidence-provider integrations still require concrete providers and real outcome collection. This milestone creates the protected feedback and decision-allocation interfaces, not those external services.
