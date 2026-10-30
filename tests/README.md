# Testing Strategy

Tests will cover correctness, safety constraints, reproducibility, and evaluation integrity.

## Planned test groups

- **Unit tests:** reliability features, calibration helpers, risk calculations, and cost calculations.
- **Constraint tests:** unsafe, unavailable, or unaffordable actions must never be selected.
- **Policy tests:** deterministic behavior for fixed inputs and policy versions.
- **API tests:** schema validation, error handling, and response contracts.
- **Leakage tests:** test labels are not passed into inference-time feature or policy paths.
- **Evaluation tests:** metrics handle abstentions and report coverage alongside risk.
- **Simulation tests:** simulated action outcomes are marked and never represented as real human review.

## Minimum safety invariants

1. The learned action evaluator cannot override hard constraints.
2. An unavailable evidence source cannot be selected as an executable action.
3. Abstention is available as a defined fallback.
4. Test labels are reserved for offline evaluation.
5. Missing verification outcomes are not silently treated as successful outcomes.