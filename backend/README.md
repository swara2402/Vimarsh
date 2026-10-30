# Backend

The planned backend is a Python service, with FastAPI as the initial API framework.

## Proposed responsibilities

- Accept model predictions and associated metadata.
- Validate reliability-signal inputs.
- Run reliability and risk estimators.
- Apply hard safety and resource constraints.
- Evaluate feasible verification actions.
- Record decisions, outcomes, and measured costs.
- Expose experiment and evaluation summaries.

## Initial module proposal

```text
backend/
├── app/
│   ├── api/
│   ├── core/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── tests/
├── pyproject.toml
└── README.md
```

This is a proposed layout, not implemented code yet.

## Engineering requirements

- Validate all incoming data with explicit schemas.
- Keep model inference separate from policy selection.
- Enforce safety constraints independently of learned scores.
- Log action, policy version, signal values, outcome status, and cost.
- Do not log secrets or unnecessary personal data.
- Make simulation mode explicit in API responses and stored records.
- Add tests for invalid inputs, unavailable actions, and abstention behavior.

Implementation should begin after the first baseline experiment and research interfaces are sufficiently specified.