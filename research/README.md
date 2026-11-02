# Research Workspace

This directory will contain reproducible experiments, dataset documentation, baseline results, and evaluation artifacts.

## Suggested structure

```text
research/
├── datasets/
│   └── README.md
├── notebooks/
├── configs/
├── experiments/
├── results/
└── README.md
```

## First experiment

1. Obtain and inspect the Wine Quality dataset from its authoritative source.
2. Document its provenance, license/terms, features, target, and known limitations.
3. Create a reproducible train/validation/test split.
4. Train Logistic Regression and Random Forest baselines.
5. Record predicted probabilities and correctness on validation data.
6. Compare confidence with empirical correctness.
7. Save metrics, configuration, random seed, and environment details.

Do not commit sensitive data, credentials, or large datasets without checking licensing and repository policy. Keep test labels out of all policy-training and inference-time decision paths.

## Reproducibility checklist

- Fixed and recorded random seeds.
- Versioned dependencies and dataset provenance.
- Explicit split strategy.
- No preprocessing fit on test data.
- Baseline metrics and calibration metrics.
- Results saved with experiment configuration.
- Clear distinction between measured, simulated, and illustrative values.

## Action-outcome ledger storage

The protected research API retains the append-only JSONL ledger by default for compatibility. An optional SQLite backend provides transactional local multi-process writes while preserving record validation, unique event IDs, SHA-256 hash-chain verification, and the completed propensity-logged analysis filter.

Configure with:

```bash
VIMARSH_OUTCOME_LEDGER_BACKEND=sqlite
VIMARSH_OUTCOME_LEDGER=research/data/action_outcomes.sqlite3
```

See [Milestone 66: Transactional outcome ledger](docs/milestone66-transactional-outcome-ledger.md) for the storage guarantees and limitations. SQLite is local transactional storage, not a distributed database.
