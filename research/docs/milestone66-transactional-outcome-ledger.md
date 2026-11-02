# Milestone 66: Transactional outcome ledger backend

## Delivered implementation

- Added an optional SQLite-backed append-only action-outcome ledger.
- Preserved the existing JSONL ledger as the default for compatibility with prior local research data.
- Serialized SQLite writers with `BEGIN IMMEDIATE` before reading the previous chain hash and inserting the next event.
- Added a unique database constraint for event IDs, WAL journaling, a busy timeout, and full synchronous commits.
- Kept the same record validation, SHA-256 hash-chain verification, and completed propensity-logged analysis filter.
- Added concurrent-writer, duplicate-event, tamper-detection, pending-record, and protected-API integration tests.
- Added `VIMARSH_OUTCOME_LEDGER_BACKEND=jsonl|sqlite` configuration and operator documentation.
- Added a verified, resumable JSONL-to-SQLite migration command. It verifies the source chain, accepts only an exact destination prefix, preserves the source file, and verifies the complete destination after migration.

## Configuration

Legacy mode remains the default:

```bash
export VIMARSH_OUTCOME_LEDGER_BACKEND=jsonl
export VIMARSH_OUTCOME_LEDGER=research/data/action_outcomes.jsonl
```

For transactional local storage:

```bash
export VIMARSH_OUTCOME_LEDGER_BACKEND=sqlite
export VIMARSH_OUTCOME_LEDGER=research/data/action_outcomes.sqlite3
```

The database file should be kept outside version control and included in an operator-managed backup policy.

## Integrity and safety

- Records remain append-only; SQLite triggers reject ordinary UPDATE and DELETE statements, and there is no API endpoint to edit or delete historical events.
- Each record is validated before insertion and linked to the prior record hash.
- Duplicate event IDs are rejected by a database uniqueness constraint.
- A writer obtains a SQLite write reservation before reading the chain head, preventing concurrent local API workers from forking the hash chain.
- Reads re-compute every record hash and verify the complete chain before returning records for estimation.
- Only completed records with a recorded propensity are passed to the action-benefit estimator.

## Scope and limitations

SQLite provides transactional coordination for a single host and local filesystem. It is not a distributed database, does not provide identity/role management, and does not replace encrypted backups, retention controls, or production operations. The estimator remains observational and its lower-bound diagnostic is not a causal guarantee. Human review and additional-evidence providers remain unavailable until real integrations and outcome collection exist.

## Migrating existing data

Run from the repository's `research/` directory after installing the research dependencies:

```bash
python -m src.verification.ledger_migration --source data/action_outcomes.jsonl --destination data/action_outcomes.sqlite3
```

The migration is non-destructive and resumable. It never overwrites or deletes the source JSONL file. Re-running it against the same destination verifies the existing prefix and adds only missing records. A divergent or tampered source/destination is rejected.