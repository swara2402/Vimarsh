"""Verified, resumable migration from the legacy JSONL ledger to SQLite."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .action_outcome_ledger import ActionOutcomeLedger, ActionOutcomeRecord
from .sqlite_action_outcome_ledger import SQLiteActionOutcomeLedger


def migrate_jsonl_to_sqlite(
    source_path: str | Path,
    destination_path: str | Path,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()
    if source == destination:
        raise ValueError("source and destination must be different files")
    if not source.is_file():
        raise FileNotFoundError(f"source JSONL ledger not found: {source}")

    source_rows = ActionOutcomeLedger(source).read_verified()
    destination_ledger = SQLiteActionOutcomeLedger(destination)
    existing_rows = destination_ledger.read_verified()

    if len(existing_rows) > len(source_rows):
        raise ValueError("destination contains records beyond the source ledger")
    if existing_rows != source_rows[: len(existing_rows)]:
        raise ValueError("destination is not an exact prefix of the verified source ledger")

    for row in source_rows[len(existing_rows):]:
        destination_ledger.append(ActionOutcomeRecord(**row))

    verified_rows = destination_ledger.read_verified()
    if verified_rows != source_rows:
        raise ValueError("post-migration verification did not match the source ledger")

    return {
        "source_records": len(source_rows),
        "already_present": len(existing_rows),
        "migrated_records": len(source_rows) - len(existing_rows),
        "destination_records": len(verified_rows),
        "source_preserved": True,
        "destination_integrity": "verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and migrate a VIMARSH JSONL outcome ledger to SQLite."
    )
    parser.add_argument("--source", required=True, help="Path to the existing JSONL ledger")
    parser.add_argument("--destination", required=True, help="Path for the SQLite database")
    args = parser.parse_args()
    result = migrate_jsonl_to_sqlite(args.source, args.destination)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
