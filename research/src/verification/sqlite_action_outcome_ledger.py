"""Transactional SQLite backend for the append-only VIMARSH outcome ledger."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .action_outcome_ledger import (
    ActionOutcomeLedger,
    ActionOutcomeRecord,
    GENESIS_HASH,
)


class SQLiteActionOutcomeLedger:
    """Append-only, hash-chained ledger with serialized multi-process writes.

    SQLite's BEGIN IMMEDIATE transaction serializes writers before the previous
    hash is read, preventing concurrent requests from creating competing chain
    heads. Each method opens its own connection so instances are process-safe.
    """

    def __init__(self, path: str | Path, timeout: float = 30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS action_outcomes (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    previous_hash TEXT NOT NULL,
                    record_hash TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)}")
        return connection

    @staticmethod
    def _digest(record: dict[str, Any]) -> str:
        return ActionOutcomeLedger._digest(record)

    def read_verified(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_id, previous_hash, record_hash, record_json
                FROM action_outcomes ORDER BY sequence ASC
                """
            ).fetchall()

        verified: list[dict[str, Any]] = []
        previous = GENESIS_HASH
        for row in rows:
            try:
                record = json.loads(row["record_json"])
                if row["event_id"] != record["event_id"]:
                    raise ValueError("event_id mismatch")
                if row["previous_hash"] != previous:
                    raise ValueError("previous hash mismatch")
                if row["record_hash"] != self._digest(record):
                    raise ValueError("record hash mismatch")
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid SQLite ledger entry at sequence {row['sequence']}"
                ) from exc
            except ValueError as exc:
                raise ValueError(
                    f"ledger integrity failure at sequence {row['sequence']}: {exc}"
                ) from exc
            previous = row["record_hash"]
            verified.append(record)
        return verified

    def append(self, record: ActionOutcomeRecord) -> dict[str, Any]:
        row = record.validated()
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = self._digest(row)

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            previous_row = connection.execute(
                "SELECT record_hash FROM action_outcomes ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous = previous_row["record_hash"] if previous_row else GENESIS_HASH
            connection.execute(
                """
                INSERT INTO action_outcomes
                    (event_id, previous_hash, record_hash, record_json)
                VALUES (?, ?, ?, ?)
                """,
                (row["event_id"], previous, digest, encoded),
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise ValueError("event_id already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return row

    def analysis_ready_records(self) -> list[dict[str, Any]]:
        return [
            record for record in self.read_verified()
            if record["status"] == "completed" and record["propensity"] is not None
        ]
