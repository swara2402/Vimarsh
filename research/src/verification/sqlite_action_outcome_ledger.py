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


_COMPLETION_FIELDS = {
    "event_id",
    "parent_event_id",
    "occurred_at",
    "status",
    "final_prediction",
    "ground_truth",
    "observed_cost",
    "latency_ms",
    "provider_version",
}


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
        connection = self._connect()
        try:
            # Serialize first-time schema creation and legacy-column upgrades so
            # concurrent API workers cannot race on ALTER TABLE.
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS action_outcomes (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    parent_event_id TEXT,
                    previous_hash TEXT NOT NULL,
                    record_hash TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(action_outcomes)").fetchall()
            }
            if "parent_event_id" not in columns:
                connection.execute("ALTER TABLE action_outcomes ADD COLUMN parent_event_id TEXT")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS action_outcomes_parent_event_unique
                ON action_outcomes(parent_event_id)
                WHERE parent_event_id IS NOT NULL
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS action_outcomes_no_update
                BEFORE UPDATE ON action_outcomes
                BEGIN
                    SELECT RAISE(ABORT, 'outcome ledger is append-only');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS action_outcomes_no_delete
                BEFORE DELETE ON action_outcomes
                BEGIN
                    SELECT RAISE(ABORT, 'outcome ledger is append-only');
                END
                """
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)}")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @staticmethod
    def _digest(record: dict[str, Any]) -> str:
        return ActionOutcomeLedger._digest(record)

    def read_verified(self) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT sequence, event_id, parent_event_id, previous_hash, record_hash, record_json
                FROM action_outcomes ORDER BY sequence ASC
                """
            ).fetchall()
        finally:
            connection.close()

        verified: list[dict[str, Any]] = []
        previous = GENESIS_HASH
        for row in rows:
            try:
                record = json.loads(row["record_json"])
                if row["event_id"] != record["event_id"]:
                    raise ValueError("event_id mismatch")
                if row["parent_event_id"] != record.get("parent_event_id"):
                    raise ValueError("parent_event_id mismatch")
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
            parent_event_id = row.get("parent_event_id")
            if parent_event_id is not None:
                parent_row = connection.execute(
                    "SELECT record_json FROM action_outcomes WHERE event_id = ?",
                    (parent_event_id,),
                ).fetchone()
                if parent_row is None:
                    raise ValueError("parent_event_id does not reference an existing ledger event")
                parent = json.loads(parent_row["record_json"])
                if parent.get("status") != "pending" or row["status"] != "completed":
                    raise ValueError("linked completion must reference a pending event and be completed")
                for field, value in parent.items():
                    if field not in _COMPLETION_FIELDS and row.get(field) != value:
                        raise ValueError("linked completion must preserve the original execution context")
                existing_child = connection.execute(
                    "SELECT 1 FROM action_outcomes WHERE parent_event_id = ?",
                    (parent_event_id,),
                ).fetchone()
                if existing_child is not None:
                    raise ValueError("parent event already has a completion record")

            previous_row = connection.execute(
                "SELECT record_hash FROM action_outcomes ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous = previous_row["record_hash"] if previous_row else GENESIS_HASH
            connection.execute(
                """
                INSERT INTO action_outcomes
                    (event_id, parent_event_id, previous_hash, record_hash, record_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (row["event_id"], parent_event_id, previous, digest, encoded),
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            if "parent_event_id" in str(exc) or "action_outcomes_parent_event_unique" in str(exc):
                raise ValueError("parent event already has a completion record") from exc
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
