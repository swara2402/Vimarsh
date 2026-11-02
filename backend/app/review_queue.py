"""Persistent local review queue with an append-only, hash-chained audit trail.

Reviewer IDs are operator-supplied labels behind a shared token, not individual
authenticated accounts. This module does not claim identity or RBAC.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Header, HTTPException


def require_reviewer_token(x_vimarsh_reviewer_token: str | None = Header(default=None)) -> None:
    expected = os.getenv("VIMARSH_REVIEWER_API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Human-review queue is disabled until VIMARSH_REVIEWER_API_TOKEN is configured")
    if not x_vimarsh_reviewer_token or not secrets.compare_digest(x_vimarsh_reviewer_token, expected):
        raise HTTPException(status_code=401, detail="Invalid reviewer token")


def _db_path() -> Path:
    configured = os.getenv("VIMARSH_REVIEW_QUEUE_DB", "research/data/review_queue.sqlite3")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=20, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=20000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_cases (
            review_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL CHECK(status IN ('pending','claimed','completed','cancelled')),
            priority INTEGER NOT NULL CHECK(priority BETWEEN 0 AND 5),
            prediction TEXT NOT NULL,
            rationale_json TEXT NOT NULL,
            context_json TEXT NOT NULL,
            reviewer_id TEXT,
            reviewer_note TEXT,
            final_prediction TEXT,
            ground_truth TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS review_cases_queue_idx ON review_cases(status, priority DESC, created_at ASC)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            review_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            actor TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            previous_hash TEXT NOT NULL,
            event_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""CREATE TRIGGER IF NOT EXISTS review_events_no_update
        BEFORE UPDATE ON review_events BEGIN SELECT RAISE(ABORT, 'review audit log is append-only'); END""")
    conn.execute("""CREATE TRIGGER IF NOT EXISTS review_events_no_delete
        BEFORE DELETE ON review_events BEGIN SELECT RAISE(ABORT, 'review audit log is append-only'); END""")
    return conn


def _append_event(conn: sqlite3.Connection, review_id: str, event_type: str,
                  actor: str, payload: dict[str, Any]) -> None:
    previous = conn.execute("SELECT event_hash FROM review_events ORDER BY sequence DESC LIMIT 1").fetchone()
    previous_hash = previous["event_hash"] if previous else "0" * 64
    event_id = str(uuid4())
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    envelope = {
        "event_id": event_id, "review_id": review_id, "event_type": event_type,
        "actor": actor, "payload_json": payload_json, "previous_hash": previous_hash,
    }
    event_hash = hashlib.sha256(json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    conn.execute("""INSERT INTO review_events
        (event_id, review_id, event_type, actor, payload_json, previous_hash, event_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (event_id, review_id, event_type, actor, payload_json, previous_hash, event_hash))


def _verify_audit_chain(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("SELECT * FROM review_events ORDER BY sequence ASC").fetchall()
    previous_hash = "0" * 64
    for row in rows:
        envelope = {
            "event_id": row["event_id"], "review_id": row["review_id"],
            "event_type": row["event_type"], "actor": row["actor"],
            "payload_json": row["payload_json"], "previous_hash": previous_hash,
        }
        expected = hashlib.sha256(json.dumps(
            envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        if row["previous_hash"] != previous_hash or not secrets.compare_digest(row["event_hash"], expected):
            return False
        previous_hash = row["event_hash"]
    return True


def verify_audit_chain() -> bool:
    conn = _connect()
    try:
        conn.execute("BEGIN")
        return _verify_audit_chain(conn)
    finally:
        conn.close()


def list_events(review_id: str) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        # Verify and retrieve inside one SQLite read snapshot so a concurrent
        # append cannot slip between integrity verification and result loading.
        conn.execute("BEGIN")
        if not _verify_audit_chain(conn):
            raise ValueError("review audit chain integrity verification failed")
        rows = conn.execute(
            "SELECT * FROM review_events WHERE review_id=? ORDER BY sequence ASC", (review_id,)
        ).fetchall()
        return [{
            "event_id": row["event_id"], "review_id": row["review_id"],
            "event_type": row["event_type"], "actor": row["actor"],
            "payload": json.loads(row["payload_json"]), "previous_hash": row["previous_hash"],
            "event_hash": row["event_hash"], "created_at": row["created_at"],
        } for row in rows]
    finally:
        conn.close()


def enqueue(payload: dict[str, Any]) -> dict[str, Any]:
    conn = _connect()
    review_id = str(uuid4())
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """INSERT INTO review_cases
               (review_id, case_id, status, priority, prediction, rationale_json, context_json)
               VALUES (?, ?, 'pending', ?, ?, ?, ?)""",
            (review_id, payload["case_id"], payload["priority"], payload["prediction"],
             json.dumps(payload["rationale"], allow_nan=False),
             json.dumps(payload["context"], sort_keys=True, allow_nan=False)),
        )
        _append_event(conn, review_id, "created", "operator", {
            "case_id": payload["case_id"], "priority": payload["priority"],
            "prediction": payload["prediction"],
        })
        conn.commit()
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise ValueError("case_id already exists in the review queue") from exc
    finally:
        conn.close()
    return get_case(review_id)


def list_cases(status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT * FROM review_cases WHERE status = ?
               ORDER BY priority DESC, created_at ASC LIMIT ?""",
            (status, limit),
        ).fetchall()
        return [_decode(row) for row in rows]
    finally:
        conn.close()


def get_case(review_id: str) -> dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM review_cases WHERE review_id = ?", (review_id,)).fetchone()
        if row is None:
            raise ValueError("review case not found")
        return _decode(row)
    finally:
        conn.close()


def claim_case(review_id: str, reviewer_id: str) -> dict[str, Any]:
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """UPDATE review_cases SET status='claimed', reviewer_id=?, updated_at=CURRENT_TIMESTAMP
               WHERE review_id=? AND status='pending'""",
            (reviewer_id, review_id),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError("review case is missing or no longer pending")
        _append_event(conn, review_id, "claimed", reviewer_id, {})
        conn.commit()
    finally:
        conn.close()
    return get_case(review_id)


def cancel_case(review_id: str, actor: str, reason: str) -> dict[str, Any]:
    """Cancel a pending or claimed case while preserving its audit history."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status FROM review_cases WHERE review_id = ?", (review_id,)
        ).fetchone()
        if row is None or row["status"] not in {"pending", "claimed"}:
            conn.rollback()
            raise ValueError("only pending or claimed cases can be cancelled")
        conn.execute(
            """UPDATE review_cases SET status='cancelled', updated_at=CURRENT_TIMESTAMP
               WHERE review_id=?""",
            (review_id,),
        )
        _append_event(conn, review_id, "cancelled", actor, {"reason": reason})
        conn.commit()
    finally:
        conn.close()
    return get_case(review_id)


def complete_case(review_id: str, reviewer_id: str, reviewer_note: str,
                  final_prediction: str, ground_truth: str | None) -> dict[str, Any]:
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """UPDATE review_cases SET status='completed', reviewer_note=?,
               final_prediction=?, ground_truth=?, updated_at=CURRENT_TIMESTAMP
               WHERE review_id=? AND status='claimed' AND reviewer_id=?""",
            (reviewer_note, final_prediction, ground_truth, review_id, reviewer_id),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError("case must be claimed by this reviewer before completion")
        _append_event(conn, review_id, "completed", reviewer_id, {
            "final_prediction": final_prediction, "ground_truth_recorded": ground_truth is not None,
        })
        conn.commit()
    finally:
        conn.close()
    return get_case(review_id)


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "review_id": row["review_id"], "case_id": row["case_id"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
        "status": row["status"], "priority": row["priority"],
        "prediction": row["prediction"], "rationale": json.loads(row["rationale_json"]),
        "context": json.loads(row["context_json"]), "reviewer_id": row["reviewer_id"],
        "reviewer_note": row["reviewer_note"], "final_prediction": row["final_prediction"],
        "ground_truth": row["ground_truth"],
    }
