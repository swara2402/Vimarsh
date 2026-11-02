"""Persistent local human-review queue for explicitly routed VIMARSH cases.

This queue is an operator workflow, not an automated reviewer or a claim of
human-review availability. Access is gated by a separate server-side token.
"""
from __future__ import annotations

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
    return conn


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
