"""Append-only operator-submitted evidence register.

This is a manual evidence intake adapter. It does not scrape, browse, or
independently verify external sources.
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


def require_evidence_token(x_vimarsh_evidence_token: str | None = Header(default=None)) -> None:
    expected = os.getenv("VIMARSH_EVIDENCE_API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Evidence intake is disabled until VIMARSH_EVIDENCE_API_TOKEN is configured")
    if not x_vimarsh_evidence_token or not secrets.compare_digest(x_vimarsh_evidence_token, expected):
        raise HTTPException(status_code=401, detail="Invalid evidence intake token")


def _connect() -> sqlite3.Connection:
    configured = os.getenv("VIMARSH_EVIDENCE_DB", "research/data/evidence_register.sqlite3")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=20, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=20000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("""CREATE TABLE IF NOT EXISTS evidence_items (
        evidence_id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL,
        source_name TEXT NOT NULL,
        source_reference TEXT NOT NULL,
        evidence_text TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        content_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS evidence_case_idx ON evidence_items(case_id, created_at)")
    conn.execute("""CREATE TRIGGER IF NOT EXISTS evidence_no_update BEFORE UPDATE ON evidence_items
        BEGIN SELECT RAISE(ABORT, 'evidence register is append-only'); END""")
    conn.execute("""CREATE TRIGGER IF NOT EXISTS evidence_no_delete BEFORE DELETE ON evidence_items
        BEGIN SELECT RAISE(ABORT, 'evidence register is append-only'); END""")
    return conn


def add_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(
        {"case_id": payload["case_id"], "source_name": payload["source_name"],
         "source_reference": payload["source_reference"], "evidence_text": payload["evidence_text"],
         "metadata": payload["metadata"]},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    evidence_id = str(uuid4())
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("""INSERT INTO evidence_items
            (evidence_id, case_id, source_name, source_reference, evidence_text, metadata_json, content_sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (evidence_id, payload["case_id"], payload["source_name"], payload["source_reference"],
             payload["evidence_text"], json.dumps(payload["metadata"], sort_keys=True, allow_nan=False), digest))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"evidence_id": evidence_id, "case_id": payload["case_id"],
            "source_name": payload["source_name"], "source_reference": payload["source_reference"],
            "content_sha256": digest, "integrity": "sha256_recorded"}


def list_evidence(case_id: str, limit: int = 50) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute("""SELECT * FROM evidence_items WHERE case_id=?
            ORDER BY created_at DESC LIMIT ?""", (case_id, limit)).fetchall()
        return [{
            "evidence_id": row["evidence_id"], "case_id": row["case_id"],
            "source_name": row["source_name"], "source_reference": row["source_reference"],
            "evidence_text": row["evidence_text"], "metadata": json.loads(row["metadata_json"]),
            "content_sha256": row["content_sha256"], "created_at": row["created_at"],
        } for row in rows]
    finally:
        conn.close()
