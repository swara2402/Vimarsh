"""Append-only, hash-chained ledger for observed verification action outcomes."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

GENESIS_HASH = "0" * 64
ALLOWED_ACTIONS = {"alternate_model", "human_review", "additional_evidence"}
ALLOWED_STATUSES = {"pending", "completed", "failed", "not_applicable"}

def _finite(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its permitted range")
    return result

def _scalar(value: Any, name: str) -> Any:
    if isinstance(value, bool) or isinstance(value, (str, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{name} must be a finite JSON scalar")

@dataclass(frozen=True)
class ActionOutcomeRecord:
    case_id: str
    action: str
    policy_version: str
    model_version: str
    baseline_prediction: Any
    status: str
    estimated_action_cost: float
    propensity: float | None = None
    randomized_assignment: bool = False
    assignment_id: str | None = None
    context: Mapping[str, float] | None = None
    action_result: Any = None
    final_prediction: Any = None
    ground_truth: Any = None
    observed_cost: float | None = None
    latency_ms: float | None = None
    provider_version: str | None = None
    event_id: str | None = None
    occurred_at: str | None = None

    def validated(self) -> dict[str, Any]:
        if not all(isinstance(v, str) and v.strip() for v in (self.case_id, self.policy_version, self.model_version)):
            raise ValueError("case_id, policy_version and model_version are required")
        if self.action not in ALLOWED_ACTIONS or self.status not in ALLOWED_STATUSES:
            raise ValueError("unsupported action or status")
        if self.status == "completed" and (self.final_prediction is None or self.ground_truth is None):
            raise ValueError("completed outcomes require final_prediction and retrospective ground_truth")
        if self.status != "completed" and self.ground_truth is not None:
            raise ValueError("ground_truth is only permitted for completed outcomes")
        if self.randomized_assignment and not self.assignment_id:
            raise ValueError("randomized assignment requires assignment_id")
        propensity = None if self.propensity is None else _finite(self.propensity, "propensity", 0.0)
        if propensity is not None and (propensity <= 0.0 or propensity > 1.0):
            raise ValueError("propensity must be in (0, 1]")
        context = dict(self.context or {})
        forbidden = {"label", "target", "ground_truth", "true_label", "y_true"}
        if any(str(k).lower() in forbidden for k in context):
            raise ValueError("retrospective labels cannot appear in policy context")
        for key, value in context.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("context feature names must be non-empty strings")
            context[key] = _finite(value, f"context[{key}]")
        return {
            **asdict(self),
            "event_id": self.event_id or str(uuid4()),
            "occurred_at": self.occurred_at or datetime.now(timezone.utc).isoformat(),
            "baseline_prediction": _scalar(self.baseline_prediction, "baseline_prediction"),
            "final_prediction": None if self.final_prediction is None else _scalar(self.final_prediction, "final_prediction"),
            "ground_truth": None if self.ground_truth is None else _scalar(self.ground_truth, "ground_truth"),
            "estimated_action_cost": _finite(self.estimated_action_cost, "estimated_action_cost", 0.0),
            "observed_cost": None if self.observed_cost is None else _finite(self.observed_cost, "observed_cost", 0.0),
            "latency_ms": None if self.latency_ms is None else _finite(self.latency_ms, "latency_ms", 0.0),
            "propensity": propensity,
            "context": context,
        }

class ActionOutcomeLedger:
    """Local JSONL ledger with duplicate protection and a tamper-evident hash chain."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _digest(payload: Mapping[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def read_verified(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows, previous, seen = [], GENESIS_HASH, set()
        with self.path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    envelope = json.loads(line)
                    row = envelope["record"]
                    if envelope["previous_hash"] != previous:
                        raise ValueError("previous hash mismatch")
                    if envelope["record_hash"] != self._digest(row):
                        raise ValueError("record hash mismatch")
                    if row["event_id"] in seen:
                        raise ValueError("duplicate event_id")
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid ledger entry at line {line_no}") from exc
                except ValueError as exc:
                    raise ValueError(f"ledger integrity failure at line {line_no}: {exc}") from exc
                previous = envelope["record_hash"]
                seen.add(row["event_id"])
                rows.append(row)
        return rows

    def append(self, record: ActionOutcomeRecord) -> dict[str, Any]:
        row = record.validated()
        existing = self.read_verified()
        if any(item["event_id"] == row["event_id"] for item in existing):
            raise ValueError("event_id already exists")
        previous = self._digest(existing[-1]) if existing else GENESIS_HASH
        envelope = {"previous_hash": previous, "record_hash": self._digest(row), "record": row}
        encoded = (json.dumps(envelope, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, encoded)
        finally:
            os.close(fd)
        return row

    def analysis_ready_records(self) -> list[dict[str, Any]]:
        return [r for r in self.read_verified() if r["status"] == "completed" and r["propensity"] is not None]
