from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from src.verification.action_outcome_ledger import ActionOutcomeRecord
from src.verification.sqlite_action_outcome_ledger import SQLiteActionOutcomeLedger


def record(index):
    return ActionOutcomeRecord(
        case_id=f"case-{index}",
        action="alternate_model",
        policy_version="policy-v1",
        model_version="model-v1",
        baseline_prediction=0,
        status="completed",
        estimated_action_cost=1.0,
        propensity=0.5,
        randomized_assignment=True,
        assignment_id=f"assignment-{index}",
        context={"uncertainty": 0.4, "harm": 1.0},
        action_result={"prediction": 1},
        final_prediction=1,
        ground_truth=1,
        observed_cost=0.8,
        latency_ms=4.2,
        provider_version="test-runner-v1",
        event_id=f"event-{index}",
    )


def test_sqlite_ledger_appends_and_verifies_hash_chain(tmp_path):
    ledger = SQLiteActionOutcomeLedger(tmp_path / "outcomes.sqlite3")
    first = ledger.append(record(1))
    second = ledger.append(record(2))

    verified = ledger.read_verified()
    assert [row["event_id"] for row in verified] == ["event-1", "event-2"]
    assert ledger.analysis_ready_records() == verified
    assert first["event_id"] == "event-1"
    assert second["event_id"] == "event-2"


def test_sqlite_ledger_rejects_duplicate_event_ids(tmp_path):
    ledger = SQLiteActionOutcomeLedger(tmp_path / "outcomes.sqlite3")
    ledger.append(record(1))
    with pytest.raises(ValueError, match="event_id already exists"):
        ledger.append(record(1))


def test_sqlite_ledger_serializes_concurrent_writers(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    SQLiteActionOutcomeLedger(path)

    def append_one(index):
        return SQLiteActionOutcomeLedger(path).append(record(index))["event_id"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        inserted = list(executor.map(append_one, range(24)))

    verified = SQLiteActionOutcomeLedger(path).read_verified()
    assert len(inserted) == 24
    assert len({row["event_id"] for row in verified}) == 24
    assert {row["event_id"] for row in verified} == set(inserted)


def test_sqlite_ledger_detects_record_tampering(tmp_path):
    path = tmp_path / "tampered.sqlite3"
    ledger = SQLiteActionOutcomeLedger(path)
    ledger.append(record(1))

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE action_outcomes SET record_json = ? WHERE event_id = ?",
            ('{"event_id":"event-1","status":"completed"}', "event-1"),
        )

    with pytest.raises(ValueError, match="record hash mismatch"):
        ledger.read_verified()


def test_sqlite_ledger_preserves_pending_records_outside_analysis_set(tmp_path):
    ledger = SQLiteActionOutcomeLedger(tmp_path / "pending.sqlite3")
    pending = ActionOutcomeRecord(
        case_id="pending-case",
        action="alternate_model",
        policy_version="policy-v1",
        model_version="model-v1",
        baseline_prediction=0,
        status="pending",
        estimated_action_cost=1.0,
        context={"uncertainty": 0.4},
        event_id="pending-event",
    )
    ledger.append(pending)

    assert len(ledger.read_verified()) == 1
    assert ledger.analysis_ready_records() == []
