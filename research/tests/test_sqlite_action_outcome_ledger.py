from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from src.verification.action_outcome_ledger import ActionOutcomeLedger, ActionOutcomeRecord
from src.verification.ledger_migration import migrate_jsonl_to_sqlite
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


def pending_record(index):
    return ActionOutcomeRecord(
        case_id=f"pending-case-{index}",
        action="alternate_model",
        policy_version="policy-v1",
        model_version="model-v1",
        baseline_prediction=0,
        status="pending",
        estimated_action_cost=1.0,
        propensity=0.5,
        randomized_assignment=True,
        assignment_id=f"pending-assignment-{index}",
        context={"uncertainty": 0.4, "harm": 1.0},
        event_id=f"pending-event-{index}",
    )


def completed_child(parent, event_id="child-event"):
    return ActionOutcomeRecord(
        case_id=parent.case_id,
        action=parent.action,
        policy_version=parent.policy_version,
        model_version=parent.model_version,
        baseline_prediction=parent.baseline_prediction,
        status="completed",
        estimated_action_cost=parent.estimated_action_cost,
        propensity=parent.propensity,
        randomized_assignment=parent.randomized_assignment,
        assignment_id=parent.assignment_id,
        context=parent.context,
        action_result=parent.action_result,
        final_prediction=1,
        ground_truth=1,
        observed_cost=0.8,
        latency_ms=5.0,
        provider_version="completion-runner-v1",
        event_id=event_id,
        parent_event_id=parent.event_id,
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
        connection.execute("DROP TRIGGER action_outcomes_no_update")
        connection.execute(
            "UPDATE action_outcomes SET record_json = ? WHERE event_id = ?",
            ('{"event_id":"event-1","status":"completed"}', "event-1"),
        )
    with pytest.raises(ValueError, match="record hash mismatch"):
        ledger.read_verified()


def test_sqlite_ledger_preserves_pending_records_outside_analysis_set(tmp_path):
    ledger = SQLiteActionOutcomeLedger(tmp_path / "pending.sqlite3")
    ledger.append(pending_record(1))
    assert len(ledger.read_verified()) == 1
    assert ledger.analysis_ready_records() == []


def test_sqlite_ledger_completes_pending_event_as_immutable_child(tmp_path):
    ledger = SQLiteActionOutcomeLedger(tmp_path / "completion.sqlite3")
    parent = pending_record(2)
    original = ledger.append(parent)
    child = ledger.append(completed_child(parent))
    rows = ledger.read_verified()
    assert rows[0] == original
    assert rows[0]["status"] == "pending"
    assert rows[0]["ground_truth"] is None
    assert rows[1]["parent_event_id"] == original["event_id"]
    assert rows[1]["status"] == "completed"
    assert ledger.analysis_ready_records() == [rows[1]]


def test_sqlite_ledger_rejects_orphan_or_non_pending_parent(tmp_path):
    ledger = SQLiteActionOutcomeLedger(tmp_path / "invalid-parent.sqlite3")
    with pytest.raises(ValueError, match="does not reference"):
        ledger.append(ActionOutcomeRecord(**{
            **completed_child(pending_record(3)).__dict__,
            "parent_event_id": "missing-parent",
        }))

    completed = record(4)
    ledger.append(completed)
    child = ActionOutcomeRecord(
        case_id=completed.case_id,
        action=completed.action,
        policy_version=completed.policy_version,
        model_version=completed.model_version,
        baseline_prediction=completed.baseline_prediction,
        status="completed",
        estimated_action_cost=completed.estimated_action_cost,
        propensity=completed.propensity,
        randomized_assignment=completed.randomized_assignment,
        assignment_id=completed.assignment_id,
        context=completed.context,
        final_prediction=1,
        ground_truth=1,
        parent_event_id=completed.event_id,
    )
    with pytest.raises(ValueError, match="pending event"):
        ledger.append(child)


def test_sqlite_ledger_rejects_duplicate_completion_and_context_changes(tmp_path):
    ledger = SQLiteActionOutcomeLedger(tmp_path / "duplicate-parent.sqlite3")
    parent = pending_record(5)
    ledger.append(parent)
    ledger.append(completed_child(parent, "child-one"))
    with pytest.raises(ValueError, match="already has a completion"):
        ledger.append(completed_child(parent, "child-two"))

    other = pending_record(6)
    ledger.append(other)
    altered = ActionOutcomeRecord(
        case_id=other.case_id,
        action=other.action,
        policy_version=other.policy_version,
        model_version=other.model_version,
        baseline_prediction=other.baseline_prediction,
        status="completed",
        estimated_action_cost=other.estimated_action_cost,
        propensity=0.9,
        randomized_assignment=other.randomized_assignment,
        assignment_id=other.assignment_id,
        context=other.context,
        final_prediction=1,
        ground_truth=1,
        parent_event_id=other.event_id,
    )
    with pytest.raises(ValueError, match="preserve the original execution context"):
        ledger.append(altered)


def test_sqlite_ledger_rejects_updates_and_deletes(tmp_path):
    path = tmp_path / "append-only.sqlite3"
    ledger = SQLiteActionOutcomeLedger(path)
    ledger.append(record(401))
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE action_outcomes SET record_json = ? WHERE event_id = ?",
                ("{}", "event-401"),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM action_outcomes WHERE event_id = ?", ("event-401",))
    assert len(ledger.read_verified()) == 1


def test_jsonl_to_sqlite_migration_verifies_and_is_resumable(tmp_path):
    source = ActionOutcomeLedger(tmp_path / "legacy.jsonl")
    source.append(record(101))
    source.append(record(102))
    destination = tmp_path / "migrated.sqlite3"
    first = migrate_jsonl_to_sqlite(source.path, destination)
    second = migrate_jsonl_to_sqlite(source.path, destination)
    assert first["migrated_records"] == 2
    assert first["destination_integrity"] == "verified"
    assert second["migrated_records"] == 0
    assert second["already_present"] == 2
    assert SQLiteActionOutcomeLedger(destination).read_verified() == source.read_verified()
    assert source.path.exists()


def test_jsonl_to_sqlite_migration_refuses_divergent_destination(tmp_path):
    source = ActionOutcomeLedger(tmp_path / "legacy.jsonl")
    source.append(record(201))
    destination = tmp_path / "migrated.sqlite3"
    SQLiteActionOutcomeLedger(destination).append(record(999))
    with pytest.raises(ValueError, match="exact prefix"):
        migrate_jsonl_to_sqlite(source.path, destination)


def test_jsonl_to_sqlite_migration_rejects_tampered_source(tmp_path):
    source_path = tmp_path / "legacy.jsonl"
    source = ActionOutcomeLedger(source_path)
    source.append(record(301))
    source_path.write_text(source_path.read_text().replace('"uncertainty":0.4', '"uncertainty":0.9'))
    with pytest.raises(ValueError, match="integrity"):
        migrate_jsonl_to_sqlite(source_path, tmp_path / "migrated.sqlite3")


def test_jsonl_migration_accepts_legacy_records_without_parent_event_id(tmp_path):
    import json
    from src.verification.action_outcome_ledger import GENESIS_HASH

    source_path = tmp_path / "legacy-pre-linkage.jsonl"
    legacy_row = record(601).validated()
    legacy_row.pop("parent_event_id", None)
    envelope = {
        "previous_hash": GENESIS_HASH,
        "record_hash": ActionOutcomeLedger._digest(legacy_row),
        "record": legacy_row,
    }
    source_path.write_text(json.dumps(envelope, sort_keys=True) + "\\n")

    destination = tmp_path / "legacy-migrated.sqlite3"
    result = migrate_jsonl_to_sqlite(source_path, destination)
    migrated = SQLiteActionOutcomeLedger(destination).read_verified()

    assert result["migrated_records"] == 1
    assert migrated[0]["event_id"] == legacy_row["event_id"]
    assert migrated[0]["parent_event_id"] is None
