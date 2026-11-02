import json
import pytest
from src.verification.action_outcome_ledger import ActionOutcomeLedger, ActionOutcomeRecord

def record(**overrides):
    values = dict(case_id="case-001", action="alternate_model", policy_version="policy-v1",
        model_version="rf-v2", baseline_prediction=0, status="completed",
        estimated_action_cost=1.2, propensity=0.25, randomized_assignment=True,
        assignment_id="trial-17", context={"uncertainty": 0.4, "disagreement": 1.0},
        action_result={"model": "rf-v2"}, final_prediction=1, ground_truth=1,
        observed_cost=1.1, latency_ms=24.5, provider_version="runner-v1", event_id="event-001")
    values.update(overrides)
    return ActionOutcomeRecord(**values)

def test_append_and_verify_hash_chain(tmp_path):
    ledger = ActionOutcomeLedger(tmp_path / "events.jsonl")
    ledger.append(record())
    ledger.append(record(event_id="event-002", case_id="case-002"))
    assert [r["event_id"] for r in ledger.read_verified()] == ["event-001", "event-002"]
    assert len(ledger.analysis_ready_records()) == 2

def test_duplicate_event_id_is_rejected(tmp_path):
    ledger = ActionOutcomeLedger(tmp_path / "events.jsonl")
    ledger.append(record())
    with pytest.raises(ValueError, match="already exists"):
        ledger.append(record())

@pytest.mark.parametrize("context", [{"true_label": 1}, {"ground_truth": 0}, {"y_true": 1}, {"target": 1}])
def test_retrospective_labels_cannot_enter_policy_context(tmp_path, context):
    with pytest.raises(ValueError, match="retrospective labels"):
        ActionOutcomeLedger(tmp_path / "events.jsonl").append(record(context=context))

def test_pending_record_cannot_include_ground_truth(tmp_path):
    with pytest.raises(ValueError, match="only permitted for completed"):
        ActionOutcomeLedger(tmp_path / "events.jsonl").append(record(status="pending", final_prediction=None, ground_truth=1))

def test_completed_record_requires_final_prediction_and_ground_truth(tmp_path):
    with pytest.raises(ValueError, match="require final_prediction"):
        ActionOutcomeLedger(tmp_path / "events.jsonl").append(record(final_prediction=None, ground_truth=None))

def test_propensity_and_randomization_metadata_are_validated(tmp_path):
    with pytest.raises(ValueError, match="propensity"):
        ActionOutcomeLedger(tmp_path / "events.jsonl").append(record(propensity=0))
    with pytest.raises(ValueError, match="assignment_id"):
        ActionOutcomeLedger(tmp_path / "events.jsonl").append(record(randomized_assignment=True, assignment_id=None))

def test_analysis_ready_excludes_missing_propensity_and_pending(tmp_path):
    ledger = ActionOutcomeLedger(tmp_path / "events.jsonl")
    ledger.append(record(event_id="eligible"))
    ledger.append(record(event_id="no-propensity", propensity=None))
    ledger.append(record(event_id="pending", status="pending", final_prediction=None, ground_truth=None))
    assert [r["event_id"] for r in ledger.analysis_ready_records()] == ["eligible"]

def test_tampering_is_detected(tmp_path):
    path = tmp_path / "events.jsonl"
    ledger = ActionOutcomeLedger(path)
    ledger.append(record())
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["record"]["observed_cost"] = 999
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record hash mismatch"):
        ledger.read_verified()

def test_invalid_numeric_values_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        ActionOutcomeLedger(tmp_path / "events.jsonl").append(record(latency_ms=float("nan")))
