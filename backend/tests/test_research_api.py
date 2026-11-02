import json
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
TOKEN = "test-research-token"
HEADERS = {"X-Vimarsh-Research-Token": TOKEN}


def configure(monkeypatch, tmp_path):
    monkeypatch.setenv("VIMARSH_RESEARCH_API_TOKEN", TOKEN)
    monkeypatch.setenv("VIMARSH_OUTCOME_LEDGER", str(tmp_path / "outcomes.jsonl"))


def completed_payload(index):
    return {
        "case_id": f"case-{index}",
        "action": "alternate_model",
        "policy_version": "policy-v1",
        "model_version": "primary-v1",
        "baseline_prediction": 0,
        "status": "completed",
        "estimated_action_cost": 1.0,
        "propensity": 0.5,
        "randomized_assignment": True,
        "assignment_id": f"trial-{index}",
        "context": {"uncertainty": 0.4, "harm": 1.0},
        "action_result": {"alternate_prediction": 1},
        "final_prediction": 1,
        "ground_truth": 1,
        "observed_cost": 0.9,
        "latency_ms": 12.0,
        "provider_version": "local-runner-v1",
        "event_id": f"event-{index}",
    }


def test_research_endpoints_are_disabled_without_server_token(monkeypatch):
    monkeypatch.delenv("VIMARSH_RESEARCH_API_TOKEN", raising=False)
    response = client.get("/v1/research/outcomes/summary")
    assert response.status_code == 503


def test_research_endpoints_require_valid_token(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    assert client.get("/v1/research/outcomes/summary").status_code == 401
    assert client.get("/v1/research/outcomes/summary", headers=HEADERS).status_code == 200


def test_outcome_write_and_summary_do_not_expose_retrospective_labels(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    response = client.post("/v1/research/outcomes", headers=HEADERS, json=completed_payload(1))
    assert response.status_code == 200
    assert response.json()["ground_truth_stored"] is True
    summary = client.get("/v1/research/outcomes/summary", headers=HEADERS).json()
    assert summary["total_records"] == 1
    assert summary["analysis_ready_records"] == 1
    assert "ground_truth" not in summary


def test_completed_outcome_requires_retrospective_truth(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    payload = completed_payload(2)
    payload["ground_truth"] = None
    response = client.post(
        "/v1/research/outcomes", headers={**HEADERS, "Content-Type": "application/json"},
        content=json.dumps(payload, allow_nan=True),
    )
    assert response.status_code == 422


def test_contextual_benefit_and_batch_allocation_use_logged_outcomes(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    for index in range(12):
        response = client.post(
            "/v1/research/outcomes", headers=HEADERS, json=completed_payload(index)
        )
        assert response.status_code == 200

    estimate = client.post(
        "/v1/research/action-benefit",
        headers=HEADERS,
        json={
            "action": "alternate_model",
            "query_context": {"uncertainty": 0.4, "harm": 1.0},
            "min_effective_samples": 10,
        },
    )
    assert estimate.status_code == 200
    assert estimate.json()["status"] == "recommend_for_further_evaluation"

    allocation = client.post(
        "/v1/research/allocate",
        headers=HEADERS,
        json={
            "cases": [{
                "case_id": "new-case",
                "context": {"uncertainty": 0.4, "harm": 1.0},
                "available_actions": ["alternate_model"],
                "estimated_action_costs": {"alternate_model": 1.0},
            }],
            "total_budget": 1.0,
            "min_effective_samples": 10,
        },
    )
    assert allocation.status_code == 200
    body = allocation.json()
    assert body["mode"] == "offline_research_recommendation_only"
    assert body["execution_status"] == "not_executed"
    assert body["allocation"]["status"] == "allocated"
    assert body["allocation"]["selected"][0]["case_id"] == "new-case"


def test_policy_context_rejects_ground_truth_leakage(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    payload = completed_payload(3)
    payload["context"] = {"uncertainty": 0.4, "harm": 1.0, "ground_truth": 1.0}
    response = client.post(
        "/v1/research/outcomes", headers={**HEADERS, "Content-Type": "application/json"},
        content=json.dumps(payload, allow_nan=True),
    )
    assert response.status_code == 422


def test_explicit_alternate_model_execution_logs_pending_outcome(monkeypatch, tmp_path):
    from app import main

    configure(monkeypatch, tmp_path)

    class FakeRuntime:
        def score_one(self, sample):
            return {
                "prediction": "primary-class",
                "confidence": 0.82,
                "estimated_error_probability": 0.24,
            }

        def run_alternate_models(self, sample):
            return {
                "alternate_models": [{
                    "model_index": 0,
                    "model_type": "FakeAlternate",
                    "prediction": "alternate-class",
                    "confidence": 0.91,
                    "latency_ms": 3.2,
                }],
                "total_latency_ms": 3.2,
                "executed": True,
            }

    monkeypatch.setattr(main, "get_reliability_runtime", lambda: FakeRuntime())
    response = client.post(
        "/v1/research/execute/alternate-model",
        headers=HEADERS,
        json={
            "case_id": "execution-case-1",
            "sample": {"x": 1.0, "y": 2.0},
            "harm": 2.0,
            "estimated_action_cost": 0.5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["execution_status"] == "executed"
    assert body["ground_truth_status"] == "pending_retrospective_evaluation"
    assert body["alternate_models"][0]["prediction"] == "alternate-class"
    summary = client.get("/v1/research/outcomes/summary", headers=HEADERS).json()
    assert summary["total_records"] == 1
    assert summary["analysis_ready_records"] == 0


def test_batch_allocator_excludes_actions_without_operational_providers(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    for index in range(12):
        client.post("/v1/research/outcomes", headers=HEADERS, json=completed_payload(index))
    response = client.post(
        "/v1/research/allocate",
        headers=HEADERS,
        json={
            "cases": [{
                "case_id": "provider-check",
                "context": {"uncertainty": 0.4, "harm": 1.0},
                "available_actions": ["alternate_model", "human_review", "additional_evidence"],
                "estimated_action_costs": {
                    "alternate_model": 1.0,
                    "human_review": 0.5,
                    "additional_evidence": 0.25,
                },
            }],
            "total_budget": 2.0,
            "min_effective_samples": 10,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert all(item["action"] == "alternate_model" for item in body["allocation"]["selected"])
    excluded = [item for item in body["diagnostics"] if item.get("status") == "excluded"]
    assert {item["action"] for item in excluded} == {"human_review", "additional_evidence"}


@pytest.mark.parametrize("bad_number", [float("nan"), float("inf"), float("-inf")])
def test_research_api_rejects_non_finite_allocation_context(monkeypatch, tmp_path, bad_number):
    configure(monkeypatch, tmp_path)
    payload = {
        "cases": [{
            "case_id": "finite-check",
            "context": {"uncertainty": bad_number},
            "available_actions": ["alternate_model"],
            "estimated_action_costs": {"alternate_model": 1.0},
        }],
        "total_budget": 1.0,
    }
    response = client.post(
        "/v1/research/allocate", headers={**HEADERS, "Content-Type": "application/json"},
        content=json.dumps(payload, allow_nan=True),
    )
    assert response.status_code == 422


@pytest.mark.parametrize("bad_cost", [-0.01, float("nan"), float("inf")])
def test_allocation_rejects_negative_or_non_finite_action_cost(monkeypatch, tmp_path, bad_cost):
    configure(monkeypatch, tmp_path)
    payload = {
        "cases": [{
            "case_id": "bad-cost",
            "context": {"uncertainty": 0.4},
            "available_actions": ["alternate_model"],
            "estimated_action_costs": {"alternate_model": bad_cost},
        }],
        "total_budget": 1.0,
    }
    response = client.post(
        "/v1/research/allocate", headers={**HEADERS, "Content-Type": "application/json"},
        content=json.dumps(payload, allow_nan=True),
    )
    assert response.status_code == 422


def test_allocation_rejects_negative_capacity(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    response = client.post(
        "/v1/research/allocate",
        headers=HEADERS,
        json={
            "cases": [{
                "case_id": "capacity-check",
                "context": {"uncertainty": 0.4},
                "available_actions": ["alternate_model"],
                "estimated_action_costs": {"alternate_model": 1.0},
            }],
            "total_budget": 1.0,
            "action_capacities": {"alternate_model": -1},
        },
    )
    assert response.status_code == 422


def test_allocation_rejects_duplicate_cases_actions_and_orphan_costs(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    case = {
        "case_id": "duplicate-check",
        "context": {"uncertainty": 0.4},
        "available_actions": ["alternate_model"],
        "estimated_action_costs": {"alternate_model": 1.0},
    }
    duplicate_case = client.post(
        "/v1/research/allocate", headers=HEADERS,
        json={"cases": [case, case], "total_budget": 2.0},
    )
    duplicate_action = client.post(
        "/v1/research/allocate", headers=HEADERS,
        json={"cases": [{**case, "available_actions": ["alternate_model", "alternate_model"]}],
              "total_budget": 2.0},
    )
    orphan_cost = client.post(
        "/v1/research/allocate", headers=HEADERS,
        json={"cases": [{**case, "available_actions": [], "estimated_action_costs": {"alternate_model": 1.0}}],
              "total_budget": 2.0},
    )
    assert duplicate_case.status_code == 422
    assert duplicate_action.status_code == 422
    assert orphan_cost.status_code == 422


def test_outcome_api_rejects_non_finite_context(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    payload = completed_payload(44)
    payload["context"] = {"uncertainty": float("nan")}
    response = client.post(
        "/v1/research/outcomes", headers={**HEADERS, "Content-Type": "application/json"},
        content=json.dumps(payload, allow_nan=True),
    )
    assert response.status_code == 422



def test_pending_outcome_can_be_completed_as_a_linked_immutable_event(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    pending = completed_payload(70)
    pending.update({
        "status": "pending",
        "final_prediction": None,
        "ground_truth": None,
        "observed_cost": None,
        "event_id": "pending-execution-70",
    })
    created = client.post("/v1/research/outcomes", headers=HEADERS, json=pending)
    assert created.status_code == 200

    completed = client.post(
        "/v1/research/outcomes/pending-execution-70/complete",
        headers=HEADERS,
        json={
            "final_prediction": 1,
            "ground_truth": 1,
            "observed_cost": 0.8,
            "latency_ms": 15.0,
            "provider_version": "reviewed-local-run-v1",
        },
    )
    assert completed.status_code == 200
    body = completed.json()
    assert body["parent_event_id"] == "pending-execution-70"
    assert body["status"] == "completed"
    assert body["analysis_eligible"] is True

    rows = __import__("app.research_services", fromlist=["get_ledger"]).get_ledger().read_verified()
    original = next(row for row in rows if row["event_id"] == "pending-execution-70")
    child = next(row for row in rows if row["event_id"] == body["event_id"])
    assert original["status"] == "pending"
    assert original["ground_truth"] is None
    assert child["parent_event_id"] == original["event_id"]
    assert child["status"] == "completed"
    assert child["ground_truth"] == 1
    summary = client.get("/v1/research/outcomes/summary", headers=HEADERS).json()
    assert summary["total_records"] == 2
    assert summary["completed_records"] == 1
    assert summary["analysis_ready_records"] == 1


def test_pending_outcome_completion_rejects_missing_non_pending_and_duplicate_events(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    payload = completed_payload(71)
    payload.update({"status": "pending", "final_prediction": None, "ground_truth": None})
    client.post("/v1/research/outcomes", headers=HEADERS, json=payload)
    completion = {
        "final_prediction": 1,
        "ground_truth": 1,
        "observed_cost": 1.0,
    }
    first = client.post(
        f"/v1/research/outcomes/{payload['event_id']}/complete",
        headers=HEADERS,
        json=completion,
    )
    second = client.post(
        f"/v1/research/outcomes/{payload['event_id']}/complete",
        headers=HEADERS,
        json=completion,
    )
    missing = client.post(
        "/v1/research/outcomes/not-a-real-event/complete",
        headers=HEADERS,
        json=completion,
    )
    assert first.status_code == 200
    assert second.status_code == 409
    assert missing.status_code == 409

    completed = completed_payload(72)
    client.post("/v1/research/outcomes", headers=HEADERS, json=completed)
    not_pending = client.post(
        f"/v1/research/outcomes/{completed['event_id']}/complete",
        headers=HEADERS,
        json=completion,
    )
    assert not_pending.status_code == 409


def test_protected_research_api_supports_transactional_sqlite_ledger(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    monkeypatch.setenv("VIMARSH_OUTCOME_LEDGER_BACKEND", "sqlite")
    monkeypatch.setenv("VIMARSH_OUTCOME_LEDGER", str(tmp_path / "outcomes.sqlite3"))

    first = client.post(
        "/v1/research/outcomes", headers=HEADERS, json=completed_payload(501)
    )
    second = client.post(
        "/v1/research/outcomes", headers=HEADERS, json=completed_payload(502)
    )
    summary = client.get("/v1/research/outcomes/summary", headers=HEADERS)

    assert first.status_code == 200
    assert second.status_code == 200
    assert summary.status_code == 200
    assert summary.json()["total_records"] == 2
    assert summary.json()["analysis_ready_records"] == 2
    assert summary.json()["ledger_integrity"] == "verified"


def test_pending_outcome_completion_works_with_sqlite_ledger(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    monkeypatch.setenv("VIMARSH_OUTCOME_LEDGER_BACKEND", "sqlite")
    monkeypatch.setenv("VIMARSH_OUTCOME_LEDGER", str(tmp_path / "outcomes.sqlite3"))
    pending = completed_payload(503)
    pending.update({
        "status": "pending",
        "final_prediction": None,
        "ground_truth": None,
        "observed_cost": None,
        "event_id": "sqlite-pending-event",
    })
    created = client.post("/v1/research/outcomes", headers=HEADERS, json=pending)
    assert created.status_code == 200

    completed = client.post(
        "/v1/research/outcomes/sqlite-pending-event/complete",
        headers=HEADERS,
        json={"final_prediction": 1, "ground_truth": 1, "observed_cost": 0.8},
    )
    assert completed.status_code == 200
    assert completed.json()["parent_event_id"] == "sqlite-pending-event"
    summary = client.get("/v1/research/outcomes/summary", headers=HEADERS).json()
    assert summary["total_records"] == 2
    assert summary["completed_records"] == 1
    assert summary["analysis_ready_records"] == 1


def test_research_ledger_backend_rejects_unknown_configuration(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    monkeypatch.setenv("VIMARSH_OUTCOME_LEDGER_BACKEND", "remote")
    response = client.get("/v1/research/outcomes/summary", headers=HEADERS)
    assert response.status_code == 503
