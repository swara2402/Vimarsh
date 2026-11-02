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
    response = client.post("/v1/research/outcomes", headers=HEADERS, json=payload)
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
    response = client.post("/v1/research/outcomes", headers=HEADERS, json=payload)
    assert response.status_code == 422
