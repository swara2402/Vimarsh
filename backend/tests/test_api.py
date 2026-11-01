from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def payload(**overrides):
    data = {
        "prediction": "class_a",
        "confidence": 0.91,
        "estimated_error_probability": 0.2,
        "harm": 1.0,
        "available_actions": [],
        "estimated_action_costs": {},
        "remaining_budget": 0.0,
    }
    data.update(overrides)
    return data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["mode"] == "recommendation_only"


def test_low_risk_prediction_is_accepted():
    response = client.post("/v1/decisions", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "accept"
    assert body["recommendation_only"] is True
    assert body["execution_status"] == "not_executed"
    assert body["simulation"] is False


def test_feasible_alternate_model_is_recommended_for_high_risk():
    response = client.post(
        "/v1/decisions",
        json=payload(
            estimated_error_probability=0.9,
            harm=2.0,
            available_actions=["alternate_model"],
            estimated_action_costs={"alternate_model": 2.5},
            remaining_budget=3.0,
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "request_alternate_model"
    assert body["eligible_actions"] == ["alternate_model"]


def test_unavailable_or_over_budget_action_leads_to_abstention():
    response = client.post(
        "/v1/decisions",
        json=payload(
            estimated_error_probability=0.9,
            harm=2.0,
            available_actions=["alternate_model"],
            estimated_action_costs={"alternate_model": 5.0},
            remaining_budget=1.0,
        ),
    )
    assert response.status_code == 200
    assert response.json()["action"] == "abstain"


def test_human_review_and_evidence_are_not_claimed_as_operational():
    response = client.post(
        "/v1/decisions",
        json=payload(
            estimated_error_probability=0.9,
            harm=2.0,
            available_actions=["human_review", "additional_evidence"],
            remaining_budget=100.0,
        ),
    )
    body = response.json()
    assert body["action"] == "abstain"
    assert any("no operational reviewer" in item for item in body["rationale"])
    assert any("no operational evidence" in item for item in body["rationale"])


def test_unknown_fields_are_rejected_so_labels_cannot_leak_in():
    response = client.post("/v1/decisions", json=payload(true_label=1))
    assert response.status_code == 422


def test_invalid_probability_and_negative_cost_are_rejected():
    bad_probability = client.post(
        "/v1/decisions", json=payload(estimated_error_probability=1.2)
    )
    bad_cost = client.post(
        "/v1/decisions",
        json=payload(
            estimated_action_costs={"alternate_model": -1.0},
            available_actions=["alternate_model"],
        ),
    )
    assert bad_probability.status_code == 422
    assert bad_cost.status_code == 422
