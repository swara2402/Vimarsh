from fastapi.testclient import TestClient
import pytest

from app import main
from app.main import app

client = TestClient(app)


class FakeRuntime:
    def score_one(self, sample):
        assert sample == {"feature_a": 1.0, "feature_b": 2.0}
        return {
            "prediction": "positive",
            "confidence": 0.88,
            "estimated_error_probability": 0.72,
            "base_model_count": 3,
        }


def test_reliability_endpoint_requires_configured_bundle(monkeypatch):
    monkeypatch.setattr(main, "get_reliability_runtime", lambda: (_ for _ in ()).throw(FileNotFoundError("missing")))
    response = client.post("/v1/reliability/decisions", json={
        "sample": {"feature_a": 1.0, "feature_b": 2.0},
        "harm": 2.0,
    })
    assert response.status_code == 503


def test_reliability_endpoint_uses_bundle_scores_and_returns_recommendation(monkeypatch):
    monkeypatch.setattr(main, "get_reliability_runtime", lambda: FakeRuntime())
    response = client.post("/v1/reliability/decisions", json={
        "sample": {"feature_a": 1.0, "feature_b": 2.0},
        "harm": 2.0,
        "available_actions": ["alternate_model"],
        "estimated_action_costs": {"alternate_model": 1.0},
        "remaining_budget": 2.0,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "positive"
    assert body["estimated_error_probability"] == pytest.approx(0.72)
    assert body["baseline_risk"] == pytest.approx(1.44)
    assert body["action"] == "request_alternate_model"
    assert body["base_model_count"] == 3
    assert body["execution_status"] == "not_executed"


def test_reliability_endpoint_rejects_unknown_fields(monkeypatch):
    monkeypatch.setattr(main, "get_reliability_runtime", lambda: FakeRuntime())
    response = client.post("/v1/reliability/decisions", json={
        "sample": {"feature_a": 1.0, "feature_b": 2.0},
        "harm": 2.0,
        "true_label": 1,
    })
    assert response.status_code == 422


def test_reliability_schema_returns_bundle_feature_contract(monkeypatch):
    class SchemaRuntime:
        feature_names = ["feature_a", "feature_b"]
        primary_model_index = 1
        base_models = [object(), object(), object()]

    monkeypatch.setattr(main, "get_reliability_runtime", lambda: SchemaRuntime())
    response = client.get("/v1/reliability/schema")
    assert response.status_code == 200
    assert response.json() == {
        "feature_names": ["feature_a", "feature_b"],
        "primary_model_index": 1,
        "base_model_count": 3,
        "artifact_version": 2,
    }


def test_dashboard_origin_is_allowed_by_default():
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
