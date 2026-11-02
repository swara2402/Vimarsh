import sqlite3

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def operator_headers(token="review-test-token"):
    return {"X-Vimarsh-Reviewer-Token": token}


def evidence_headers(token="evidence-test-token"):
    return {"X-Vimarsh-Evidence-Token": token}


def configure_operator_workflows(monkeypatch, tmp_path):
    monkeypatch.setenv("VIMARSH_REVIEWER_API_TOKEN", "review-test-token")
    monkeypatch.setenv("VIMARSH_REVIEW_QUEUE_DB", str(tmp_path / "reviews.sqlite3"))
    monkeypatch.setenv("VIMARSH_EVIDENCE_API_TOKEN", "evidence-test-token")
    monkeypatch.setenv("VIMARSH_EVIDENCE_DB", str(tmp_path / "evidence.sqlite3"))


def test_operator_endpoints_fail_closed_without_tokens(monkeypatch):
    monkeypatch.delenv("VIMARSH_REVIEWER_API_TOKEN", raising=False)
    monkeypatch.delenv("VIMARSH_EVIDENCE_API_TOKEN", raising=False)
    assert client.get("/v1/reviews").status_code == 503
    assert client.post("/v1/evidence", json={}).status_code == 503


def test_review_queue_claim_completion_and_audit_chain(monkeypatch, tmp_path):
    configure_operator_workflows(monkeypatch, tmp_path)
    assert client.get("/v1/reviews", headers={"X-Vimarsh-Reviewer-Token": "wrong"}).status_code == 401

    created = client.post("/v1/reviews", headers=operator_headers(), json={
        "case_id": "review-case-1",
        "prediction": "class-a",
        "rationale": ["low confidence", "model disagreement"],
        "context": {"uncertainty": 0.42},
        "priority": 4,
    })
    assert created.status_code == 200
    review_id = created.json()["review_id"]

    duplicate = client.post("/v1/reviews", headers=operator_headers(), json={
        "case_id": "review-case-1", "prediction": "class-b", "priority": 1,
    })
    assert duplicate.status_code == 409

    claimed = client.post(
        f"/v1/reviews/{review_id}/claim", headers=operator_headers(),
        json={"reviewer_id": "reviewer-01"},
    )
    assert claimed.status_code == 200
    assert claimed.json()["status"] == "claimed"

    second_claim = client.post(
        f"/v1/reviews/{review_id}/claim", headers=operator_headers(),
        json={"reviewer_id": "reviewer-02"},
    )
    assert second_claim.status_code == 409

    completed = client.post(
        f"/v1/reviews/{review_id}/complete", headers=operator_headers(),
        json={
            "reviewer_id": "reviewer-01",
            "reviewer_note": "Reviewed supporting signals.",
            "final_prediction": "class-a",
            "ground_truth": "class-a",
        },
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"

    history = client.get(f"/v1/reviews/{review_id}/events", headers=operator_headers())
    assert history.status_code == 200
    assert history.json()["integrity"] == "verified"
    assert [event["event_type"] for event in history.json()["events"]] == [
        "created", "claimed", "completed",
    ]

    completed_list = client.get(
        "/v1/reviews?status=completed", headers=operator_headers()
    )
    assert completed_list.status_code == 200
    assert completed_list.json()["cases"][0]["case_id"] == "review-case-1"


def test_evidence_intake_is_append_only_and_digest_is_verified(monkeypatch, tmp_path):
    configure_operator_workflows(monkeypatch, tmp_path)
    payload = {
        "case_id": "evidence-case-1",
        "source_name": "Operator source",
        "source_reference": "https://example.test/source/1",
        "evidence_text": "A manually supplied evidence excerpt.",
        "metadata": {"kind": "manual", "page": 2},
    }
    created = client.post("/v1/evidence", headers=evidence_headers(), json=payload)
    assert created.status_code == 200
    digest = created.json()["content_sha256"]
    assert len(digest) == 64

    listed = client.get(
        "/v1/evidence/evidence-case-1", headers=evidence_headers()
    )
    assert listed.status_code == 200
    item = listed.json()["evidence"][0]
    assert item["content_sha256"] == digest
    assert item["integrity_valid"] is True

    with sqlite3.connect(tmp_path / "evidence.sqlite3") as connection:
        try:
            connection.execute(
                "UPDATE evidence_items SET evidence_text='tampered' WHERE evidence_id=?",
                (created.json()["evidence_id"],),
            )
            assert False, "append-only trigger should reject evidence updates"
        except sqlite3.DatabaseError:
            pass


def test_operator_workflow_capabilities_reflect_server_configuration(monkeypatch, tmp_path):
    configure_operator_workflows(monkeypatch, tmp_path)
    response = client.get("/v1/actions/capabilities")
    assert response.status_code == 200
    actions = response.json()["actions"]
    assert actions["human_review"]["available"] is True
    assert actions["human_review"]["mode"] == "operator_managed_queue"
    assert actions["additional_evidence"]["available"] is True
    assert actions["additional_evidence"]["mode"] == "operator_submitted_evidence"


def test_operator_token_headers_are_allowed_by_cors_and_api_responses_are_hardened():
    response = client.options(
        "/v1/reviews",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-vimarsh-reviewer-token,x-vimarsh-evidence-token",
        },
    )
    assert response.status_code == 200
    allowed = response.headers.get("access-control-allow-headers", "").lower()
    assert "x-vimarsh-reviewer-token" in allowed
    assert "x-vimarsh-evidence-token" in allowed

    health = client.get("/health")
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["x-frame-options"] == "DENY"
    assert health.headers["referrer-policy"] == "no-referrer"
    assert health.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"


def test_crossref_doi_metadata_provider_is_token_protected_and_logged(monkeypatch, tmp_path):
    configure_operator_workflows(monkeypatch, tmp_path)
    from app import main

    monkeypatch.setattr(main, "resolve_crossref_doi", lambda doi: {
        "provider": "Crossref",
        "doi": "10.1234/example",
        "title": "A metadata-only research record",
        "authors": ["Example Author"],
        "publisher": "Example Publisher",
        "year": 2025,
        "work_type": "journal-article",
        "container_title": "Example Journal",
        "url": "https://doi.org/10.1234/example",
        "metadata_only": True,
        "claim_verification": "not_performed",
    })
    response = client.post(
        "/v1/evidence/resolve-doi",
        headers=evidence_headers(),
        json={"case_id": "doi-case-1", "doi": "10.1234/example"},
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "crossref"
    assert response.json()["claim_verification"] == "not_performed"

    evidence = client.get("/v1/evidence/doi-case-1", headers=evidence_headers())
    assert evidence.status_code == 200
    assert evidence.json()["evidence"][0]["integrity_valid"] is True


def test_crossref_doi_provider_rejects_invalid_doi_without_network():
    response = client.post(
        "/v1/evidence/resolve-doi",
        headers=evidence_headers(),
        json={"case_id": "doi-case-invalid", "doi": "not-a-doi"},
    )
    assert response.status_code == 422
