"""Integration tests for rule suggestion approval/rejection endpoints (DB-backed)."""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.main import create_app
from app.api.dependencies import get_session_factory
from app.repositories.profile_run_repository import ProfileRunRepository
from app.repositories.rule_repository import RuleRepository
from app.models.rule_suggestion import RuleSuggestion
from app.models.rule_definition import RuleDefinition
from app.models.profile_run import ProfileRun


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session():
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def run_id(db_session):
    """Create a lightweight profile run row to attach suggestions to."""
    rid = uuid.uuid4()
    repo = ProfileRunRepository(db_session)
    repo.create_run(rid, "Payments", "test_upload.csv", "csv")
    db_session.commit()
    yield rid
    db_session.execute(delete(ProfileRun).where(ProfileRun.id == rid))
    db_session.commit()


@pytest.fixture
def seeded_suggestions(db_session, run_id):
    """Seed three rule suggestions on `run_id`: two proposed, one already approved."""
    suffix = run_id.hex[:8]
    keys = {
        "positive": f"amount_positive_{suffix}",
        "non_null": f"customer_id_non_null_{suffix}",
        "approved": f"status_allowed_{suffix}",
    }

    repo = RuleRepository(db_session)
    repo.persist_suggestions(run_id, [
        {
            "rule_key": keys["positive"],
            "type": "numeric_range",
            "description": "Amount should be positive",
            "reasoning": "All sampled values are positive",
            "confidence": 0.88,
            "severity": "medium",
            "target_columns": ["amount"],
            "target_column": "amount",
            "min_value": 0,
            "inclusive_min": False,
            "engine_compatible": True,
            "status": "proposed",
        },
        {
            "rule_key": keys["non_null"],
            "type": "non_null",
            "description": "Customer ID should not be null",
            "reasoning": "No nulls observed in the sample",
            "confidence": 0.92,
            "severity": "high",
            "target_columns": ["customer_id"],
            "target_column": "customer_id",
            "engine_compatible": True,
            "status": "proposed",
        },
        {
            "rule_key": keys["approved"],
            "type": "allowed_values",
            "description": "Status must be valid",
            "reasoning": "Only two distinct values observed",
            "confidence": 0.75,
            "severity": "low",
            "target_columns": ["status"],
            "target_column": "status",
            "values": ["approved", "declined"],
            "engine_compatible": True,
            "status": "approved",  # Already approved
        },
    ])
    db_session.commit()

    by_key = {
        s.suggested_definition_json["rule_key"]: s
        for s in repo.list_suggestions()
        if s.run_id == run_id
    }
    ids = {name: str(by_key[key].id) for name, key in keys.items()}

    yield ids

    db_session.execute(delete(RuleDefinition).where(RuleDefinition.rule_key.in_(keys.values())))
    db_session.execute(delete(RuleSuggestion).where(RuleSuggestion.run_id == run_id))
    db_session.commit()


class TestRuleSuggestionEndpoints:
    """Test rule suggestion API endpoints against the real database."""

    def test_list_suggestions_includes_seeded(self, client, seeded_suggestions):
        resp = client.get("/api/v1/rule-suggestions")
        assert resp.status_code == 200
        returned_ids = {s["suggestion_id"] for s in resp.json()["suggestions"]}
        assert set(seeded_suggestions.values()) <= returned_ids

    def test_list_suggestions_filter_by_status(self, client, seeded_suggestions):
        resp = client.get("/api/v1/rule-suggestions?status=proposed")
        assert resp.status_code == 200
        returned_ids = {s["suggestion_id"] for s in resp.json()["suggestions"]}
        assert seeded_suggestions["positive"] in returned_ids
        assert seeded_suggestions["non_null"] in returned_ids
        assert seeded_suggestions["approved"] not in returned_ids

    def test_get_suggestion(self, client, seeded_suggestions):
        resp = client.get(f"/api/v1/rule-suggestions/{seeded_suggestions['positive']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["definition"]["type"] == "numeric_range"
        assert data["status"] == "proposed"

    def test_suggestion_ids_visible_in_run_result(self, client, run_id, seeded_suggestions):
        """The pipeline result (what the orchestrator returns verbatim) must carry real suggestion IDs."""
        resp = client.get(f"/api/v1/profile-runs/{run_id}/result")
        assert resp.status_code == 200
        returned_ids = {s["suggestion_id"] for s in resp.json()["rule_suggestions"]}
        assert set(seeded_suggestions.values()) == returned_ids

    def test_approve_proposed_rule(self, client, seeded_suggestions, db_session):
        resp = client.post(
            f"/api/v1/rule-suggestions/{seeded_suggestions['positive']}/approve",
            json={"comment": "Confirmed by data team"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["new_status"] == "approved"
        assert data["rule_definition_id"]

        rule_def = db_session.get(RuleDefinition, uuid.UUID(data["rule_definition_id"]))
        assert rule_def is not None
        assert rule_def.source == "approved_suggestion"
        assert rule_def.status == "active"
        assert rule_def.domain == "Payments"

        # And the suggestion itself now reflects the approval
        get_resp = client.get(f"/api/v1/rule-suggestions/{seeded_suggestions['positive']}")
        assert get_resp.json()["status"] == "approved"

    def test_reject_proposed_rule(self, client, seeded_suggestions):
        resp = client.post(
            f"/api/v1/rule-suggestions/{seeded_suggestions['non_null']}/reject",
            json={"reason": "Not applicable to this domain"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["new_status"] == "rejected"

        get_resp = client.get(f"/api/v1/rule-suggestions/{seeded_suggestions['non_null']}")
        assert get_resp.json()["status"] == "rejected"
        assert get_resp.json()["rejection_reason"] == "Not applicable to this domain"

    def test_cannot_approve_already_approved(self, client, seeded_suggestions):
        resp = client.post(
            f"/api/v1/rule-suggestions/{seeded_suggestions['approved']}/approve",
            json={},
        )
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "INVALID_RULE_TRANSITION"

    def test_cannot_reject_already_approved(self, client, seeded_suggestions):
        resp = client.post(
            f"/api/v1/rule-suggestions/{seeded_suggestions['approved']}/reject",
            json={"reason": "Too late"},
        )
        assert resp.status_code == 409

    def test_not_found_suggestion(self, client):
        resp = client.post(
            f"/api/v1/rule-suggestions/{uuid.uuid4()}/approve",
            json={},
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "RULE_SUGGESTION_NOT_FOUND"

    def test_malformed_suggestion_id(self, client):
        resp = client.get("/api/v1/rule-suggestions/not-a-uuid")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "RULE_SUGGESTION_NOT_FOUND"
