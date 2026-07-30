"""Integration tests for rule evaluation persistence and the read API (DB-backed)."""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.main import create_app
from app.api.dependencies import get_session_factory
from app.repositories.profile_run_repository import ProfileRunRepository
from app.repositories.rule_repository import RuleRepository
from app.models.rule_evaluation import RuleEvaluation
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
    rid = uuid.uuid4()
    repo = ProfileRunRepository(db_session)
    repo.create_run(rid, "Payments", "test_upload.csv", "csv")
    db_session.commit()
    yield rid
    db_session.execute(delete(RuleEvaluation).where(RuleEvaluation.run_id == rid))
    db_session.execute(delete(ProfileRun).where(ProfileRun.id == rid))
    db_session.commit()


@pytest.fixture
def approved_rule_definition(db_session):
    """A pre-existing active rule definition, standing in for an approved suggestion."""
    suffix = uuid.uuid4().hex[:8]
    rule_key = f"amount_non_negative_{suffix}"
    rule_def = RuleDefinition(
        rule_key=rule_key,
        domain="Payments",
        secondary_domain=None,
        definition_json={"rule_key": rule_key, "type": "numeric_range", "target_column": "amount", "min_value": 0},
        source="approved_suggestion",
        status="active",
    )
    db_session.add(rule_def)
    db_session.commit()
    yield rule_def
    # Evaluations created during the test reference this row via FK — clear those first.
    db_session.execute(delete(RuleEvaluation).where(RuleEvaluation.rule_definition_id == rule_def.id))
    db_session.execute(delete(RuleDefinition).where(RuleDefinition.id == rule_def.id))
    db_session.commit()


class TestRuleEvaluationPersistence:
    """Test that RuleRepository.persist_evaluations actually writes rows, and the read API returns them."""

    def test_persist_and_list_via_repository(self, db_session, run_id, approved_rule_definition):
        rule_repo = RuleRepository(db_session)
        rule_repo.persist_evaluations(
            run_id,
            evaluations=[
                {
                    "rule_key": "country_non_null",
                    "rule_type": "non_null",
                    "source": "domain_configuration",
                    "records_checked": 40,
                    "pass_count": 40,
                    "fail_count": 0,
                    "score": 1.0,
                    "target_columns": ["country"],
                    "error": None,
                },
                {
                    "rule_key": approved_rule_definition.rule_key,
                    "rule_type": "numeric_range",
                    "source": "approved_suggestion",
                    "records_checked": 40,
                    "pass_count": 38,
                    "fail_count": 2,
                    "score": 0.95,
                    "target_columns": ["amount"],
                    "error": None,
                },
            ],
            rule_definition_ids_by_key={approved_rule_definition.rule_key: approved_rule_definition.id},
        )
        db_session.commit()

        rows = rule_repo.list_evaluations(run_id)
        assert len(rows) == 2

        yaml_row = next(r for r in rows if r.rule_key == "country_non_null")
        assert yaml_row.source == "domain_configuration"
        assert yaml_row.rule_definition_id is None  # YAML rules never get a rule_definitions row
        assert yaml_row.fail_count == 0

        approved_row = next(r for r in rows if r.rule_key == approved_rule_definition.rule_key)
        assert approved_row.source == "approved_suggestion"
        assert approved_row.rule_definition_id == approved_rule_definition.id
        assert approved_row.fail_count == 2

    def test_get_rule_evaluations_endpoint(self, client, db_session, run_id, approved_rule_definition):
        rule_repo = RuleRepository(db_session)
        rule_repo.persist_evaluations(
            run_id,
            evaluations=[{
                "rule_key": approved_rule_definition.rule_key,
                "rule_type": "numeric_range",
                "source": "approved_suggestion",
                "records_checked": 10,
                "pass_count": 9,
                "fail_count": 1,
                "score": 0.9,
                "target_columns": ["amount"],
                "error": None,
            }],
            rule_definition_ids_by_key={approved_rule_definition.rule_key: approved_rule_definition.id},
        )
        db_session.commit()

        resp = client.get(f"/api/v1/profile-runs/{run_id}/rule-evaluations")
        assert resp.status_code == 200
        evaluations = resp.json()["evaluations"]
        assert len(evaluations) == 1
        assert evaluations[0]["rule_key"] == approved_rule_definition.rule_key
        assert evaluations[0]["fail_count"] == 1
        assert evaluations[0]["rule_definition_id"] == str(approved_rule_definition.id)

    def test_get_rule_evaluations_run_not_found(self, client):
        resp = client.get(f"/api/v1/profile-runs/{uuid.uuid4()}/rule-evaluations")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "RUN_NOT_FOUND"
