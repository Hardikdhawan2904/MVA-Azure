"""Rule repository — persists rule suggestions and approved rule definitions to PostgreSQL."""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.rule_suggestion import RuleSuggestion
from app.models.rule_definition import RuleDefinition
from app.models.rule_evaluation import RuleEvaluation


class RuleRepository:
    """Persists rule suggestions and definitions to PostgreSQL."""

    def __init__(self, session: Session):
        self._session = session

    def persist_suggestions(self, run_id: uuid.UUID, suggestions: list[dict[str, Any]]) -> None:
        """Bulk insert rule suggestions produced for a run."""
        for s in suggestions:
            suggestion = RuleSuggestion(
                run_id=run_id,
                suggested_definition_json=s,
                confidence=s.get("confidence", 0.0),
                status=s.get("status", "proposed"),
                evidence_json={
                    "reasoning": s.get("reasoning", ""),
                    "target_columns": s.get("target_columns", []),
                    "engine_compatible": s.get("engine_compatible", True),
                },
            )
            self._session.add(suggestion)
        self._session.flush()

    def get_suggestion(self, suggestion_id: uuid.UUID) -> RuleSuggestion | None:
        """Fetch a single suggestion by id."""
        return self._session.get(RuleSuggestion, suggestion_id)

    def list_suggestions(self, status: str | None = None) -> list[RuleSuggestion]:
        """List rule suggestions, optionally filtered by status."""
        stmt = select(RuleSuggestion).order_by(RuleSuggestion.created_at.desc())
        if status:
            stmt = stmt.where(RuleSuggestion.status == status)
        return list(self._session.scalars(stmt).all())

    def list_suggestions_for_run(self, run_id: uuid.UUID) -> list[RuleSuggestion]:
        """List rule suggestions generated for a specific run."""
        stmt = select(RuleSuggestion).where(RuleSuggestion.run_id == run_id).order_by(RuleSuggestion.created_at.desc())
        return list(self._session.scalars(stmt).all())

    def update_suggestion_status(
        self,
        suggestion_id: uuid.UUID,
        status: str,
        comment: str | None = None,
        rejection_reason: str | None = None,
    ) -> RuleSuggestion | None:
        """Transition a suggestion's status and record the review outcome."""
        suggestion = self._session.get(RuleSuggestion, suggestion_id)
        if not suggestion:
            return None
        suggestion.status = status
        suggestion.comment = comment
        suggestion.rejection_reason = rejection_reason
        suggestion.reviewed_at = datetime.now(timezone.utc)
        self._session.flush()
        return suggestion

    def create_rule_definition_from_suggestion(self, suggestion: RuleSuggestion, domain: str) -> RuleDefinition:
        """Materialize an approved suggestion into an active, engine-loadable rule definition."""
        definition_json = dict(suggestion.suggested_definition_json)
        rule_def = RuleDefinition(
            rule_key=definition_json.get("rule_key", f"approved_{suggestion.id}"),
            domain=domain,
            secondary_domain=None,
            definition_json=definition_json,
            source="approved_suggestion",
            status="active",
            approved_at=datetime.now(timezone.utc),
        )
        self._session.add(rule_def)
        self._session.flush()
        return rule_def

    def list_active_definitions(self, domain: str, secondary_domain: str | None = None) -> list[RuleDefinition]:
        """List active approved rule definitions for a domain, to be evaluated on future runs."""
        stmt = select(RuleDefinition).where(
            RuleDefinition.domain == domain,
            RuleDefinition.status == "active",
        )
        if secondary_domain:
            stmt = stmt.where(
                (RuleDefinition.secondary_domain == secondary_domain)
                | (RuleDefinition.secondary_domain.is_(None))
            )
        return list(self._session.scalars(stmt).all())

    def persist_evaluations(
        self,
        run_id: uuid.UUID,
        evaluations: list[dict[str, Any]],
        rule_definition_ids_by_key: dict[str, uuid.UUID] | None = None,
    ) -> None:
        """
        Bulk insert rule evaluation results for a run.

        `evaluations` is RuleEvaluationResult.to_dict() output (rule_key, rule_type,
        records_checked, pass_count, fail_count, score, target_columns, source, error).
        `rule_definition_ids_by_key` maps rule_key -> rule_definitions.id for rules that
        have one (only approved-suggestion-sourced rules do) — the FK is only set when
        both the source is approved_suggestion and a matching id is found.
        """
        def_ids = rule_definition_ids_by_key or {}
        for e in evaluations:
            rule_definition_id = (
                def_ids.get(e["rule_key"]) if e.get("source") == "approved_suggestion" else None
            )
            evaluation = RuleEvaluation(
                run_id=run_id,
                rule_definition_id=rule_definition_id,
                rule_key=e["rule_key"],
                rule_type=e["rule_type"],
                source=e.get("source", "unknown"),
                records_checked=e["records_checked"],
                pass_count=e["pass_count"],
                fail_count=e["fail_count"],
                score=e["score"],
                evidence_json={"target_columns": e.get("target_columns", []), "error": e.get("error")},
            )
            self._session.add(evaluation)
        self._session.flush()

    def list_evaluations(self, run_id: uuid.UUID) -> list[RuleEvaluation]:
        """List rule evaluations persisted for a run."""
        stmt = select(RuleEvaluation).where(RuleEvaluation.run_id == run_id)
        return list(self._session.scalars(stmt).all())

    def commit(self) -> None:
        """Commit the current transaction."""
        self._session.commit()
