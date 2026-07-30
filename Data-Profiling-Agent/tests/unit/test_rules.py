"""Tests for business rule engine, loader, and approval."""

import pytest
import pandas as pd
from pathlib import Path

from app.core.enums import RuleType, RuleSource, RuleSuggestionStatus
from app.core.exceptions import InvalidRuleTransitionError
from app.repositories.configuration_repository import ConfigurationRepository
from app.services.rules.rule_loader import RuleLoader, RuleDefinition
from app.services.rules.rule_engine import RuleEngine
from app.services.rules.approval_service import ApprovalService
from app.services.rules.suggestion_generator import SuggestedRule


@pytest.fixture
def config_repo() -> ConfigurationRepository:
    return ConfigurationRepository(config_dir=Path(__file__).parent.parent.parent / "config")


@pytest.fixture
def engine() -> RuleEngine:
    return RuleEngine()


class TestRuleLoader:
    """Test rule loading from configuration."""

    def test_load_payments_domain_rules(self, config_repo):
        loader = RuleLoader(config_repo)
        rules = loader.load_domain_rules("Payments")
        assert len(rules) >= 1
        assert all(r.source == RuleSource.DOMAIN_CONFIGURATION for r in rules)
        assert all(r.active is True for r in rules)

    def test_load_request_rules(self, config_repo):
        loader = RuleLoader(config_repo)
        request_rules = [
            {"rule_key": "test_rule", "type": "non_null", "target_column": "amount"},
            {"rule_key": "test_regex", "type": "regex_match", "target_column": "code", "pattern": "^[A-Z]{3}$"},
        ]
        rules = loader.load_request_rules(request_rules, "Payments")
        assert len(rules) == 2
        assert rules[0].source == RuleSource.REQUEST
        assert rules[0].rule_type == RuleType.NON_NULL

    def test_invalid_rule_type_skipped(self, config_repo):
        loader = RuleLoader(config_repo)
        request_rules = [{"rule_key": "bad", "type": "nonexistent_type"}]
        rules = loader.load_request_rules(request_rules, "Payments")
        assert len(rules) == 0


class TestRuleEngine:
    """Test deterministic rule evaluation."""

    def test_non_null_rule(self, engine):
        df = pd.DataFrame({"amount": ["100", "200", None, "400", None]})
        rule = RuleDefinition(
            rule_key="test_non_null", rule_type=RuleType.NON_NULL,
            source=RuleSource.REQUEST, domain="Payments", secondary_domain=None,
            parameters={"target_column": "amount"},
        )
        results = engine.evaluate(df, [rule])
        assert len(results) == 1
        assert results[0].records_checked == 5
        assert results[0].fail_count == 2
        assert results[0].pass_count == 3
        assert results[0].score == 0.6
        assert results[0].source == RuleSource.REQUEST
        assert results[0].to_dict()["source"] == "request"

    def test_expected_unique_rule(self, engine):
        df = pd.DataFrame({"id": ["A", "B", "C", "A", "D"]})
        rule = RuleDefinition(
            rule_key="test_unique", rule_type=RuleType.EXPECTED_UNIQUE,
            source=RuleSource.REQUEST, domain="Payments", secondary_domain=None,
            parameters={"target_column": "id"},
        )
        results = engine.evaluate(df, [rule])
        assert results[0].fail_count == 1  # One duplicate (second "A")
        assert results[0].score == 0.8

    def test_regex_match_rule(self, engine):
        df = pd.DataFrame({"code": ["USD", "EUR", "INVALID", "GBP", "12"]})
        rule = RuleDefinition(
            rule_key="test_regex", rule_type=RuleType.REGEX_MATCH,
            source=RuleSource.REQUEST, domain="Payments", secondary_domain=None,
            parameters={"target_column": "code", "pattern": r"^[A-Z]{3}$"},
        )
        results = engine.evaluate(df, [rule])
        assert results[0].pass_count == 3  # USD, EUR, GBP
        assert results[0].fail_count == 2

    def test_allowed_values_rule(self, engine):
        df = pd.DataFrame({"status": ["approved", "declined", "unknown", "approved", "pending"]})
        rule = RuleDefinition(
            rule_key="test_allowed", rule_type=RuleType.ALLOWED_VALUES,
            source=RuleSource.REQUEST, domain="Payments", secondary_domain=None,
            parameters={"target_column": "status", "values": ["approved", "declined", "pending"]},
        )
        results = engine.evaluate(df, [rule])
        assert results[0].fail_count == 1  # "unknown"
        assert results[0].pass_count == 4

    def test_numeric_range_rule(self, engine):
        df = pd.DataFrame({"amount": ["100", "200", "-5", "0", "500"]})
        rule = RuleDefinition(
            rule_key="test_range", rule_type=RuleType.NUMERIC_RANGE,
            source=RuleSource.REQUEST, domain="Payments", secondary_domain=None,
            parameters={"target_column": "amount", "min_value": 0, "inclusive_min": False},
        )
        results = engine.evaluate(df, [rule])
        assert results[0].fail_count == 2  # -5 and 0 (exclusive min)
        assert results[0].pass_count == 3

    def test_column_comparison_rule(self, engine):
        df = pd.DataFrame({
            "settlement_date": ["2024-01-05", "2024-01-03", "2024-01-10"],
            "authorization_date": ["2024-01-01", "2024-01-04", "2024-01-02"],
        })
        rule = RuleDefinition(
            rule_key="test_compare", rule_type=RuleType.COLUMN_COMPARISON,
            source=RuleSource.DOMAIN_CONFIGURATION, domain="Payments", secondary_domain=None,
            parameters={
                "left_column": "settlement_date",
                "operator": ">=",
                "right_column": "authorization_date",
            },
        )
        results = engine.evaluate(df, [rule])
        # Row 0: 2024-01-05 >= 2024-01-01 ✓
        # Row 1: 2024-01-03 >= 2024-01-04 ✗
        # Row 2: 2024-01-10 >= 2024-01-02 ✓
        assert results[0].pass_count == 2
        assert results[0].fail_count == 1

    def test_conditional_required_rule(self, engine):
        df = pd.DataFrame({
            "status": ["approved", "declined", "approved", "declined", "approved"],
            "reason": [None, "insufficient", None, "expired", None],
        })
        rule = RuleDefinition(
            rule_key="test_cond", rule_type=RuleType.CONDITIONAL_REQUIRED,
            source=RuleSource.REQUEST, domain="Payments", secondary_domain=None,
            parameters={
                "condition_column": "status",
                "condition_value": "declined",
                "required_column": "reason",
            },
        )
        results = engine.evaluate(df, [rule])
        # 2 declined rows, both have reasons → 100%
        assert results[0].records_checked == 2
        assert results[0].pass_count == 2
        assert results[0].score == 1.0

    def test_role_mapping_resolution(self, engine):
        """Rules can reference columns by semantic role."""
        df = pd.DataFrame({"txn_amount": ["100", "-5", "200"]})
        rule = RuleDefinition(
            rule_key="test_role", rule_type=RuleType.NUMERIC_RANGE,
            source=RuleSource.DOMAIN_CONFIGURATION, domain="Payments", secondary_domain=None,
            parameters={"target_role": "monetary_amount", "min_value": 0, "inclusive_min": False},
        )
        role_map = {"monetary_amount": "txn_amount"}
        results = engine.evaluate(df, [rule], column_role_map=role_map)
        assert results[0].fail_count == 1  # -5
        assert results[0].target_columns == ["txn_amount"]

    def test_inactive_rule_skipped(self, engine):
        df = pd.DataFrame({"x": [1, 2, 3]})
        rule = RuleDefinition(
            rule_key="inactive", rule_type=RuleType.NON_NULL,
            source=RuleSource.REQUEST, domain="Payments", secondary_domain=None,
            parameters={"target_column": "x"}, active=False,
        )
        results = engine.evaluate(df, [rule])
        assert len(results) == 0

    def test_missing_column_returns_error(self, engine):
        df = pd.DataFrame({"x": [1, 2, 3]})
        rule = RuleDefinition(
            rule_key="missing", rule_type=RuleType.NON_NULL,
            source=RuleSource.REQUEST, domain="Payments", secondary_domain=None,
            parameters={"target_column": "nonexistent"},
        )
        results = engine.evaluate(df, [rule])
        assert results[0].error is not None


class TestApprovalService:
    """Test rule suggestion approval/rejection."""

    def test_approve_proposed_rule(self):
        service = ApprovalService()
        result = service.approve("uuid-123", RuleSuggestionStatus.PROPOSED, "Looks good")
        assert result["new_status"] == "approved"
        assert result["comment"] == "Looks good"

    def test_reject_proposed_rule(self):
        service = ApprovalService()
        result = service.reject("uuid-123", RuleSuggestionStatus.PROPOSED, "Not applicable")
        assert result["new_status"] == "rejected"
        assert result["rejection_reason"] == "Not applicable"

    def test_cannot_approve_already_approved(self):
        service = ApprovalService()
        with pytest.raises(InvalidRuleTransitionError):
            service.approve("uuid-123", RuleSuggestionStatus.APPROVED)

    def test_cannot_reject_already_rejected(self):
        service = ApprovalService()
        with pytest.raises(InvalidRuleTransitionError):
            service.reject("uuid-123", RuleSuggestionStatus.REJECTED)

    def test_ai_suggestion_not_automatically_active(self):
        """Suggested rules must have status=proposed, never active."""
        suggestion = SuggestedRule(
            suggestion_id="test-id",
            definition={
                "rule_key": "amount_positive",
                "type": "numeric_range",
                "description": "Amount should be positive",
                "reasoning": "Most transactions are positive",
                "target_columns": ["amount"],
                "target_column": "amount",
                "min_value": 0,
                "inclusive_min": False,
            },
            confidence=0.85,
            engine_compatible=True,
        )
        assert suggestion.status == RuleSuggestionStatus.PROPOSED
        assert suggestion.engine_compatible is True

    def test_unsupported_suggestion_type_flagged_not_engine_compatible(self):
        """A suggestion with a defined-but-unimplemented type (e.g. date_range) is stored but flagged."""
        suggestion = SuggestedRule(
            suggestion_id="test-id-2",
            definition={"rule_key": "date_check", "type": "date_range", "description": "d"},
            confidence=0.5,
            engine_compatible=False,
        )
        assert suggestion.engine_compatible is False
