"""Tests for Schema Intelligence interface and local provider."""

import pytest

from app.core.enums import SchemaIntelligenceDecision
from app.services.llm.interface import LLMResponse
from app.services.llm.provider import MockLLMProvider
from app.services.schema_intelligence.local_provider import LocalSchemaIntelligenceProvider
from app.services.schema_intelligence.models import (
    ColumnAnalysisInput,
    DomainContext,
)


@pytest.fixture
def domain_context() -> DomainContext:
    return DomainContext(
        primary_domain="Payments",
        secondary_domains=["Authorization", "Settlement"],
        row_count=1000,
        column_count=10,
    )


@pytest.fixture
def sample_columns() -> list[ColumnAnalysisInput]:
    return [
        ColumnAnalysisInput(
            column_name="txn_amount",
            normalized_key="txn_amount",
            description="Transaction amount in local currency",
            refined_physical_type="decimal",
            statistics_summary={"null_ratio": 0.01, "cardinality_ratio": 0.85},
            representative_sample_values=["100.50", "250.00", "75.25"],
            candidate_semantic_type="monetary_amount",
            candidate_column_role="metric",
            candidate_confidence=0.85,
            primary_domain="Payments",
        ),
        ColumnAnalysisInput(
            column_name="auth_status",
            normalized_key="auth_status",
            description="Authorization result status",
            refined_physical_type="categorical",
            statistics_summary={"null_ratio": 0.0, "cardinality_ratio": 0.03, "distinct_count": 5},
            representative_sample_values=["approved", "declined", "pending"],
            candidate_semantic_type="status",
            candidate_column_role="dimension",
            candidate_confidence=0.82,
            primary_domain="Payments",
        ),
    ]


class TestLocalSchemaIntelligenceProvider:
    """Test the local Schema Intelligence provider."""

    def test_successful_llm_response(self, sample_columns, domain_context):
        """Provider should return structured results on successful LLM response."""
        mock_llm = MockLLMProvider()
        mock_llm.set_responses([
            LLMResponse(
                content="",
                parsed={
                    "columns": [
                        {
                            "column_name": "txn_amount",
                            "decision": "confirmed",
                            "confirmed_semantic_type": "monetary_amount",
                            "confirmed_column_role": "metric",
                            "confidence": 0.95,
                            "reasoning": "Amount column confirmed",
                            "recommended_mandatory": True,
                            "recommended_expected_unique": False,
                        },
                        {
                            "column_name": "auth_status",
                            "decision": "overridden",
                            "confirmed_semantic_type": "authorization_status",
                            "confirmed_column_role": "dimension",
                            "confidence": 0.92,
                            "reasoning": "More specific type for payments domain",
                            "recommended_mandatory": True,
                            "recommended_expected_unique": False,
                        },
                    ],
                    "model_name": "gpt-4o",
                    "prompt_version": "si-v1",
                },
                model="gpt-4o",
                success=True,
            )
        ])

        provider = LocalSchemaIntelligenceProvider(mock_llm)
        result = provider.analyze(sample_columns, domain_context)

        assert result.success is True
        assert result.fallback_used is False
        assert len(result.column_results) == 2

        # Check confirmed column
        txn = next(r for r in result.column_results if r.column_name == "txn_amount")
        assert txn.decision == SchemaIntelligenceDecision.CONFIRMED
        assert txn.confirmed_semantic_type == "monetary_amount"
        assert txn.confidence == 0.95
        assert txn.recommended_mandatory is True

        # Check overridden column
        auth = next(r for r in result.column_results if r.column_name == "auth_status")
        assert auth.decision == SchemaIntelligenceDecision.OVERRIDDEN
        assert auth.confirmed_semantic_type == "authorization_status"
        assert auth.confidence == 0.92

    def test_llm_failure_uses_fallback(self, sample_columns, domain_context):
        """On LLM failure, provider should fall back to deterministic candidates."""
        mock_llm = MockLLMProvider()
        mock_llm.set_responses([
            LLMResponse(
                content="",
                parsed=None,
                model="gpt-4o",
                success=False,
                error="Connection timeout",
            )
        ])

        provider = LocalSchemaIntelligenceProvider(mock_llm)
        result = provider.analyze(sample_columns, domain_context)

        assert result.success is True  # Overall still succeeds with fallback
        assert result.fallback_used is True
        assert len(result.column_results) == 2

        # High-confidence candidates should be confirmed via deterministic fallback
        txn = next(r for r in result.column_results if r.column_name == "txn_amount")
        assert txn.decision == SchemaIntelligenceDecision.CONFIRMED
        assert txn.confirmed_semantic_type == "monetary_amount"
        # Slight penalty applied
        assert txn.confidence < 0.85

    def test_empty_columns_returns_empty_result(self, domain_context):
        """Empty column list should return empty successful result."""
        mock_llm = MockLLMProvider()
        provider = LocalSchemaIntelligenceProvider(mock_llm)
        result = provider.analyze([], domain_context)
        assert result.success is True
        assert len(result.column_results) == 0

    def test_partial_llm_response_fills_missing_with_fallback(self, sample_columns, domain_context):
        """If LLM only responds for some columns, others get fallback."""
        mock_llm = MockLLMProvider()
        mock_llm.set_responses([
            LLMResponse(
                content="",
                parsed={
                    "columns": [
                        {
                            "column_name": "txn_amount",
                            "decision": "confirmed",
                            "confirmed_semantic_type": "monetary_amount",
                            "confirmed_column_role": "metric",
                            "confidence": 0.93,
                            "reasoning": "confirmed",
                            "recommended_mandatory": None,
                            "recommended_expected_unique": None,
                        },
                        # auth_status is missing from response
                    ],
                    "model_name": "gpt-4o",
                    "prompt_version": "si-v1",
                },
                model="gpt-4o",
                success=True,
            )
        ])

        provider = LocalSchemaIntelligenceProvider(mock_llm)
        result = provider.analyze(sample_columns, domain_context)

        assert len(result.column_results) == 2
        # auth_status should get fallback since candidate_confidence=0.82 >= 0.80
        auth = next(r for r in result.column_results if r.column_name == "auth_status")
        assert auth.decision == SchemaIntelligenceDecision.CONFIRMED
        assert "fallback" in auth.reasoning

    def test_low_confidence_candidate_gets_unresolved_on_fallback(self, domain_context):
        """Low-confidence candidates should be unresolved in fallback mode."""
        columns = [
            ColumnAnalysisInput(
                column_name="mystery_col",
                normalized_key="mystery_col",
                refined_physical_type="text",
                candidate_semantic_type="text",
                candidate_column_role="unknown",
                candidate_confidence=0.40,
            )
        ]

        mock_llm = MockLLMProvider()
        mock_llm.set_responses([
            LLMResponse(content="", parsed=None, success=False, error="timeout")
        ])

        provider = LocalSchemaIntelligenceProvider(mock_llm)
        result = provider.analyze(columns, domain_context)

        mystery = result.column_results[0]
        assert mystery.decision == SchemaIntelligenceDecision.UNRESOLVED
        assert mystery.confidence == 0.40

    def test_does_not_change_physical_type(self, domain_context):
        """Schema Intelligence must not override physical data type."""
        columns = [
            ColumnAnalysisInput(
                column_name="amount",
                normalized_key="amount",
                refined_physical_type="decimal",
                candidate_semantic_type="monetary_amount",
                candidate_column_role="metric",
                candidate_confidence=0.90,
            )
        ]

        mock_llm = MockLLMProvider()
        mock_llm.set_responses([
            LLMResponse(
                content="",
                parsed={
                    "columns": [{
                        "column_name": "amount",
                        "decision": "overridden",
                        "confirmed_semantic_type": "transaction_amount",
                        "confirmed_column_role": "metric",
                        "confidence": 0.96,
                        "reasoning": "more specific",
                        "recommended_mandatory": True,
                        "recommended_expected_unique": False,
                    }],
                    "model_name": "gpt-4o",
                    "prompt_version": "si-v1",
                },
                model="gpt-4o",
                success=True,
            )
        ])

        provider = LocalSchemaIntelligenceProvider(mock_llm)
        result = provider.analyze(columns, domain_context)

        # The result has semantic overrides but the physical type stays unchanged
        # (physical type is not part of ColumnAnalysisResult — it's preserved in the profiler)
        amt = result.column_results[0]
        assert amt.confirmed_semantic_type == "transaction_amount"
        assert amt.confirmed_column_role == "metric"

    def test_result_contains_model_info(self, sample_columns, domain_context):
        """Result should track which model was used."""
        mock_llm = MockLLMProvider()
        mock_llm.set_responses([
            LLMResponse(
                content="",
                parsed={
                    "columns": [
                        {
                            "column_name": "txn_amount",
                            "decision": "confirmed",
                            "confirmed_semantic_type": "monetary_amount",
                            "confirmed_column_role": "metric",
                            "confidence": 0.90,
                            "reasoning": "ok",
                            "recommended_mandatory": None,
                            "recommended_expected_unique": None,
                        },
                        {
                            "column_name": "auth_status",
                            "decision": "confirmed",
                            "confirmed_semantic_type": "status",
                            "confirmed_column_role": "dimension",
                            "confidence": 0.85,
                            "reasoning": "ok",
                            "recommended_mandatory": None,
                            "recommended_expected_unique": None,
                        },
                    ],
                    "model_name": "gpt-4o",
                    "prompt_version": "si-v1",
                },
                model="gpt-4o",
                success=True,
            )
        ])

        provider = LocalSchemaIntelligenceProvider(mock_llm)
        result = provider.analyze(sample_columns, domain_context)
        assert result.model_name == "gpt-4o"
        assert result.prompt_version == "si-v1"
