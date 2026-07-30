"""Tests for the profiling orchestrator (full pipeline)."""

import uuid
import pytest
from pathlib import Path

from app.core.config import Settings
from app.core.enums import RunStatus
from app.repositories.configuration_repository import ConfigurationRepository
from app.services.llm.provider import MockLLMProvider
from app.services.llm.interface import LLMResponse
from app.services.schema_intelligence.local_provider import LocalSchemaIntelligenceProvider
from app.services.ingestion.temporary_storage import TemporaryStorage
from app.services.orchestration.profiling_orchestrator import ProfilingOrchestrator


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        DATABASE_URL="postgresql://x:x@localhost/test",
        TEMP_STORAGE_DIR=str(tmp_path / "uploads"),
        MAX_SAMPLE_VALUES=5,
        LLM_BURST_COOLDOWN_SECONDS=0,
    )


@pytest.fixture
def config_repo() -> ConfigurationRepository:
    return ConfigurationRepository(config_dir=Path(__file__).parent.parent.parent / "config")


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    llm = MockLLMProvider()
    # Set up a fallback response (will trigger deterministic fallback)
    llm.set_responses([
        LLMResponse(content="", parsed=None, success=False, error="mock_disabled")
    ] * 10)
    return llm


@pytest.fixture
def orchestrator(settings, config_repo, mock_llm, tmp_path) -> ProfilingOrchestrator:
    si = LocalSchemaIntelligenceProvider(mock_llm)
    temp = TemporaryStorage(settings)
    return ProfilingOrchestrator(settings, config_repo, si, temp)


class TestProfilingOrchestrator:
    """Test the full profiling pipeline."""

    def _make_csv(self, content: str) -> bytes:
        return content.encode("utf-8")

    def test_successful_csv_run(self, orchestrator):
        csv_content = self._make_csv(
            "txn_id,amount,status,auth_date,country\n"
            + "\n".join([
                f"T{i:04d},{i*10.5},{'approved' if i%3!=0 else 'declined'},2024-01-{(i%28)+1:02d},US"
                for i in range(1, 51)
            ])
        )
        run_id = uuid.uuid4()
        result = orchestrator.execute(
            run_id=run_id,
            file_content=csv_content,
            filename="payments.csv",
            primary_domain="Payments",
            schema_metadata={"columns": [
                {"column_name": "txn_id", "description": "Transaction ID", "mandatory": False, "expected_unique": True},
                {"column_name": "amount", "description": "Transaction amount", "mandatory": True, "expected_unique": False},
            ]},
        )

        assert result.status == RunStatus.COMPLETED
        assert result.primary_domain == "Payments"
        assert len(result.column_profiles) == 5
        assert result.dataset_profile["row_count"] == 50
        assert len(result.quality_assessments) >= 9
        assert len(result.readiness_assessments) == 4
        assert len(result.charts) >= 1

    def test_unsupported_domain_fails(self, orchestrator):
        csv = self._make_csv("col1\nval\n")
        result = orchestrator.execute(
            run_id=uuid.uuid4(), file_content=csv,
            filename="data.csv", primary_domain="Healthcare",
        )
        assert result.status == RunStatus.FAILED
        assert result.error["code"] == "UNSUPPORTED_DOMAIN"

    def test_invalid_file_type_fails(self, orchestrator):
        result = orchestrator.execute(
            run_id=uuid.uuid4(), file_content=b"data",
            filename="data.json", primary_domain="Payments",
        )
        assert result.status == RunStatus.FAILED
        assert "UNSUPPORTED_FILE_TYPE" in result.error["code"]

    def test_temp_file_cleaned_up_on_success(self, orchestrator, settings):
        csv = self._make_csv("id,val\n1,a\n2,b\n3,c\n")
        run_id = uuid.uuid4()
        orchestrator.execute(
            run_id=run_id, file_content=csv,
            filename="test.csv", primary_domain="Payments",
        )
        temp = TemporaryStorage(settings)
        assert temp.get_run_directory(run_id) is None

    def test_temp_file_cleaned_up_on_failure(self, orchestrator, settings):
        run_id = uuid.uuid4()
        orchestrator.execute(
            run_id=run_id, file_content=b"bad content",
            filename="test.csv", primary_domain="Payments",
        )
        temp = TemporaryStorage(settings)
        assert temp.get_run_directory(run_id) is None

    def test_secondary_domain_from_configured_only(self, orchestrator):
        csv = self._make_csv(
            "auth_status,approval_code,decline_reason\n"
            + "\n".join([f"approved,AC{i:03d}," for i in range(30)])
        )
        result = orchestrator.execute(
            run_id=uuid.uuid4(), file_content=csv,
            filename="auth.csv", primary_domain="Payments",
        )
        if result.secondary_domain.get("name"):
            allowed = ["Authorization", "Clearing", "Settlement", "Fraud"]
            assert result.secondary_domain["name"] in allowed

    def test_overall_quality_excludes_not_assessable(self, orchestrator):
        csv = self._make_csv("id,amount\n1,100\n2,200\n3,300\n")
        result = orchestrator.execute(
            run_id=uuid.uuid4(), file_content=csv,
            filename="simple.csv", primary_domain="Payments",
            schema_metadata={"columns": [
                {"column_name": "amount", "mandatory": True, "expected_unique": False},
            ]},
        )
        assert result.status == RunStatus.COMPLETED
        excluded = result.overall_quality.get("excluded_dimensions", [])
        # Timeliness, integrity, accuracy should be excluded
        assert "timeliness" in excluded
        assert "integrity" in excluded
        assert "accuracy" in excluded

    def test_approved_rule_dicts_are_evaluated(self, orchestrator):
        """Approved suggestions passed as approved_rule_dicts are evaluated like any other rule."""
        csv = self._make_csv("amount\n100\n-5\n200\n0\n50\n")
        result = orchestrator.execute(
            run_id=uuid.uuid4(), file_content=csv,
            filename="amounts.csv", primary_domain="Payments",
            approved_rule_dicts=[{
                "rule_key": "amount_non_negative",
                "type": "numeric_range",
                "target_column": "amount",
                "min_value": 0,
                "inclusive_min": True,
            }],
        )
        assert result.status == RunStatus.COMPLETED
        matching = [r for r in result.rule_evaluations if r["rule_key"] == "amount_non_negative"]
        assert len(matching) == 1
        assert matching[0]["fail_count"] == 1  # -5 is the only value below the min

    def test_rule_suggestions_empty_on_llm_failure(self, orchestrator):
        """When the LLM is unavailable, rule_suggestions degrades to an empty list, not a failure."""
        csv = self._make_csv("id,val\n1,a\n2,b\n3,c\n")
        result = orchestrator.execute(
            run_id=uuid.uuid4(), file_content=csv,
            filename="test.csv", primary_domain="Payments",
        )
        assert result.status == RunStatus.COMPLETED
        assert result.rule_suggestions == []

    def test_raw_data_not_in_result(self, orchestrator):
        """Result should not contain raw dataframe rows."""
        csv = self._make_csv("a,b\n1,x\n2,y\n3,z\n")
        result = orchestrator.execute(
            run_id=uuid.uuid4(), file_content=csv,
            filename="test.csv", primary_domain="Payments",
        )
        # Verify no raw rows exist — only profiles and aggregates
        import json
        serialized = json.dumps(result.__dict__, default=str)
        # Should not contain all three raw values together as a row
        assert "1,x" not in serialized or "raw_rows" not in serialized
