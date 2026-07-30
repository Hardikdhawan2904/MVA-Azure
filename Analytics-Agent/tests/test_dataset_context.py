"""tests/test_dataset_context.py — Agent 3 redesign, Phase 0 (plan
"zany-giggling-crayon"): DatasetContextBuilder and LocalSchemaInferer.

LocalSchemaInferer is tested against two real, structurally different
datasets already in this monorepo — the Insurance reference dataset and
one of Data-Profiling-Agent's small HR fixture CSVs — specifically to
prove it generalizes beyond Insurance, which is the entire point of Stage
0 existing. DatasetContextBuilder's rich (Agent-2-derived) path is tested
against hand-built column_profiles matching Agent 2's actual confirmed
response shape (verified this session by reading Data-Profiling-Agent's
code directly, not guessed).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from app.services.dataset_context.context_builder import DatasetContextBuilder
from app.services.dataset_context.local_schema_inferer import LocalSchemaInferer

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_INSURANCE_CSV = _FIXTURES_DIR / "insurance_variance_data_native.csv"
_HR_CSV = _FIXTURES_DIR / "hr_employee_payroll.csv"

pytestmark_insurance = pytest.mark.skipif(not _INSURANCE_CSV.exists(), reason="Insurance dataset not found")
pytestmark_hr = pytest.mark.skipif(not _HR_CSV.exists(), reason="HR fixture not found")


# ── LocalSchemaInferer — real datasets, no fabricated data ──────────────────

@pytestmark_insurance
def test_local_schema_inferer_classifies_insurance_dataset_sensibly():
    df = pd.read_csv(_INSURANCE_CSV)
    context = LocalSchemaInferer().infer(df)

    assert context.context_source == "local_fallback"
    assert context.row_count == len(df)
    assert context.column_count == len(df.columns)

    reporting_date = context.column("reporting_date")
    assert reporting_date.is_temporal
    assert reporting_date.semantic_role == "temporal_dimension"

    region = context.column("region")
    assert region.semantic_role == "dimension"

    gwp = context.column("gross_written_premium_actual")
    assert gwp.semantic_role == "metric"


@pytestmark_hr
def test_local_schema_inferer_generalizes_to_hr_dataset():
    """The actual point of this stage: it must work on a dataset it has
    never seen, with zero Insurance-specific assumptions."""
    df = pd.read_csv(_HR_CSV)
    context = LocalSchemaInferer().infer(df)

    assert context.context_source == "local_fallback"
    assert context.row_count == len(df)

    employee_id = context.column("employee_id")
    assert employee_id.is_identifier
    assert employee_id.semantic_role == "identifier"

    hire_date = context.column("hire_date")
    assert hire_date.is_temporal
    assert hire_date.semantic_role == "temporal_dimension"

    salary = context.column("salary")
    assert salary.semantic_role == "metric"
    assert salary.semantic_type == "salary_amount"  # name-pattern hint fires

    department = context.column("department")
    assert department.semantic_role == "dimension"


def test_local_schema_inferer_never_fabricates_confidence():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    context = LocalSchemaInferer().infer(df)
    for col in context.columns:
        assert col.confidence is None  # heuristic-only, never claims a score it didn't earn


# ── DatasetContextBuilder ────────────────────────────────────────────────────

def _sample_agent2_column_profile(**overrides):
    base = {
        "physical_name": "gross_written_premium_actual",
        "refined_data_type": "decimal",
        "statistics": {"cardinality_ratio": 0.98, "null_ratio": 0.0, "sample_values": [1000.5, 2000.1]},
        "candidate_semantic_type": "premium_amount",
        "candidate_column_role": "metric",
        "candidate_confidence": 0.8,
        "confirmed_semantic_type": "premium_amount",
        "confirmed_column_role": "metric",
        "schema_confidence": 0.93,
        "identifier_score": 0.0,
        "is_grain_key": False,
    }
    base.update(overrides)
    return base


def test_dataset_context_builder_uses_rich_path_when_column_profiles_present():
    df = pd.DataFrame({"gross_written_premium_actual": [1000.5, 2000.1, 3000.0]})
    context = DatasetContextBuilder().build(
        df,
        column_profiles=[_sample_agent2_column_profile()],
        hierarchy={"status": "accepted", "template_key": "insurance_geographic", "level_columns": ["region"], "average_confidence": 0.87},
        charts=[{"chart_key": "premium_trend"}],
        full_feature_recommendation={"target_column": "underwriting_result_actual"},
        detected_domain="Insurance",
    )
    assert context.context_source == "agent2"
    assert context.detected_domain == "Insurance"
    assert context.hierarchy.status == "accepted"
    assert context.hierarchy.template_key == "insurance_geographic"
    assert context.charts == [{"chart_key": "premium_trend"}]
    assert context.feature_recommendation == {"target_column": "underwriting_result_actual"}

    col = context.column("gross_written_premium_actual")
    assert col.semantic_type == "premium_amount"
    assert col.semantic_role == "metric"
    assert col.confidence == 0.93  # schema_confidence wins over candidate_confidence


def test_dataset_context_builder_prefers_confirmed_over_candidate():
    profile = _sample_agent2_column_profile(
        confirmed_semantic_type="revenue_amount", confirmed_column_role="metric",
        candidate_semantic_type="expense_amount", candidate_column_role="dimension",
    )
    df = pd.DataFrame({"gross_written_premium_actual": [1.0]})
    context = DatasetContextBuilder().build(df, column_profiles=[profile])
    col = context.column("gross_written_premium_actual")
    assert col.semantic_type == "revenue_amount"
    assert col.semantic_role == "metric"


def test_dataset_context_builder_falls_back_to_candidate_when_unconfirmed():
    profile = _sample_agent2_column_profile(
        confirmed_semantic_type=None, confirmed_column_role=None, schema_confidence=None,
    )
    df = pd.DataFrame({"gross_written_premium_actual": [1.0]})
    context = DatasetContextBuilder().build(df, column_profiles=[profile])
    col = context.column("gross_written_premium_actual")
    assert col.semantic_type == "premium_amount"  # candidate value
    assert col.confidence == 0.8  # candidate_confidence, since schema_confidence was None


def test_dataset_context_builder_falls_back_to_local_inference_without_column_profiles():
    df = pd.DataFrame({"salary": [1000, 2000], "department": ["Eng", "HR"]})
    context = DatasetContextBuilder().build(df, column_profiles=None)
    assert context.context_source == "local_fallback"
    assert context.column("salary").semantic_role == "metric"


def test_dataset_context_builder_treats_empty_column_profiles_as_missing():
    df = pd.DataFrame({"salary": [1000, 2000]})
    context = DatasetContextBuilder().build(df, column_profiles=[])
    assert context.context_source == "local_fallback"


def test_dataset_context_builder_detects_temporal_column_agent2_classified_as_identifier():
    """Real bug, found live: a one-row-per-day reporting_date column is
    100% unique, so Agent 2 classifies it candidate/confirmed_column_role
    "identifier" (is_grain_key=True) rather than "temporal_dimension" —
    confirmed against a real /pipeline/run response. Trusting semantic_role
    alone made every temporal-dependent planning rule (trend/forecast/
    correlation/anomaly_detection/time_series) silently produce nothing for
    any dataset shaped this way. refined_data_type ("date"/"datetime") must
    still mark the column temporal even when Agent 2's role says
    identifier — this is the fix, locked in here."""
    profile = _sample_agent2_column_profile(
        physical_name="reporting_date",
        refined_data_type="date",
        candidate_semantic_type="identifier",
        candidate_column_role="identifier",
        confirmed_semantic_type="identifier",
        confirmed_column_role="identifier",
        is_grain_key=True,
    )
    df = pd.DataFrame({"reporting_date": ["2025-01-01", "2025-01-02", "2025-01-03"]})
    context = DatasetContextBuilder().build(df, column_profiles=[profile])
    col = context.column("reporting_date")

    assert col.is_temporal is True
    assert col.is_identifier is True  # both — a grain key can genuinely also be the time axis
    assert col.semantic_role == "identifier"  # Agent 2's role classification itself is untouched


def test_dataset_context_builder_datetime_refined_type_also_counts_as_temporal():
    profile = _sample_agent2_column_profile(
        physical_name="created_at", refined_data_type="datetime",
        candidate_column_role="identifier", confirmed_column_role="identifier",
    )
    df = pd.DataFrame({"created_at": ["2025-01-01T00:00:00"]})
    context = DatasetContextBuilder().build(df, column_profiles=[profile])
    assert context.column("created_at").is_temporal is True


def test_dataset_context_builder_non_date_identifier_stays_non_temporal():
    """The fix must not make every identifier temporal — only ones Agent 2
    actually typed as date/datetime."""
    profile = _sample_agent2_column_profile(
        physical_name="policy_id", refined_data_type="text",
        candidate_column_role="identifier", confirmed_column_role="identifier",
        is_grain_key=True,
    )
    df = pd.DataFrame({"policy_id": ["P001", "P002"]})
    context = DatasetContextBuilder().build(df, column_profiles=[profile])
    col = context.column("policy_id")
    assert col.is_temporal is False
    assert col.is_identifier is True


# ── DatasetContext convenience methods ───────────────────────────────────────

def test_dataset_context_query_helpers():
    df = pd.DataFrame({"salary": [1, 2], "department": ["a", "b"], "hire_date": ["2024-01-01", "2024-02-01"]})
    context = LocalSchemaInferer().infer(df)

    assert context.column("nonexistent") is None
    metrics = context.columns_with_role("metric")
    assert any(c.name == "salary" for c in metrics)

    salary_type_cols = context.columns_with_semantic_type("salary_amount")
    assert [c.name for c in salary_type_cols] == ["salary"]
