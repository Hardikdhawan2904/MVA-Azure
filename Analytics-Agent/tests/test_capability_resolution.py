"""tests/test_capability_resolution.py — Agent 3 redesign, Phase 1, Stage 1
(plan "zany-giggling-crayon"): AnalyticsCapabilityResolver.

Covers the structural/execution split, the exact confidence/threshold
behavior, and — explicitly, per the plan's own risk mitigation — a test
that the resolver's signature makes readiness-scoring duplication
impossible: it can only ever receive the two already-computed scores plus
a DatasetContext, never a raw DataFrame or Agent 2's evidence/strengths/
blocking_issues.
"""

import inspect
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from app.config import ML_READINESS_THRESHOLD
from app.services.capability_resolution.analytics_capability_resolver import AnalyticsCapabilityResolver
from app.services.capability_resolution.models import (
    ALL_CAPABILITIES, ANOMALY_DETECTION, ASSOCIATION_RULES, CLASSIFICATION,
    CLUSTERING, CORRELATION, FORECASTING, REGRESSION, SEGMENTATION, TIME_SERIES,
)
from app.services.dataset_context.local_schema_inferer import LocalSchemaInferer
from app.services.dataset_context.models import ColumnContext, DatasetContext

_INSURANCE_CSV = Path(__file__).parent / "fixtures" / "insurance_variance_data_native.csv"
_HR_CSV = Path(r"C:\Users\dhawa\mva\Data-Profiling-Agent\tests\fixtures\hr_employee_payroll.csv")

pytestmark_insurance = pytest.mark.skipif(not _INSURANCE_CSV.exists(), reason="Insurance dataset not found")
pytestmark_hr = pytest.mark.skipif(not _HR_CSV.exists(), reason="HR fixture not found")


# ── No-duplication signature test ────────────────────────────────────────────

def test_resolver_signature_cannot_receive_raw_dataframe_or_agent2_evidence():
    """Enforces the plan's hard constraint at the signature level: the only
    way this class could recompute Agent 2's readiness scoring is if it had
    access to raw data or Agent 2's own evidence/strengths/blocking_issues.
    It doesn't and structurally can't — confirmed by inspecting the actual
    parameter names, not just trusting the docstring."""
    sig = inspect.signature(AnalyticsCapabilityResolver.resolve)
    params = set(sig.parameters) - {"self"}
    assert params == {"dataset_context", "ml_readiness_score", "llm_readiness_score"}
    # Extra safety: neither parameter name suggests raw-data or evidence access.
    forbidden_substrings = ["dataframe", "df", "evidence", "strengths", "blocking_issues", "quality_result"]
    for param_name in params:
        assert not any(f in param_name.lower() for f in forbidden_substrings), (
            f"Parameter '{param_name}' looks like it could carry raw data or Agent 2's evidence — "
            f"this would let capability resolution silently duplicate readiness scoring."
        )


# ── Structural / execution split — pure unit tests ──────────────────────────

def _bare_context(columns: list[ColumnContext], row_count: int = 1000, feature_recommendation=None) -> DatasetContext:
    return DatasetContext(
        row_count=row_count, column_count=len(columns), columns=columns,
        context_source="local_fallback", feature_recommendation=feature_recommendation,
    )


def test_structural_false_means_execution_is_never_evaluated():
    ctx = _bare_context([ColumnContext(name="x", semantic_role="dimension")])  # no metric, no temporal
    profile = AnalyticsCapabilityResolver().resolve(ctx, ml_readiness_score=99.0, llm_readiness_score=99.0)
    forecasting = profile.capabilities[FORECASTING]
    assert forecasting.structural.supported is False
    assert forecasting.execution is None  # moot — never evaluated


def test_execution_confidence_is_agent2_score_unmodified():
    ctx = _bare_context([
        ColumnContext(name="revenue", semantic_role="metric"),
        ColumnContext(name="cost", semantic_role="metric"),
    ])
    profile = AnalyticsCapabilityResolver().resolve(ctx, ml_readiness_score=83.25, llm_readiness_score=99.0)
    execution = profile.capabilities[CLUSTERING].execution
    assert execution.supported is True
    assert execution.confidence == 0.8325  # exactly score/100, not a new computation


def test_execution_fails_below_threshold_with_reason():
    ctx = _bare_context([
        ColumnContext(name="revenue", semantic_role="metric"),
        ColumnContext(name="cost", semantic_role="metric"),
    ])
    below = ML_READINESS_THRESHOLD - 5
    profile = AnalyticsCapabilityResolver().resolve(ctx, ml_readiness_score=below, llm_readiness_score=99.0)
    execution = profile.capabilities[CORRELATION].execution
    assert execution.supported is False
    assert execution.confidence is None
    assert str(ML_READINESS_THRESHOLD) in execution.reason


def test_classification_and_regression_require_matching_target_and_problem_type():
    ctx_no_target = _bare_context([ColumnContext(name="x", semantic_role="metric")], feature_recommendation=None)
    profile = AnalyticsCapabilityResolver().resolve(ctx_no_target, 99.0, 99.0)
    assert profile.capabilities[CLASSIFICATION].structural.supported is False
    assert profile.capabilities[REGRESSION].structural.supported is False

    ctx_classification = _bare_context(
        [ColumnContext(name="churned", semantic_role="metric")],
        feature_recommendation={"target_column": "churned", "problem_type": "classification"},
    )
    profile2 = AnalyticsCapabilityResolver().resolve(ctx_classification, 99.0, 99.0)
    assert profile2.capabilities[CLASSIFICATION].structural.supported is True
    assert profile2.capabilities[REGRESSION].structural.supported is False  # wrong problem_type


def test_association_rules_needs_two_workable_categoricals():
    ctx_one_dim = _bare_context([ColumnContext(name="region", semantic_role="dimension", cardinality_ratio=0.05)])
    assert AnalyticsCapabilityResolver().resolve(ctx_one_dim, 99.0, 99.0).capabilities[ASSOCIATION_RULES].structural.supported is False

    ctx_two_dims = _bare_context([
        ColumnContext(name="region", semantic_role="dimension", cardinality_ratio=0.05),
        ColumnContext(name="segment", semantic_role="dimension", cardinality_ratio=0.1),
    ])
    assert AnalyticsCapabilityResolver().resolve(ctx_two_dims, 99.0, 99.0).capabilities[ASSOCIATION_RULES].structural.supported is True


def test_association_rules_excludes_near_unique_or_single_valued_columns():
    ctx = _bare_context([
        ColumnContext(name="id", semantic_role="dimension", cardinality_ratio=0.99),   # near-identifier, excluded
        ColumnContext(name="flag", semantic_role="dimension", cardinality_ratio=0.001),  # below min ratio, excluded
        ColumnContext(name="region", semantic_role="dimension", cardinality_ratio=0.05),  # workable
    ])
    result = AnalyticsCapabilityResolver().resolve(ctx, 99.0, 99.0).capabilities[ASSOCIATION_RULES]
    assert result.structural.supported is False  # only 1 workable column


def test_segmentation_accepts_either_a_metric_or_a_workable_categorical():
    ctx_metric_only = _bare_context([ColumnContext(name="salary", semantic_role="metric")])
    assert AnalyticsCapabilityResolver().resolve(ctx_metric_only, 99.0, 99.0).capabilities[SEGMENTATION].structural.supported is True

    ctx_dim_only = _bare_context([ColumnContext(name="department", semantic_role="dimension", cardinality_ratio=0.1)])
    assert AnalyticsCapabilityResolver().resolve(ctx_dim_only, 99.0, 99.0).capabilities[SEGMENTATION].structural.supported is True

    ctx_neither = _bare_context([ColumnContext(name="id", semantic_role="identifier", cardinality_ratio=0.99)])
    assert AnalyticsCapabilityResolver().resolve(ctx_neither, 99.0, 99.0).capabilities[SEGMENTATION].structural.supported is False


def test_time_series_and_forecasting_require_min_rows():
    ctx = DatasetContext(
        row_count=5, column_count=2,
        columns=[
            ColumnContext(name="date", semantic_role="temporal_dimension", is_temporal=True),
            ColumnContext(name="revenue", semantic_role="metric"),
        ],
        context_source="local_fallback",
    )
    result = AnalyticsCapabilityResolver().resolve(ctx, 99.0, 99.0)
    assert result.capabilities[FORECASTING].structural.supported is False
    assert "row" in result.capabilities[FORECASTING].structural.reason.lower()


def test_capability_profile_convenience_methods():
    ctx = _bare_context([
        ColumnContext(name="revenue", semantic_role="metric"),
        ColumnContext(name="cost", semantic_role="metric"),
    ])
    profile = AnalyticsCapabilityResolver().resolve(ctx, ml_readiness_score=90.0, llm_readiness_score=90.0)
    assert profile.is_structurally_possible(CLUSTERING) is True
    assert profile.is_ml_viable(CLUSTERING) is True
    assert profile.confidence_for(CLUSTERING) == 0.9
    assert profile.is_structurally_possible(CLASSIFICATION) is False
    assert profile.is_ml_viable(CLASSIFICATION) is False
    assert profile.confidence_for(CLASSIFICATION) is None


# ── Real, cross-domain datasets — no fabricated data ─────────────────────────

@pytestmark_insurance
def test_resolver_against_real_insurance_dataset_high_readiness():
    df = pd.read_csv(_INSURANCE_CSV)
    ctx = LocalSchemaInferer().infer(df)
    profile = AnalyticsCapabilityResolver().resolve(ctx, ml_readiness_score=92.0, llm_readiness_score=95.0)

    assert profile.is_structurally_possible(FORECASTING) is True
    assert profile.is_ml_viable(FORECASTING) is True
    assert profile.is_structurally_possible(TIME_SERIES) is True
    assert profile.is_structurally_possible(CLUSTERING) is True
    assert profile.is_structurally_possible(CORRELATION) is True
    assert profile.is_structurally_possible(ANOMALY_DETECTION) is True
    # LocalSchemaInferer never populates feature_recommendation -> no target -> structurally impossible
    assert profile.is_structurally_possible(CLASSIFICATION) is False
    assert profile.is_structurally_possible(REGRESSION) is False


@pytestmark_insurance
def test_resolver_against_real_insurance_dataset_low_readiness():
    df = pd.read_csv(_INSURANCE_CSV)
    ctx = LocalSchemaInferer().infer(df)
    profile = AnalyticsCapabilityResolver().resolve(ctx, ml_readiness_score=40.0, llm_readiness_score=40.0)

    forecasting = profile.capabilities[FORECASTING]
    assert forecasting.structural.supported is True   # shape didn't change
    assert forecasting.execution.supported is False    # only readiness gate failed
    assert "40.0%" in forecasting.execution.reason


@pytestmark_hr
def test_resolver_generalizes_to_hr_dataset():
    """The actual point of this stage: works on a domain it's never seen."""
    df = pd.read_csv(_HR_CSV)
    ctx = LocalSchemaInferer().infer(df)
    profile = AnalyticsCapabilityResolver().resolve(ctx, ml_readiness_score=90.0, llm_readiness_score=90.0)

    # hire_date (temporal) + salary (metric) -> forecasting/time_series structurally possible
    assert profile.is_structurally_possible(FORECASTING) is True
    # salary + bonus are both numeric metric columns -> clustering's >= 2 threshold is met
    assert profile.is_structurally_possible(CLUSTERING) is True
    # department is a workable categorical dimension -> segmentation possible via grouping
    assert profile.is_structurally_possible(SEGMENTATION) is True


def test_all_capabilities_covered():
    """Every capability the plan names actually has a structural check —
    catches a capability silently falling through with no rule."""
    ctx = _bare_context([ColumnContext(name="x", semantic_role="metric")])
    profile = AnalyticsCapabilityResolver().resolve(ctx, 99.0, 99.0)
    assert set(profile.capabilities.keys()) == set(ALL_CAPABILITIES)
