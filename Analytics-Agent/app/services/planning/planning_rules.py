"""app/services/planning/planning_rules.py — Stage 4 of the Agent 3
redesign (plan "zany-giggling-crayon"): PlanningRule implementations.

Every rule matches on `ColumnContext.semantic_role`/`semantic_type`
combinations (Agent 2's classification, or LocalSchemaInferer's fallback)
— never a column name, never a domain check. Each rule only ever checks
`capability_profile.is_analysis_type_possible(...)` (structural, never
execution — that's the Model Selector's job in Stage 6) before proposing
an analysis type, so a structurally-impossible analysis is never even
added to the plan in the first place. Chain of Responsibility, same shape
as `kpi_discovery.kpi_discovery_rules` — every rule runs, contributions
concatenated, deduplicated by `analysis_type` afterward (first occurrence
wins, rules ordered most-specific-first).
"""

from __future__ import annotations

import re
from typing import Callable

from app.services.capability_resolution.models import CapabilityProfile
from app.services.dataset_context.models import (
    DatasetContext, SEMANTIC_ROLE_DIMENSION, SEMANTIC_ROLE_IDENTIFIER, SEMANTIC_ROLE_METRIC,
)
from app.services.kpi_discovery.models import DiscoveredKPI
from app.services.planning.models import PlannedAnalysis
from app.services.question_interpreter.models import QuestionIntent

_CARDINALITY_RATIO_MIN = 0.01
_CARDINALITY_RATIO_MAX = 0.5

# Narrow, deliberate exception to this file's "never a column name" rule --
# same class of exception root_cause_analyzer.py's _HEURISTIC_DRIVER_NAME_RE
# already accepts: domain-neutral calendar vocabulary, not a business term.
# A month/quarter/weekday-of-year column has no year component, so grouping
# by it on a dataset spanning more than one year of that period silently
# sums different years together -- confirmed live where "compare by month"
# on a 2-year banking dataset reported a "highest in March" headline that
# was actually March 2023 + March 2024 summed, not a real single period.
_PERIOD_FRAGMENT_NAME_RE = re.compile(
    r"\b(month|quarter|weekday|day[_ ]of[_ ]week|day[_ ]name)\b", re.IGNORECASE,
)
_PERIOD_FRAGMENT_MAX_DISTINCT = 12  # month=12, quarter=4, weekday=7 -- generous upper bound


def _is_period_fragment_dimension(c, row_count: int) -> bool:
    if not _PERIOD_FRAGMENT_NAME_RE.search(c.name):
        return False
    if c.cardinality_ratio is None or row_count <= 0:
        return False
    approx_distinct = c.cardinality_ratio * row_count
    return approx_distinct <= _PERIOD_FRAGMENT_MAX_DISTINCT


def _workable_dimensions(ctx: DatasetContext):
    candidates = [
        c for c in ctx.columns_with_role(SEMANTIC_ROLE_DIMENSION)
        if c.cardinality_ratio is not None and _CARDINALITY_RATIO_MIN <= c.cardinality_ratio <= _CARDINALITY_RATIO_MAX
    ]
    # Period-fragment dimensions are still structurally valid -- pushed to
    # the end (not excluded), so they remain available when a question
    # names one explicitly (preferred_dim) or nothing else qualifies.
    # Stable sort preserves file order within each group.
    return sorted(candidates, key=lambda c: _is_period_fragment_dimension(c, ctx.row_count))


def _propose(capability_profile: CapabilityProfile, analysis_type: str, target_columns, rationale, priority, is_kpi_grounded=False):
    if not capability_profile.is_analysis_type_possible(analysis_type):
        return None
    return PlannedAnalysis(
        analysis_type=analysis_type, target_columns=list(target_columns),
        rationale=rationale, priority=priority, is_kpi_grounded=is_kpi_grounded,
    )


# ── Rules, most-specific first (dedup keeps the first occurrence) ──────────

_USEFULNESS_ORDER = {"high": 0, "medium": 1, "low": 2}


def _recommended_feature_columns(ctx: DatasetContext, rec: dict, target: str) -> list[str]:
    """Pulls feature_recommendation's LLM-rated feature list, best-usefulness
    -first, filtered to columns that structurally exist in this
    DatasetContext (a stale/renamed name in the recommendation must not
    crash the analyzer). Consumed by rule_target_column_driven so
    ClassificationAnalyzer/RegressionAnalyzer/FeatureImportanceAnalyzer
    (which all read target_columns[1:] as their feature set) actually
    receive real features instead of just the bare target column."""
    entries = rec.get("feature_columns") or []
    named = [
        (e.get("column"), _USEFULNESS_ORDER.get((e.get("usefulness") or "").lower(), 1))
        for e in entries
        if e.get("column") and e.get("column") != target and ctx.column(e.get("column")) is not None
    ]
    named.sort(key=lambda pair: pair[1])
    return [name for name, _ in named]


def rule_target_column_driven(
    ctx: DatasetContext, cp: CapabilityProfile, kpis: list[DiscoveredKPI], question_intent: QuestionIntent | None = None,
) -> list[PlannedAnalysis]:
    rec = ctx.feature_recommendation or {}
    target = rec.get("target_column")
    problem_type = rec.get("problem_type")
    if not target or problem_type not in ("classification", "regression"):
        return []
    feature_cols = _recommended_feature_columns(ctx, rec, target)
    target_columns = [target] + feature_cols
    out = []
    p = _propose(cp, problem_type, target_columns, f"Target column '{target}' identified with problem_type='{problem_type}'", priority=1)
    if p:
        out.append(p)
    fi = _propose(cp, "feature_importance", target_columns, f"Explain what drives target column '{target}'", priority=1)
    if fi:
        out.append(fi)
    return out


def rule_kpi_grounded_root_cause(
    ctx: DatasetContext, cp: CapabilityProfile, kpis: list[DiscoveredKPI], question_intent: QuestionIntent | None = None,
) -> list[PlannedAnalysis]:
    """root_cause has no capability gate (always structurally available,
    deterministic-primary by design — see Stage 7) — proposed whenever a
    discovered/curated KPI exists to explain, OR the question itself names
    a real metric column (Phase 4.6). Whether this survives into the final
    plan for a *specific* question is Stage 3's job (question
    intersection), not decided here.

    Phase 4.6: when the question names a column directly
    (question_intent.preferred_metrics, populated by
    BusinessQuestionInterpreter from a phrase match against the dataset's
    own column names), that wins over discovered_kpis[0] — a KPI-discovery
    rule race is not a reliable signal of what the question actually asked
    about (e.g. "Success Rate" being discovered before "Variance vs
    Budget" purely by ALL_RULES order says nothing about relevance to a
    "why did net profit decline" question). is_kpi_grounded=False here:
    this is question-grounded, not KPI-grounded, and the Scheduler already
    never trims an explicitly-requested analysis regardless of that flag.

    Otherwise, target_columns is deliberately just [kpi.source_columns[0]]
    — the single primary metric, never the rest of source_columns. A
    discovered KPI's other source columns (e.g. Variance vs Budget's
    reference/budget column, Profit Margin's expense column) are *inputs
    to computing the KPI*, not a pre-labeled driver decomposition —
    RootCauseAnalyzer would otherwise treat that lone column as "the
    driver" and trivially explain "100% of the variance" with it, a real
    correctness bug caught via live testing against a Finance dataset with
    no domain plugin. A genuine driver decomposition only ever comes from
    a domain plugin's get_driver_columns() (InsurancePlugin.enhance_plan()
    replaces this rule's proposal entirely when one applies); with no
    plugin, correlation-based mode (RootCauseAnalyzer's generic default)
    is the honest answer.
    """
    if question_intent and question_intent.preferred_metrics:
        metric_col = question_intent.preferred_metrics[0]
        p = _propose(
            cp, "root_cause", [metric_col],
            f"Question names metric column '{metric_col}' directly", priority=1, is_kpi_grounded=False,
        )
        return [p] if p else []
    if not kpis:
        return []
    kpi = kpis[0]
    p = _propose(
        cp, "root_cause", kpi.source_columns[:1],
        f"Root-cause decomposition available for discovered KPI '{kpi.name}'", priority=2, is_kpi_grounded=True,
    )
    return [p] if p else []


def rule_temporal_metric(
    ctx: DatasetContext, cp: CapabilityProfile, kpis: list[DiscoveredKPI], question_intent: QuestionIntent | None = None,
) -> list[PlannedAnalysis]:
    temporal = [c for c in ctx.columns if c.is_temporal]
    metrics = ctx.columns_with_role(SEMANTIC_ROLE_METRIC)
    if not temporal or not metrics:
        return []
    # Phase 4.6 (same fix as rule_dimension_metric): prefer whichever
    # metric the question actually named, if it's among this rule's own
    # structurally-valid metric candidates — otherwise "first metric-role
    # column in file order" silently answers a different question than
    # the one asked (found live: "forecast revenue" defaulted to
    # forecasting units_sold because it happened to appear first).
    metric_names = {m.name for m in metrics}
    preferred_metric = next(
        (m for m in (question_intent.preferred_metrics if question_intent else []) if m in metric_names), None,
    )
    metric_name = preferred_metric or metrics[0].name
    # [temporal, metric] — Stage 7's ForecastAnalyzer/TrendAnalyzer/
    # TimeSeriesAnalyzer all read target_columns[0] as the temporal column
    # and target_columns[1] as the metric to analyze (matching
    # InsurancePlugin.enhance_plan()'s same convention). correlation and
    # anomaly_detection are different: CorrelationAnalyzer/AnomalyAnalyzer
    # consume the WHOLE target_columns list as their feature set (not just
    # positions 0/1) — confirmed by reading both analyzers — so extending
    # this shared `cols` list for forecast's benefit would silently inject
    # unwanted columns into those two. Only "forecast" gets an extended list.
    cols = [temporal[0].name, metric_name]
    rationale = f"Temporal column '{temporal[0].name}' + metric column '{metric_name}' present"
    out = []
    for analysis_type, priority in [("trend", 2), ("correlation", 3), ("anomaly_detection", 3), ("time_series_analysis", 2)]:
        p = _propose(cp, analysis_type, cols, rationale, priority)
        if p:
            out.append(p)

    # forecast's XGBoost Regressor path reads target_columns[2:] as extra
    # feature columns (ForecastAnalyzer's own documented convention) — only
    # populate them from feature_recommendation.feature_columns when that
    # recommendation's target_column IS this same metric (the LLM's feature
    # picks are for predicting a specific target; using them for a
    # different metric would be a category error, not a real improvement).
    # Deterministic strategies and Prophet both ignore anything past index
    # 1, so this is additive-only for the one path that reads it.
    rec = ctx.feature_recommendation or {}
    extra_features: list[str] = []
    if rec.get("target_column") == metric_name:
        extra_features = [c for c in _recommended_feature_columns(ctx, rec, metric_name) if c != temporal[0].name]
    forecast_p = _propose(cp, "forecast", cols + extra_features, rationale, priority=2)
    if forecast_p:
        out.append(forecast_p)
    return out


def rule_dimension_metric(
    ctx: DatasetContext, cp: CapabilityProfile, kpis: list[DiscoveredKPI], question_intent: QuestionIntent | None = None,
) -> list[PlannedAnalysis]:
    dims = _workable_dimensions(ctx)
    metrics = ctx.columns_with_role(SEMANTIC_ROLE_METRIC)
    if not dims or not metrics:
        return []
    # Phase 4.6: prefer whichever metric/dimension the question actually
    # named, if it's among this rule's own structurally-valid candidates
    # (a mentioned dimension might not be "workable" — cardinality-
    # filtered out — in which case falling back to dims[0] is correct).
    metric_names = {m.name for m in metrics}
    dim_names = {d.name for d in dims}
    preferred_metric = next(
        (m for m in (question_intent.preferred_metrics if question_intent else []) if m in metric_names), None,
    )
    preferred_dim = next(
        (d for d in (question_intent.preferred_dimensions if question_intent else []) if d in dim_names), None,
    )
    metric_name = preferred_metric or metrics[0].name
    dim_name = preferred_dim or dims[0].name
    cols = [metric_name, dim_name]
    rationale = f"Categorical dimension '{dim_name}' + metric column '{metric_name}' present"
    out = []
    for analysis_type, priority in [("comparative_analysis", 4), ("ranking", 4), ("distribution_analysis", 4)]:
        p = _propose(cp, analysis_type, cols, rationale, priority)
        if p:
            out.append(p)
    # "clustering" deliberately excluded from the [metric, dim] pair above
    # -- caught via live testing: clustering only registers cluster-on-
    # features algorithms (K-Means) that treat the ENTIRE target_columns
    # list as a purely-numeric feature set. Proposing [metric_name,
    # dim_name] got K-Means selected with the categorical dimension in its
    # feature set, and KMeans.fit_predict() raised "Input X contains NaN"
    # (the dimension coerced to NaN via pd.to_numeric). rule_numeric_pair_
    # no_target already proposes "clustering" correctly (two genuine
    # metric columns) when the dataset supports it -- this rule doesn't
    # need to duplicate that with the wrong column shape.
    #
    # "segmentation" is proposed separately below with target_columns=
    # [metric_name] ONLY (no dim_name) -- it hit the exact same
    # cluster-on-features crash as clustering when given the [metric, dim]
    # pair (segmentation registers the same K-Means/DBSCAN/Hierarchical
    # Clustering algorithms), but unlike clustering, no other rule
    # proposes segmentation generically, and the single-metric convention
    # is what its bin-a-single-metric deterministic strategies (Quantile/
    # Equal-Width/Equal-Frequency Binning, and the Insurance domain
    # plugin's preferred Combined Ratio Buckets strategy) actually expect
    # -- confirmed live: dropping dim_name here is what makes the "Segment
    # portfolio by risk profile" Insurance regression test pass again.
    seg = _propose(cp, "segmentation", [metric_name], f"Metric column '{metric_name}' available to bin into segments", priority=4)
    if seg:
        out.append(seg)
    return out


def rule_numeric_pair_no_target(
    ctx: DatasetContext, cp: CapabilityProfile, kpis: list[DiscoveredKPI], question_intent: QuestionIntent | None = None,
) -> list[PlannedAnalysis]:
    rec = ctx.feature_recommendation or {}
    if rec.get("target_column"):
        return []  # rule_target_column_driven already covers this case
    metrics = ctx.columns_with_role(SEMANTIC_ROLE_METRIC)
    if len(metrics) < 2:
        return []
    cols = [c.name for c in metrics[:2]]
    rationale = f"{len(metrics)} numeric metric columns, no identified target"
    out = []
    for analysis_type, priority in [("correlation", 6), ("clustering", 6), ("outlier_detection", 6)]:
        p = _propose(cp, analysis_type, cols, rationale, priority)
        if p:
            out.append(p)
    return out


def rule_two_categoricals(
    ctx: DatasetContext, cp: CapabilityProfile, kpis: list[DiscoveredKPI], question_intent: QuestionIntent | None = None,
) -> list[PlannedAnalysis]:
    dims = _workable_dimensions(ctx)
    if len(dims) < 2:
        return []
    cols = [c.name for c in dims[:2]]
    rationale = f"{len(dims)} workable categorical columns"
    out = []
    # "distribution_analysis" deliberately excluded here -- caught via live
    # testing: DistributionAnalyzer's own documented convention is
    # [metric_col] or [metric_col, dimension_col] -- target_columns[0] MUST
    # be numeric (DescriptiveDistributionStrategy does
    # pd.to_numeric(df[metric_col], errors="coerce").dropna()). Proposing
    # it here with [dim1, dim2] (two categorical columns, no metric at
    # all) turned the metric column entirely to NaN and produced
    # "Insufficient data for distribution analysis: 0 rows" every time.
    # rule_dimension_metric already proposes it correctly with a real
    # [metric, dimension] pair.
    p = _propose(cp, "association_rules", cols, rationale, priority=5)
    if p:
        out.append(p)
    return out


def rule_identifier_monetary_transaction_shaped(
    ctx: DatasetContext, cp: CapabilityProfile, kpis: list[DiscoveredKPI], question_intent: QuestionIntent | None = None,
) -> list[PlannedAnalysis]:
    identifiers = ctx.columns_with_role(SEMANTIC_ROLE_IDENTIFIER)
    metrics = ctx.columns_with_role(SEMANTIC_ROLE_METRIC)
    if not identifiers or not metrics:
        return []
    cols = [identifiers[0].name, metrics[0].name]
    rationale = f"Identifier '{identifiers[0].name}' + monetary column '{metrics[0].name}' — transaction-shaped"
    out = []
    for analysis_type, priority in [("anomaly_detection", 3), ("segmentation", 4), ("ranking", 4)]:
        p = _propose(cp, analysis_type, cols, rationale, priority)
        if p:
            out.append(p)
    return out


def rule_hierarchy_detected(
    ctx: DatasetContext, cp: CapabilityProfile, kpis: list[DiscoveredKPI], question_intent: QuestionIntent | None = None,
) -> list[PlannedAnalysis]:
    if ctx.hierarchy is None or not ctx.hierarchy.level_columns:
        return []
    cols = list(ctx.hierarchy.level_columns)
    rationale = f"Hierarchy detected: {' > '.join(cols)}"
    out = []
    for analysis_type, priority in [("comparative_analysis", 5), ("ranking", 5)]:
        p = _propose(cp, analysis_type, cols, rationale, priority)
        if p:
            out.append(p)
    return out


ALL_RULES: list[RuleFn] = [
    rule_target_column_driven,
    rule_kpi_grounded_root_cause,
    rule_temporal_metric,
    rule_dimension_metric,
    rule_identifier_monetary_transaction_shaped,
    rule_hierarchy_detected,
    rule_two_categoricals,
    rule_numeric_pair_no_target,
]
