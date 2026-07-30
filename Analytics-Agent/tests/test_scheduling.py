"""tests/test_scheduling.py — Agent 3 redesign, Phase 1, Stage 5 (plan
"zany-giggling-crayon"): AnalyticsScheduler.

Includes the wide-dataset stress test the plan explicitly calls for
(Testing Strategy section) — proves the budget actually bounds a
pathological case, not just a handful of hand-picked examples.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.kpi_discovery.models import DiscoveredKPI, FINANCIAL_MEASURE
from app.services.planning.models import PlannedAnalysis
from app.services.question_interpreter.models import QuestionIntent
from app.services.scheduling.analytics_scheduler import AnalyticsScheduler
from app.services.scheduling.budget_config import BudgetConfig


def _pa(analysis_type, priority=5, is_kpi_grounded=False):
    return PlannedAnalysis(analysis_type=analysis_type, target_columns=["x"], rationale="test", priority=priority, is_kpi_grounded=is_kpi_grounded)


def test_trims_to_max_parallel_analyses():
    plan = [_pa(f"type_{i}", priority=i) for i in range(20)]
    budget = BudgetConfig(max_parallel_analyses=5, max_ml_analyses=3, max_expensive_operations=2)
    scheduled = AnalyticsScheduler().schedule(plan, budget, question_intent=None)
    assert len(scheduled) == 5


def test_requested_analyses_never_trimmed_even_beyond_budget():
    plan = [_pa(f"type_{i}", priority=i) for i in range(10)]
    intent = QuestionIntent(candidate_analysis_types={"type_0", "type_1", "type_2", "type_3", "type_4", "type_5", "type_6"})
    budget = BudgetConfig(max_parallel_analyses=3, max_ml_analyses=3, max_expensive_operations=2)  # smaller than the requested set
    scheduled = AnalyticsScheduler().schedule(plan, budget, question_intent=intent)
    scheduled_types = {s.planned_analysis.analysis_type for s in scheduled}
    assert intent.candidate_analysis_types <= scheduled_types  # every requested type survived


def test_priority_order_requested_then_kpi_grounded_then_priority_field():
    plan = [
        _pa("low_priority_generic", priority=9),
        _pa("kpi_thing", priority=5, is_kpi_grounded=True),
        _pa("requested_thing", priority=8),
        _pa("high_priority_generic", priority=1),
    ]
    intent = QuestionIntent(candidate_analysis_types={"requested_thing"})
    budget = BudgetConfig(max_parallel_analyses=4, max_ml_analyses=4, max_expensive_operations=4)
    scheduled = AnalyticsScheduler().schedule(plan, budget, question_intent=intent)
    order = [s.planned_analysis.analysis_type for s in scheduled]
    assert order == ["requested_thing", "kpi_thing", "high_priority_generic", "low_priority_generic"]


def test_sequence_numbers_are_1_indexed_and_ordered():
    plan = [_pa(f"type_{i}", priority=i) for i in range(4)]
    budget = BudgetConfig(max_parallel_analyses=10, max_ml_analyses=10, max_expensive_operations=10)
    scheduled = AnalyticsScheduler().schedule(plan, budget, question_intent=None)
    assert [s.sequence for s in scheduled] == [1, 2, 3, 4]


def test_ml_execution_allowed_capped_at_max_ml_analyses():
    plan = [_pa(f"type_{i}", priority=i) for i in range(6)]
    budget = BudgetConfig(max_parallel_analyses=6, max_ml_analyses=2, max_expensive_operations=2)
    scheduled = AnalyticsScheduler().schedule(plan, budget, question_intent=None)
    allowed = [s for s in scheduled if s.ml_execution_allowed]
    blocked = [s for s in scheduled if not s.ml_execution_allowed]
    assert len(allowed) == 2
    assert len(blocked) == 4
    # the top-2-priority survivors are the ones allowed, not an arbitrary subset
    assert {s.planned_analysis.analysis_type for s in allowed} == {"type_0", "type_1"}


def test_ml_execution_allowed_never_exceeds_ml_budget_even_with_more_requested():
    """ml_execution_allowed can only ever downgrade toward cheaper
    execution — being 'requested' guarantees a slot in the plan, not an
    ML slot."""
    plan = [_pa(f"type_{i}", priority=i) for i in range(5)]
    intent = QuestionIntent(candidate_analysis_types={f"type_{i}" for i in range(5)})  # all requested
    budget = BudgetConfig(max_parallel_analyses=5, max_ml_analyses=1, max_expensive_operations=1)
    scheduled = AnalyticsScheduler().schedule(plan, budget, question_intent=intent)
    assert len(scheduled) == 5  # all requested analyses present
    assert sum(1 for s in scheduled if s.ml_execution_allowed) == 1  # but ML budget still respected


def test_empty_plan_schedules_cleanly():
    budget = BudgetConfig()
    assert AnalyticsScheduler().schedule([], budget, question_intent=None) == []


# ── Wide-dataset stress test (explicit requirement from the plan) ──────────

def test_wide_dataset_stress_scheduling_stays_bounded():
    """Simulates the plan's own worked example: a dataset with 40 numeric +
    15 categorical columns, a hierarchy, and dates — which the Planner is
    intentionally unbounded against. Confirms the Scheduler is the actual
    enforcement point, not just a documentation claim."""
    import time

    # Simulate a Planner producing one PlannedAnalysis per (analysis_type,
    # column) combination across a wide schema — deliberately large.
    analysis_types = [
        "trend", "forecast", "correlation", "anomaly_detection", "time_series_analysis",
        "clustering", "segmentation", "comparative_analysis", "ranking",
        "association_rules", "distribution_analysis", "outlier_detection",
        "classification", "regression", "feature_importance", "root_cause",
    ]
    huge_plan = []
    for i, analysis_type in enumerate(analysis_types * 10):  # 160 candidate analyses
        huge_plan.append(_pa(f"{analysis_type}_{i}", priority=i % 9, is_kpi_grounded=(i % 5 == 0)))

    budget = BudgetConfig(max_parallel_analyses=8, max_ml_analyses=3, max_expensive_operations=2)
    started = time.perf_counter()
    scheduled = AnalyticsScheduler().schedule(huge_plan, budget, question_intent=None)
    elapsed = time.perf_counter() - started

    assert len(scheduled) == budget.max_parallel_analyses
    assert sum(1 for s in scheduled if s.ml_execution_allowed) == budget.max_ml_analyses
    assert elapsed < 1.0  # bounded time regardless of how wide the input plan is
