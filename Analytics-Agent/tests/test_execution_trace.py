"""tests/test_execution_trace.py — Tests for the execution trace/summary
(app/agents/analytics_agent/graph.py::_build_execution_trace).

Most cases are tested by feeding _build_execution_trace crafted final_state
dicts directly — it's a pure function over plain data, so this is faster
and more deterministic than driving full graph runs (and, unlike a live
Azure OpenAI call, doesn't depend on whether the API happens to be rate-limited
when the suite runs). A handful of real end-to-end cases against the live
dataset confirm the wiring itself (state.py fields actually get threaded
through by narrate()/record_memory()), matching this codebase's practice
of testing against real dependencies rather than mocking everything.
"""

import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.agents.analytics_agent.graph import (
    _build_execution_trace, _build_multi_analysis_trace, run_analytics_graph, _summarize_breakdown,
    _model_version_for_engine, _load_model_registry,
)
from app.config import ML_READINESS_THRESHOLD, LLM_READINESS_THRESHOLD
from app.services.evidence.evidence_builder import Evidence

_DATASET = Path(__file__).parent / "fixtures" / "insurance_variance_data_native.csv"
_REGISTRY = _load_model_registry()  # the real ml/model_registry.json — tests below assert against
                                     # its actual current values rather than hardcoding stale numbers


# ── _build_execution_trace — pure-function unit tests ────────────────────────

def test_ml_gated_happy_path_reports_model_and_passed_gate():
    state = {
        "intent": "forecast",
        "kpi_name": "underwriting_result",
        "evidence": {"kpi": "Underwriting Result"},  # no ml_readiness_blocked key -> model path ran
        "response": "...",
        "ml_readiness_score": 92.0,
        "llm_engine_used": "Azure OpenAI",
        "llm_readiness_score": 95.0,
        "tools_used": ["RuleEngine", "SQLTool", "MLTool→Prophet/LightGBM", "ExplanationTool"],
    }
    trace, summary = _build_execution_trace(state, elapsed_seconds=1.23)

    ml_step = next(s for s in trace if s["step"] == "forecast")
    assert ml_step["engine"] == "Prophet"
    assert ml_step["gate"]["name"] == "ml_readiness"
    assert ml_step["gate"]["score"] == 92.0
    assert ml_step["gate"]["threshold"] == ML_READINESS_THRESHOLD
    assert ml_step["gate"]["passed"] is True
    assert ml_step["gate"]["breakdown"] is None  # no breakdown supplied in this state

    assert summary["ml_engine"] == "Prophet"
    assert summary["fallback_used"] is False


def test_ml_gated_fallback_path_reports_fallback_engine_and_failed_gate():
    state = {
        "intent": "anomaly_detection",
        "evidence": {
            "ml_readiness_blocked": True,
            "ml_readiness_score": 40.0,
            "fallback_reason": "ML readiness score (40.00%) below threshold (75.0%).",
            "fallback_applied": "Deterministic Z-Score ratio anomaly detection (analytics_tool.rank)",
        },
        "response": "...",
        "ml_readiness_score": 40.0,
        "llm_engine_used": "Template Formatter",
        "llm_readiness_score": 30.0,
    }
    trace, summary = _build_execution_trace(state, elapsed_seconds=0.5)

    ml_step = next(s for s in trace if s["step"] == "anomaly_detection")
    assert ml_step["engine"] == "Deterministic Z-Score ratio anomaly detection (analytics_tool.rank)"
    assert ml_step["gate"]["passed"] is False
    assert ml_step["reason"] == state["evidence"]["fallback_reason"]

    assert summary["fallback_used"] is True


def test_non_ml_gated_intent_has_no_gate():
    state = {
        "intent": "show_kpi",
        "evidence": {"kpi": "Gross Written Premium", "actual": 100.0},
        "response": "...",
        "llm_engine_used": "Azure OpenAI",
        "llm_readiness_score": 99.0,
    }
    trace, summary = _build_execution_trace(state, elapsed_seconds=0.2)

    step = next(s for s in trace if s["step"] == "show_kpi")
    assert step["engine"] == "AnalyticsTool"
    assert step["gate"] is None
    assert summary["ml_engine"] is None


def test_root_cause_notes_xgboost_corroboration_when_present():
    state = {
        "intent": "root_cause",
        "evidence": {"kpi": "x", "ml_predicted_driver": "claim_frequency_variance", "ml_confidence": 0.8},
        "response": "...",
        "llm_engine_used": "Azure OpenAI",
        "llm_readiness_score": 99.0,
    }
    trace, _ = _build_execution_trace(state, elapsed_seconds=0.2)
    step = next(s for s in trace if s["step"] == "root_cause")
    assert step["engine"] == "RootCauseTool"
    assert "XGBoost" in step["reason"]


def test_root_cause_without_ml_evidence_does_not_mention_xgboost():
    state = {
        "intent": "root_cause",
        "evidence": {"kpi": "x"},
        "response": "...",
        "llm_engine_used": "Azure OpenAI",
        "llm_readiness_score": 99.0,
    }
    trace, _ = _build_execution_trace(state, elapsed_seconds=0.2)
    step = next(s for s in trace if s["step"] == "root_cause")
    assert "XGBoost" not in step["reason"]


def test_narration_azure_openai_error_is_distinct_from_never_attempted():
    """The core nuance this feature exists to capture: a passed llm_readiness
    gate whose LLM call(s) failed must not be reported the same way as
    a gate that never passed in the first place."""
    state = {
        "intent": "show_kpi",
        "evidence": {"kpi": "x"},
        "response": "...",
        "llm_engine_used": "Template Formatter (Azure OpenAI error)",
        "llm_readiness_score": 95.0,  # gate passed
    }
    trace, summary = _build_execution_trace(state, elapsed_seconds=0.2)
    narration = next(s for s in trace if s["step"] == "narration")
    assert narration["gate"]["passed"] is True
    assert narration["engine"] == "Template Formatter (Azure OpenAI error)"
    assert "Azure OpenAI call failed" in narration["reason"]
    assert "Template Formatter (Azure OpenAI error)" in narration["reason"]
    assert summary["fallback_used"] is True


def test_narration_readiness_too_low_is_a_clean_never_attempted():
    state = {
        "intent": "show_kpi",
        "evidence": {"kpi": "x"},
        "response": "...",
        "llm_engine_used": "Template Formatter",
        "llm_readiness_score": 20.0,  # gate failed
    }
    trace, summary = _build_execution_trace(state, elapsed_seconds=0.2)
    narration = next(s for s in trace if s["step"] == "narration")
    assert narration["gate"]["passed"] is False
    assert "below the" in narration["reason"]
    assert summary["fallback_used"] is True


def test_early_exit_response_only_path_produces_short_trace():
    state = {
        "intent": "variance",
        "kpi_name": "nonexistent_kpi",
        "evidence": None,  # handler returned {"response": ...} directly, no evidence dict
        "response": "KPI 'nonexistent_kpi' not found in Rule Engine.",
    }
    trace, summary = _build_execution_trace(state, elapsed_seconds=0.05)

    assert len(trace) == 2  # intent_detection + the early-exit entry, nothing else
    assert trace[1]["step"] == "variance"
    assert trace[1]["engine"] is None
    assert trace[1]["gate"] is None
    assert trace[1]["reason"] == state["response"]
    assert summary["ml_engine"] is None
    assert summary["narration_engine"] is None
    assert summary["fallback_used"] is False


def test_execution_time_is_recorded():
    _, summary = _build_execution_trace({"intent": "show_kpi", "evidence": None, "response": "x"}, elapsed_seconds=1.5)
    assert summary["execution_time_seconds"] == 1.5


# ── _summarize_breakdown ──────────────────────────────────────────────────────

def test_summarize_breakdown_names_strongest_and_weakest():
    breakdown = {"evidence": [
        {"dimension": "completeness", "value": 0.95},
        {"dimension": "feature_coverage", "value": 1.0},
        {"dimension": "data_freshness", "value": 0.55},
    ]}
    summary = _summarize_breakdown(breakdown)
    assert "Strongest on feature_coverage (100%)" in summary
    assert "weakest on data_freshness (55%)" in summary


def test_summarize_breakdown_handles_single_dimension():
    summary = _summarize_breakdown({"evidence": [{"dimension": "completeness", "value": 0.9}]})
    assert summary == "Strongest/only factor: completeness (90%)."


def test_summarize_breakdown_returns_none_for_missing_or_empty_evidence():
    assert _summarize_breakdown(None) is None
    assert _summarize_breakdown({}) is None
    assert _summarize_breakdown({"evidence": []}) is None


def test_ml_gate_reason_includes_breakdown_summary_when_supplied():
    state = {
        "intent": "forecast",
        "evidence": {"kpi": "x"},
        "response": "...",
        "ml_readiness_score": 82.0,
        "ml_readiness_breakdown": {"evidence": [
            {"dimension": "feature_coverage", "value": 1.0},
            {"dimension": "data_freshness", "value": 0.55},
        ]},
    }
    trace, _ = _build_execution_trace(state, elapsed_seconds=0.2)
    ml_step = next(s for s in trace if s["step"] == "forecast")
    assert ml_step["gate"]["breakdown"] == state["ml_readiness_breakdown"]
    assert "weakest on data_freshness" in ml_step["reason"]


def test_narration_reason_includes_llm_breakdown_summary_when_supplied():
    state = {
        "intent": "show_kpi",
        "evidence": {"kpi": "x"},
        "response": "...",
        "llm_engine_used": "Azure OpenAI",
        "llm_readiness_score": 95.0,
        "llm_readiness_breakdown": {"evidence": [{"dimension": "description_coverage", "value": 0.9}]},
    }
    trace, _ = _build_execution_trace(state, elapsed_seconds=0.2)
    narration = next(s for s in trace if s["step"] == "narration")
    assert narration["gate"]["breakdown"] == state["llm_readiness_breakdown"]
    assert "description_coverage" in narration["reason"]


# ── _model_version_for_engine ─────────────────────────────────────────────────

def test_model_version_prophet_is_refit_per_query_not_a_training_date():
    version = _model_version_for_engine("Prophet", _REGISTRY)
    assert version["refit_per_query"] is True
    assert "trained_at" not in version
    assert version["last_run_at"] == _REGISTRY["prophet_forecast"]["timestamp"]


def test_model_version_isolation_forest_has_trained_at_and_no_accuracy():
    version = _model_version_for_engine("IsolationForest", _REGISTRY)
    assert version["trained_at"] == _REGISTRY["isolation_forest"]["timestamp"]
    assert version["accuracy_metric"] is None  # unsupervised — no fabricated number


def test_model_version_kmeans_has_trained_at_and_no_accuracy():
    version = _model_version_for_engine("K-Means", _REGISTRY)
    assert version["trained_at"] == _REGISTRY["kmeans_segmentation"]["timestamp"]
    assert version["accuracy_metric"] is None


def test_model_version_returns_none_for_unknown_engine_or_empty_registry():
    assert _model_version_for_engine("SomeOtherModel", _REGISTRY) is None
    assert _model_version_for_engine("Prophet", {}) is None


def test_ml_gated_happy_path_attaches_model_version():
    state = {
        "intent": "anomaly_detection",
        "evidence": {"kpi": "x"},  # no ml_readiness_blocked -> model path ran
        "response": "...",
        "ml_readiness_score": 92.0,
    }
    trace, _ = _build_execution_trace(state, elapsed_seconds=0.2)
    step = next(s for s in trace if s["step"] == "anomaly_detection")
    assert step["model_version"]["accuracy_metric"] is None
    # Read fresh rather than the module-level _REGISTRY snapshot — this
    # pre-existing shared JSON file (ml/model_registry.json) gets rewritten
    # by other tests' live IsolationForest fits (e.g. test_analytics_graph.py's
    # end-to-end anomaly_detection case) that may run before this one in a
    # full-suite pass; _build_execution_trace() itself always reads live.
    assert step["model_version"]["trained_at"] == _load_model_registry()["isolation_forest"]["timestamp"]


def test_ml_gated_fallback_path_has_no_model_version():
    state = {
        "intent": "anomaly_detection",
        "evidence": {"ml_readiness_blocked": True, "ml_readiness_score": 40.0,
                      "fallback_reason": "below threshold", "fallback_applied": "Z-Score fallback"},
        "response": "...",
        "ml_readiness_score": 40.0,
    }
    trace, _ = _build_execution_trace(state, elapsed_seconds=0.2)
    step = next(s for s in trace if s["step"] == "anomaly_detection")
    assert "model_version" not in step  # nothing ran, so nothing to version


def test_root_cause_xgboost_corroboration_cites_real_registry_accuracy():
    state = {
        "intent": "root_cause",
        "evidence": {"kpi": "x", "ml_predicted_driver": "claim_frequency_variance", "ml_confidence": 0.8},
        "response": "...",
    }
    trace, _ = _build_execution_trace(state, elapsed_seconds=0.2)
    step = next(s for s in trace if s["step"] == "root_cause")
    xgb = _REGISTRY.get("xgboost_classifier", {})
    if "accuracy" in xgb:
        assert f"{xgb['accuracy']:.1%}" in step["reason"]


def test_forecast_key_drivers_cites_real_lightgbm_r2():
    state = {
        "intent": "forecast",
        "evidence": {"kpi": "x", "key_drivers": [{"feature": "claim_frequency", "importance": 0.4}]},
        "response": "...",
        "ml_readiness_score": 99.75,
    }
    trace, _ = _build_execution_trace(state, elapsed_seconds=0.2)
    step = next(s for s in trace if s["step"] == "forecast")
    # Read fresh rather than the module-level _REGISTRY snapshot — same
    # reasoning as test_ml_gated_happy_path_attaches_model_version above:
    # other tests' live LightGBM fits during a full-suite run can rewrite
    # ml/model_registry.json after this module's import-time snapshot was
    # taken, while _build_execution_trace() itself always reads live.
    lgbm = _load_model_registry().get("lightgbm_regressor", {})
    if "r2" in lgbm:
        assert "LightGBM" in step["reason"]
        assert f"{lgbm['r2']:.3f}" in step["reason"]


# ── Per-step duration_ms ──────────────────────────────────────────────────────

def test_duration_ms_attached_when_node_durations_supplied():
    state = {
        "intent": "show_kpi",
        "evidence": {"kpi": "x"},
        "response": "...",
        "llm_engine_used": "Azure OpenAI",
        "llm_readiness_score": 99.0,
    }
    # Agent 3 redesign, Phase 4 — there's no more per-intent handle_{step}
    # node; interpret_question produces "intent_detection" and the single
    # execute_analyses node produces every analysis-type step.
    node_durations_ms = {"interpret_question": 4.7, "execute_analyses": 345.3, "narrate": 12.1}
    trace, _ = _build_execution_trace(state, elapsed_seconds=0.5, node_durations_ms=node_durations_ms)

    intent_step = next(s for s in trace if s["step"] == "intent_detection")
    kpi_step = next(s for s in trace if s["step"] == "show_kpi")
    narration_step = next(s for s in trace if s["step"] == "narration")
    assert intent_step["duration_ms"] == 4.7
    assert kpi_step["duration_ms"] == 345.3
    assert narration_step["duration_ms"] == 12.1


def test_duration_ms_omitted_when_node_durations_not_supplied():
    """Backward compatibility: the 2-positional-arg call shape every other
    test in this file uses must keep working unchanged."""
    state = {"intent": "show_kpi", "evidence": {"kpi": "x"}, "response": "..."}
    trace, _ = _build_execution_trace(state, elapsed_seconds=0.2)
    for entry in trace:
        assert "duration_ms" not in entry


# ── Real end-to-end wiring check ──────────────────────────────────────────────

pytestmark_dataset = pytest.mark.skipif(
    not _DATASET.exists(), reason=f"Insurance test dataset not found at {_DATASET}",
)


@pytestmark_dataset
def test_end_to_end_forecast_happy_path_reports_prophet():
    with open(_DATASET, "rb") as f:
        content = f.read()
    result = run_analytics_graph(
        file_content=content,
        business_question="Forecast underwriting result for next 6 months",
        conversation_id=str(uuid.uuid4()),
        ml_readiness_score=99.75,
        llm_readiness_score=99.75,
        detected_domain="Insurance",  # see test_harness.py's run_test_case() comment
    )
    assert result["status"] == "ok"
    # Only asserting the ML engine selection here — narration engine depends
    # on Azure OpenAI's real-time availability (rate limits, outages), which this
    # test isn't about and shouldn't be flaky over. See the narration_engine
    # / fallback_used unit tests above for that behavior, tested deterministically.
    assert result["execution_summary"]["ml_engine"] == "Prophet"
    assert result["execution_summary"]["execution_time_seconds"] > 0
    forecast_step = next(s for s in result["execution_trace"] if s["step"] == "forecast" and s["engine"] == "Prophet")
    assert forecast_step["duration_ms"] > 0
    assert forecast_step["model_version"]["refit_per_query"] is True
    # Every trace entry should have real per-step timing now, via graph.stream().
    assert all(step.get("duration_ms", 0) >= 0 for step in result["execution_trace"])


@pytestmark_dataset
def test_end_to_end_forecast_fallback_path_reports_historical_trend():
    with open(_DATASET, "rb") as f:
        content = f.read()
    result = run_analytics_graph(
        file_content=content,
        business_question="Forecast underwriting result for next 6 months",
        conversation_id=str(uuid.uuid4()),
        ml_readiness_score=40.0,
        llm_readiness_score=99.75,
        detected_domain="Insurance",  # see test_harness.py's run_test_case() comment
    )
    assert result["status"] == "ok"
    assert result["execution_summary"]["fallback_used"] is True
    # "Linear Trend" (Stage 6's registry-driven deterministic forecast
    # strategy, Phase 1/2 — InsurancePlugin pins it as the closest match to
    # today's analytics_tool.trend()) replaces the old bespoke
    # "Historical Trend Analysis (analytics_tool.trend)" display string.
    assert "Linear Trend" in result["execution_summary"]["ml_engine"]
    forecast_step = next(s for s in result["execution_trace"] if s["step"] == "forecast")
    assert "model_version" not in forecast_step  # fallback path — no model ran, nothing to version
    assert forecast_step["duration_ms"] > 0


# ── AnalysisResponse round-trip ───────────────────────────────────────────────

def test_analysis_response_accepts_trace_and_summary_fields():
    from app.schemas.responses import AnalysisResponse

    resp = AnalysisResponse(
        status="ok", query="q", response="r", conversation_id="c",
        ml_readiness_score_used=99.75, llm_readiness_score_used=99.75,
        execution_trace=[{"step": "intent_detection", "engine": None, "gate": None, "reason": "x"}],
        execution_summary={"intent": "show_kpi", "tools_used": [], "ml_engine": None,
                            "narration_engine": "Azure OpenAI", "execution_time_seconds": 0.1, "fallback_used": False},
    )
    assert resp.execution_trace[0]["step"] == "intent_detection"
    assert resp.execution_summary["narration_engine"] == "Azure OpenAI"


def test_analysis_response_defaults_trace_fields_to_none():
    from app.schemas.responses import AnalysisResponse

    resp = AnalysisResponse(status="error", query="q", response="r")
    assert resp.execution_trace is None
    assert resp.execution_summary is None


# ── _build_multi_analysis_trace — ungated step gate must never be self-contradictory ──

def test_multi_analysis_trace_ungated_step_gets_no_gate_object_even_above_threshold():
    """Regression test for a bug caught via live testing: 'trend' has no
    capability gate (deterministic-first by design), so its trace entry
    must show gate=None -- never a fabricated {score, threshold,
    passed: False} built from the CURRENT ml_readiness_score, which is
    self-contradictory whenever that score is actually above threshold
    (e.g. score=76.1, threshold=75, passed=False -- exactly what a live
    /pipeline/run response showed before this fix)."""
    final_state = {
        "ml_readiness_score": 76.1,
        "ml_readiness_breakdown": {"assessment_type": "ml_readiness", "score": 76.1},
        "llm_engine_used": "Azure OpenAI",
        "llm_readiness_score": 94.24,
        "llm_readiness_breakdown": {"assessment_type": "llm_readiness", "score": 94.24},
        "tools_used": ["RuleEngine", "SQLTool", "AnalyticsTool", "MLTool→Prophet", "ExplanationTool"],
        "question_intent": None,
    }
    trend_evidence = Evidence(
        evidence={"trend_slope": 1.2},
        fallback_metadata={
            "ml_readiness_blocked": False,
            "fallback_reason": "'trend' has no ML-capable algorithm registered — deterministic by design",
            "fallback_applied": "Trend (linear regression over time)",
        },
    )
    forecast_evidence = Evidence(
        evidence={"forecast": []},
        model_metadata={"algorithm": "Prophet", "cost_tier": "expensive"},
        confidence=0.761,
    )
    entries = [("trend", "TrendAnalyzer", trend_evidence), ("forecast", "ForecastAnalyzer", forecast_evidence)]

    trace, summary = _build_multi_analysis_trace(final_state, entries, elapsed_seconds=28.2, node_durations_ms=None)

    trend_entry = next(e for e in trace if e["step"] == "trend")
    forecast_entry = next(e for e in trace if e["step"] == "forecast")
    assert trend_entry["gate"] is None
    assert forecast_entry["gate"]["passed"] is True
    assert forecast_entry["gate"]["score"] == 76.1
