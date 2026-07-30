"""Profiling pipeline as a LangGraph StateGraph.

Topology (fixed edges except where noted):

  validate_and_load → profile_data → schema_intelligence ─┬─(retry)─→ schema_intelligence
                                                            └─(continue)─→ classify_and_build_hierarchy
  classify_and_build_hierarchy → evaluate_business_rules → assess_quality
  assess_quality ─(quality_gate_router)─┬─"full"───→ ┬─ generate_suggestions ─────┬─→ assess_readiness
                                        │             └─ recommend_target_features ┘   → generate_charts → finalize → END
                                        └─"lightweight"→ finalize_lightweight → END

generate_suggestions and recommend_target_features each have their own internal ReAct
sub-graph (see rule_suggestion_agent/ and feature_target_agent/, self-contained sub-agent
packages with their own agent.yaml/config.py/state.py/tools.py/graph.py) — from this graph's
point of view each is a single node, same as schema_intelligence. recommend_target_features
only actually calls its LLM sub-graph when a business_question or target_column was supplied
in the request; otherwise it computes a deterministic (non-LLM) feature/drop split so
assess_readiness always has something to work with.

generate_suggestions and recommend_target_features are siblings, not a sequential pair:
neither reads the other's output (recommend_target_features only reads col_profiles/
sem_candidates/grain_result/business_question/target_column_override; rule_suggestions
is only read back out of final state at the very end, never by assess_readiness or
anything between them). Each is its own multi-turn ReAct loop, so run serially they add;
LangGraph runs same-superstep nodes with no edge between them concurrently (a thread per
node for sync graphs), so branching them here turns two additive LLM-loop durations into
roughly the slower of the two.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import StateGraph, END

from app.core.config import Settings
from app.core.enums import RunStatus
from app.core.logging import get_logger
from app.repositories.configuration_repository import ConfigurationRepository
from app.services.ingestion.temporary_storage import TemporaryStorage
from app.services.schema_intelligence.interface import SchemaIntelligenceProvider
from app.services.orchestration.profiling_orchestrator import PipelineResult

from app.agents.data_profiling_agent.config import get_entry_point
from app.agents.data_profiling_agent.state import ProfilingState
from app.agents.data_profiling_agent.nodes.pipeline import ProfilingGraphNodes

logger = get_logger(__name__)


def build_profiling_graph(
    settings: Settings,
    config_repo: ConfigurationRepository,
    schema_intelligence: SchemaIntelligenceProvider,
    temp_storage: TemporaryStorage,
):
    """Compile the profiling StateGraph. Call once per request (mirrors how
    ProfilingOrchestrator was previously constructed fresh per request)."""
    nodes = ProfilingGraphNodes(settings, config_repo, schema_intelligence, temp_storage)

    g = StateGraph(ProfilingState)

    g.add_node("validate_and_load", nodes.validate_and_load)
    g.add_node("profile_data", nodes.profile_data)
    g.add_node("schema_intelligence", nodes.schema_intelligence)
    g.add_node("classify_and_build_hierarchy", nodes.classify_and_build_hierarchy)
    g.add_node("evaluate_business_rules", nodes.evaluate_business_rules)
    g.add_node("assess_quality", nodes.assess_quality)
    g.add_node("generate_suggestions", nodes.generate_suggestions)
    g.add_node("recommend_target_features", nodes.recommend_target_features)
    g.add_node("assess_readiness", nodes.assess_readiness)
    g.add_node("generate_charts", nodes.generate_charts)
    g.add_node("finalize", nodes.finalize)
    g.add_node("finalize_lightweight", nodes.finalize_lightweight)

    g.set_entry_point(get_entry_point())

    g.add_edge("validate_and_load", "profile_data")
    g.add_edge("profile_data", "schema_intelligence")

    g.add_conditional_edges(
        "schema_intelligence",
        nodes.schema_intelligence_router,
        {"retry": "schema_intelligence", "continue": "classify_and_build_hierarchy"},
    )

    g.add_edge("classify_and_build_hierarchy", "evaluate_business_rules")
    g.add_edge("evaluate_business_rules", "assess_quality")

    g.add_conditional_edges(
        "assess_quality",
        nodes.quality_gate_router,
        ["generate_suggestions", "recommend_target_features", "finalize_lightweight"],
    )

    g.add_edge("generate_suggestions", "assess_readiness")
    g.add_edge("recommend_target_features", "assess_readiness")
    g.add_edge("assess_readiness", "generate_charts")
    g.add_edge("generate_charts", "finalize")
    g.add_edge("finalize", END)
    g.add_edge("finalize_lightweight", END)

    return g.compile()


def _state_to_pipeline_result(
    run_id: uuid.UUID, primary_domain: str, started_at: str, state: dict[str, Any]
) -> PipelineResult:
    """Adapt the graph's raw state dict into the same PipelineResult shape
    ProfilingOrchestrator.execute() used to return, so downstream consumers
    (profile_runs.py's persistence code) don't need to know the pipeline is
    now graph-based."""
    result = PipelineResult()
    result.run_id = str(run_id)
    result.primary_domain = primary_domain
    result.started_at = started_at
    result.completed_at = state.get("completed_at")
    result.warnings = state.get("warnings") or []

    status = state.get("status", RunStatus.COMPLETED.value)
    result.status = status if isinstance(status, RunStatus) else RunStatus(status)
    result.error = state.get("error")

    if result.status == RunStatus.FAILED:
        return result

    ds_profile_obj = state.get("dataset_profile_obj")
    result.dataset_profile = ds_profile_obj.to_dict() if ds_profile_obj is not None else {}
    result.column_profiles = state.get("column_output") or []

    sec_domain = state.get("sec_domain")
    result.secondary_domain = sec_domain.to_dict() if sec_domain is not None else {}

    result.column_classifications = [c.to_dict() for c in state.get("classifications") or []]

    hierarchy_result = state.get("hierarchy_result")
    result.hierarchy = hierarchy_result.to_dict() if hierarchy_result is not None else {}

    result.quality_assessments = state.get("quality_dims") or []
    result.overall_quality = state.get("overall_quality") or {}
    result.readiness_assessments = state.get("readiness_assessments") or []
    result.feature_recommendation = state.get("feature_recommendation") or {}
    result.charts = state.get("charts") or []
    result.drill_down_cubes = state.get("drill_down_cubes") or []
    result.rule_evaluations = [r.to_dict() for r in state.get("rule_results") or []]
    result.rule_suggestions = state.get("rule_suggestions") or []

    return result


def run_profiling_graph(
    settings: Settings,
    config_repo: ConfigurationRepository,
    schema_intelligence: SchemaIntelligenceProvider,
    temp_storage: TemporaryStorage,
    run_id: uuid.UUID,
    file_content: bytes,
    filename: str,
    primary_domain: str,
    schema_metadata: dict[str, Any] | None = None,
    request_rules: list[dict[str, Any]] | None = None,
    sheet_name: str | None = None,
    approved_rule_dicts: list[dict[str, Any]] | None = None,
    business_question: str | None = None,
    target_column_override: str | None = None,
) -> PipelineResult:
    """
    Execute the profiling graph and return a PipelineResult — the exact same
    shape ProfilingOrchestrator.execute() returned, so callers are unaffected
    by the pipeline internals moving to a graph.

    Wraps the graph invocation in the same try/except/finally guarantee the
    old linear execute() had: on any failure, return a FAILED result instead
    of raising, and always clean up the temp file.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    graph = build_profiling_graph(settings, config_repo, schema_intelligence, temp_storage)

    initial_state: ProfilingState = {
        "run_id": run_id,
        "file_content": file_content,
        "filename": filename,
        "primary_domain": primary_domain,
        "schema_metadata": schema_metadata,
        "request_rules": request_rules,
        "sheet_name": sheet_name,
        "approved_rule_dicts": approved_rule_dicts,
        "business_question": business_question,
        "target_column_override": target_column_override,
        "schema_intelligence_retry_count": 0,
        "rule_suggestion_retry_count": 0,
        "warnings": [],
    }

    try:
        state = graph.invoke(initial_state, config={"recursion_limit": 50})
        return _state_to_pipeline_result(run_id, primary_domain, started_at, state)
    except Exception as e:
        logger.error("profiling_graph_failed", run_id=str(run_id), error=str(e))
        result = PipelineResult()
        result.run_id = str(run_id)
        result.primary_domain = primary_domain
        result.started_at = started_at
        result.completed_at = datetime.now(timezone.utc).isoformat()
        result.status = RunStatus.FAILED
        result.error = {"code": getattr(e, "code", "PROCESSING_ERROR"), "message": str(e)}
        return result
    finally:
        temp_storage.cleanup_run(run_id)
