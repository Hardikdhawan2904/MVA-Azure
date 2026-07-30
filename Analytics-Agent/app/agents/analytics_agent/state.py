"""app/agents/analytics_agent/state.py — typed state threaded through the
Analytics Agent's LangGraph.

Mirrors the shape of main.py::AnalyticsAgent.process()'s old locals (intent,
kpi_name, filters, evidence) as explicit state fields instead — graph nodes
only communicate through this shared dict, so every intermediate value a
later node needs has to be named here.
"""

from typing import Any, TypedDict

from app.services.capability_resolution.models import CapabilityProfile
from app.services.dataset_context.models import DatasetContext
from app.services.evidence.evidence_builder import EvidenceBuilder
from app.services.kpi_discovery.models import DiscoveredKPI
from app.services.planning.models import AnalyticsPlan
from app.services.question_interpreter.models import QuestionIntent
from app.services.scheduling.models import ScheduledPlan


class AnalyticsState(TypedDict, total=False):
    # ── Inputs (set once, at graph invocation) ──────────────────────────────
    business_question: str
    dataset_path: str                    # temp CSV path for this request
    conversation_id: str                 # ties this request's memory to prior turns
    ml_readiness_score: float
    llm_readiness_score: float
    feature_recommendation: list[dict] | None
    # Agent 2's full readiness assessments (strengths, blocking_issues,
    # evidence) — optional, since a direct /analyze call outside the
    # Orchestrator has no way to supply them. Read by graph.py's
    # _build_execution_trace to explain *why* a readiness gate passed or
    # failed, not just report the bare score.
    ml_readiness_breakdown: dict | None
    llm_readiness_breakdown: dict | None

    # ── Set by build_dataset_context (Stage 0) ──────────────────────────────
    # Rich (Agent 2-derived) when the Orchestrator forwarded column_profiles/
    # hierarchy/charts, a LocalSchemaInferer-built fallback otherwise.
    dataset_context: DatasetContext

    # ── Set by resolve_capabilities / discover_kpis / interpret_question
    # (Stages 1-3, Phase 4) ──────────────────────────────────────────────────
    capability_profile: CapabilityProfile
    discovered_kpis: list[DiscoveredKPI]
    question_intent: QuestionIntent | None
    # The curated RuleEngine KPI dict interpret_question resolved (if any) —
    # kept separately from question_intent.resolved_kpi_name (a bare string)
    # so execute_analyses/enhance_plan don't need a second RuleEngine lookup.
    resolved_kpi: dict | None

    # ── Set by plan_analytics / schedule (Stages 4-5, Phase 4) ──────────────
    analytics_plan: AnalyticsPlan
    scheduled_plan: ScheduledPlan

    # ── Set by execute_analyses (Stages 6-7, Phase 4) ───────────────────────
    evidence_builder: EvidenceBuilder
    # Backward-compat adapter fields (plan's Phase 4 Decision 6) — populated
    # by execute_analyses only in the single-scheduled-analysis case (100%
    # of today's Insurance flow), so graph.py's _build_execution_trace()
    # needs no logic changes for that case. Unset for multi-analysis
    # "report mode" — that trace shape is built from evidence_builder
    # directly instead.
    intent: str
    kpi_name: str
    evidence: dict[str, Any]

    # ── Output ───────────────────────────────────────────────────────────────
    response: str

    # ── Set by narrate() / record_memory() — read by graph.py's post-hoc
    # execution-trace builder, not consumed inside the graph itself ────────
    llm_engine_used: str          # "Azure OpenAI" | "Template Formatter" | "Template Formatter (Azure OpenAI error)"
    tools_used: list[str]         # from _build_tools_used(state), same list memory.add_turn() already gets
