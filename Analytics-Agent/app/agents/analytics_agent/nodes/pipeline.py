"""app/agents/analytics_agent/nodes/pipeline.py — LangGraph node functions
for the Analytics Agent.

Agent 3 redesign, Phase 4 (plan "zany-giggling-crayon") — this file
replaces the old intent-dispatch pipeline (detect_intent_and_filters +
7 handle_* methods + route_by_intent/route_after_handler) with the full
Stage 1-9 pipeline built in Phases 1-3: resolve_capabilities →
discover_kpis → interpret_question → plan_analytics → schedule →
execute_analyses → narrate → record_memory. See the plan's "Phase 4
Detailed Design" section for the full gap analysis and design rationale
this rewrite is based on.

`AnalyticsGraphNodes` is constructed fresh per request (see graph.py) —
not a shared singleton; see the module docstring history for why
(SQLTool's DuckDB connection, MLTool's readiness-scored models, and
ExplanationTool's LLM gate all have to be bound to this request's inputs).
"""

import logging
import re

import pandas as pd

from app.agents.analytics_agent.state import AnalyticsState
from app.services.tools.rule_engine import RuleEngine
from app.services.tools.sql_tool import SQLTool
from app.services.tools.analytics_tool import AnalyticsTool
from app.services.tools.root_cause_tool import RootCauseTool
from app.services.tools.explanation_tool import ExplanationTool
from app.services.tools.knowledge_update_service import KnowledgeUpdateTool
from app.services.tools.ml_tool import MLTool
from app.services.memory import MemoryManager
from app.services.ml.feature_validation import validate_feature_columns
from app.services.dataset_context.context_builder import DatasetContextBuilder
from app.services.domain_plugins.registry import PluginRegistry
from app.services.domain_plugins.generic_plugin import GenericDomainPlugin
from app.services.capability_resolution.analytics_capability_resolver import AnalyticsCapabilityResolver
from app.services.capability_resolution.models import capability_for_analysis_type
from app.services.kpi_discovery.semantic_kpi_discovery import SemanticKPIDiscovery
from app.services.question_interpreter.business_question_interpreter import BusinessQuestionInterpreter
from app.services.question_interpreter.models import QuestionIntent
from app.services.planning.analytics_planner import AnalyticsPlanner
from app.services.scheduling.analytics_scheduler import AnalyticsScheduler
from app.services.scheduling.budget_config import load_budget_config
from app.services.models_registry.model_registry import ModelRegistry
from app.services.models_registry.model_selector import ModelSelector
from app.services.analyzers.base import AnalyzerRegistry
from app.services.evidence.evidence_builder import Evidence, EvidenceBuilder

logger = logging.getLogger(__name__)

# Agent 3 redesign, Phase 2 — the old _DIMENSION_KEYWORDS/_KPI_ALIASES
# module-level dicts that used to live here moved verbatim into
# domain_plugins/insurance/plugin.py's get_intent_vocabulary() — this
# pipeline reads them from self._domain_plugin. See __init__ /
# _extract_filters / _detect_kpi.

_PERIOD_KEYWORDS = {
    "quarter": ["q1", "q2", "q3", "q4"],
}
# Fiscal year / calendar year are matched by pattern, not a fixed list, so
# this doesn't silently stop working the moment a query mentions a year
# outside whatever was hardcoded here.
_FISCAL_YEAR_RE = re.compile(r"\bfy(20\d{2})\b")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

# Single-winner curated-KPI intent resolution — the exact old _INTENT_MAP
# (main.py-era), minus "show_kpi"/"ytd" (both always fell through to the
# same show_kpi/kpi_summary default anyway — see _detect_intent's old
# docstring) and renamed to the vocabulary Stages 1-7 already ship/test
# ("anomaly_detection" not "anomaly", "segmentation" not "segment",
# "kpi_variance" not "variance"). Only consulted when interpret_question
# has already resolved a curated KPI (see that method) — the generic
# BusinessQuestionInterpreter's own broader multi-type keyword groups
# (Phase 1, unchanged) already cover newer capabilities (correlation,
# classification, ...) this narrower map intentionally excludes, and stay
# untouched for datasets with no resolvable KPI at all.
_KPI_FLOW_INTENT_MAP: list[tuple[str, list[str]]] = [
    ("forecast",         ["forecast", "predict", "projection", "next quarter", "future", "expected"]),
    ("anomaly_detection", ["anomaly", "anomalies", "irregular", "outlier", "unusual", "spike", "detect"]),
    ("root_cause",        ["why", "reason", "cause", "driver", "drove", "driven", "factors", "impact", "contributing", "explain why"]),
    ("kpi_variance",      ["variance", "vs budget", "vs prior", "vs plan", "deviation", "gap", "difference"]),
    ("trend",             ["trend", "over time", "qoq", "yoy", "mom", "quarter over quarter", "growth"]),
    ("segmentation",      ["segment portfolio", "segment risk", "segment by", "cluster", "group risk", "group by risk", "portfolio risk", "risk tier", "segment by risk"]),
]


class AnalyticsGraphNodes:
    """One bound method per LangGraph node. Constructed once per request
    with everything that request needs, exactly the way
    `main.py::AnalyticsAgent.__init__` used to construct one tool set per
    subprocess invocation."""

    def __init__(
        self,
        dataset_path: str,
        conversation_id: str,
        ml_readiness_score: float = 99.75,
        llm_readiness_score: float = 99.75,
        feature_recommendation: list[dict] | None = None,
        column_profiles: list[dict] | None = None,
        hierarchy: dict | None = None,
        charts: list[dict] | None = None,
        full_feature_recommendation: dict | None = None,
        detected_domain: str | None = None,
    ):
        validate_feature_columns(feature_recommendation)

        # Agent 3 redesign, Phase 2/4 — RuleEngine/SQLTool/RootCauseTool
        # are genuinely parameterizable; this pipeline sources their
        # config/columns from a resolved DomainPlugin instead of those
        # classes' own module-level Insurance defaults. Falls back to
        # GenericDomainPlugin (empty KPI catalog, fully generic Stage 1-8
        # engine) when the forwarded domain doesn't match a registered
        # plugin or wasn't forwarded at all — falling back to
        # InsurancePlugin here (as Phase 2 did, before the generic engine
        # existed to fall back to) would silently apply Insurance's
        # curated KPI catalog to an unrelated dataset. Confirmed live
        # during Phase 4 verification against a non-Insurance HR fixture.
        plugin = PluginRegistry().find_plugin(detected_domain) or GenericDomainPlugin()
        self._domain_plugin = plugin
        intent_vocabulary = plugin.get_intent_vocabulary()
        self._kpi_aliases = intent_vocabulary.get("kpi_aliases", {})
        self._dimension_keywords = intent_vocabulary.get("dimension_keywords", {})

        self.dataset_path = dataset_path
        self.rule_engine = RuleEngine(**plugin.get_rule_engine_paths())
        self.sql         = SQLTool(dataset_path, view_name=plugin.get_view_name())
        self.analytics   = AnalyticsTool()
        self.root_cause  = RootCauseTool(driver_columns=plugin.get_driver_columns(), driver_labels=plugin.get_driver_labels())
        self.explanation = ExplanationTool(llm_readiness_score)
        self.knowledge   = KnowledgeUpdateTool(self.rule_engine)
        self.ml          = MLTool(ml_readiness_score)
        self.memory      = MemoryManager(conversation_id)

        # Agent 3 redesign, Phase 4 — Stages 1-7's services, constructed
        # once per request (mirrors every other tool above).
        self.capability_resolver  = AnalyticsCapabilityResolver()
        self.kpi_discovery        = SemanticKPIDiscovery()
        self.question_interpreter = BusinessQuestionInterpreter()
        self.planner               = AnalyticsPlanner()
        self.scheduler              = AnalyticsScheduler()
        self.budget                 = load_budget_config()
        self.model_registry         = ModelRegistry()
        self.analyzer_registry      = AnalyzerRegistry()

        # Agent 3 redesign, Phase 0 — see plan "zany-giggling-crayon".
        # Forwarded by the Orchestrator when available; all optional, all
        # None on a direct /analyze call outside the Orchestrator —
        # build_dataset_context() falls back to LocalSchemaInferer.
        self.dataset_context_builder = DatasetContextBuilder()
        self._column_profiles = column_profiles
        self._hierarchy = hierarchy
        self._charts = charts
        self._full_feature_recommendation = full_feature_recommendation
        self._detected_domain = detected_domain

    # ── Stage 0 — Dataset Context (entry node) ──────────────────────────────

    def build_dataset_context(self, state: AnalyticsState) -> dict:
        df = pd.read_csv(state["dataset_path"])
        context = self.dataset_context_builder.build(
            df,
            column_profiles=self._column_profiles,
            hierarchy=self._hierarchy,
            charts=self._charts,
            full_feature_recommendation=self._full_feature_recommendation,
            detected_domain=self._detected_domain,
        )
        logger.info(
            f"dataset_context built: source={context.context_source}, "
            f"columns={context.column_count}, rows={context.row_count}"
        )
        return {"dataset_context": context}

    # ── Stage 1 — Capability Resolution ─────────────────────────────────────

    def resolve_capabilities(self, state: AnalyticsState) -> dict:
        profile = self.capability_resolver.resolve(
            state["dataset_context"], state["ml_readiness_score"], state["llm_readiness_score"],
        )
        return {"capability_profile": profile}

    # ── Stage 2 — Semantic KPI Discovery ─────────────────────────────────────

    def discover_kpis(self, state: AnalyticsState) -> dict:
        kpis = self.kpi_discovery.discover(state["dataset_context"])
        return {"discovered_kpis": kpis}

    # ── Stage 3 — Question Interpretation (+ KPI/filter resolution) ────────

    def interpret_question(self, state: AnalyticsState) -> dict:
        business_question = state.get("business_question") or ""
        query_lower = business_question.lower()

        base_intent = self.question_interpreter.interpret(
            business_question or None, state["capability_profile"], state["discovered_kpis"],
            state["dataset_context"],
        )

        # KPI name + filter resolution — relocated verbatim from the old
        # detect_intent_and_filters/_detect_kpi/_extract_filters (domain-
        # plugin vocabulary only, unchanged since Phase 2).
        detected_kpi_name = self._detect_kpi(query_lower)
        resolved_kpi_name = (
            detected_kpi_name
            or self.memory.get_last_kpi()
            or self._domain_plugin.get_default_kpi_name()
        )
        filters = {**self.memory.get_last_filters(), **self._extract_filters(query_lower)}

        self.explanation.set_conversation_context(self.memory.get_context_for_llm())

        kpi = None
        if resolved_kpi_name:
            kpi = self.rule_engine.get_kpi(resolved_kpi_name)
            if not kpi:
                logger.info(f"KPI '{resolved_kpi_name}' missing — triggering Knowledge Update Tool")
                update_result = self.knowledge.update(resolved_kpi_name)
                if update_result.get("success"):
                    kpi = self.rule_engine.get_kpi(resolved_kpi_name)

        if detected_kpi_name and not kpi:
            # The question explicitly named a KPI (by alias) and it
            # couldn't be resolved even via auto-generation — mirrors
            # handle_show_kpi's "not found and could not be auto-generated"
            # early exit rather than silently falling through to
            # unrestricted generic planning.
            return {
                "question_intent": None,
                "resolved_kpi": None,
                "kpi_name": detected_kpi_name,
                "response": f"KPI '{detected_kpi_name}' not found and could not be auto-generated.",
            }

        candidate_types = set(base_intent.candidate_analysis_types) if base_intent else set()
        rationale = base_intent.rationale if base_intent else ""
        mentioned_kpis = base_intent.mentioned_kpis if base_intent else []
        preferred_metrics = base_intent.preferred_metrics if base_intent else []
        preferred_dimensions = base_intent.preferred_dimensions if base_intent else []

        if kpi:
            # Curated-KPI territory: reproduce the old _detect_intent's
            # single-winner-per-query resolution exactly (see
            # _KPI_FLOW_INTENT_MAP's docstring) rather than accepting the
            # generic Interpreter's broader multi-type set — "Forecast
            # underwriting result" must schedule exactly one analysis
            # (forecast), not forecast+trend+time_series_analysis+
            # anomaly_detection all at once, to stay byte-identical with
            # today's Insurance response shape. Newer capability keyword
            # groups (correlation/classification/ranking/...) the KPI-flow
            # map intentionally doesn't cover fall through to base_intent
            # unchanged — this narrowing never disables genuinely new
            # analysis types, only the ones the old system already had a
            # single-winner opinion about.
            kpi_flow_type = next((t for t, kws in _KPI_FLOW_INTENT_MAP if any(kw in query_lower for kw in kws)), None)
            if kpi_flow_type:
                candidate_types = {kpi_flow_type}
                rationale = f"Curated-KPI query matched '{kpi_flow_type}' for '{resolved_kpi_name}'"
            elif not candidate_types:
                # Direct generalization of the old _detect_intent's "always
                # resolves to something, defaulting to show_kpi" fallback.
                candidate_types = {"kpi_summary"}
                rationale = rationale or f"Defaulted to curated KPI query for '{resolved_kpi_name}'"

        question_intent = None
        if candidate_types or kpi:
            question_intent = QuestionIntent(
                candidate_analysis_types=candidate_types,
                mentioned_kpis=mentioned_kpis,
                rationale=rationale,
                resolved_kpi_name=resolved_kpi_name if kpi else None,
                filters=filters,
                preferred_metrics=preferred_metrics,
                preferred_dimensions=preferred_dimensions,
            )

        logger.info(f"QuestionIntent: types={candidate_types or None}, kpi={resolved_kpi_name if kpi else None}, filters={filters}")
        return {
            "question_intent": question_intent,
            "resolved_kpi": kpi,
            "kpi_name": resolved_kpi_name if kpi else None,
        }

    def _extract_filters(self, query: str) -> dict:
        filters = {}
        for col, keywords in _PERIOD_KEYWORDS.items():
            for kw in keywords:
                if kw in query:
                    filters[col] = kw.upper()

        fy_match = _FISCAL_YEAR_RE.search(query)
        if fy_match:
            filters["fiscal_year"] = f"FY{fy_match.group(1)}"
        year_match = _YEAR_RE.search(query)
        if year_match:
            filters["year"] = year_match.group(1)

        for col, kw_map in self._dimension_keywords.items():
            for kw, val in kw_map.items():
                if re.search(rf"\b{re.escape(kw)}\b", query):
                    filters[col] = val

        if "ytd" in query or "year to date" in query:
            filters["ytd_flag"] = "Y"
        return filters

    def _detect_kpi(self, query: str) -> str | None:
        for alias, key in self._kpi_aliases.items():
            if alias in query:
                return key
        return None  # No explicit KPI mentioned — caller decides the fallback

    def route_after_interpret(self, state: AnalyticsState) -> str:
        """interpret_question can early-exit with "response" set directly
        (a query explicitly named an unresolvable KPI) — skip straight to
        record_memory rather than letting plan_analytics/schedule/
        execute_analyses run unrestricted and overwrite that message."""
        return "record_memory" if "response" in state else "plan_analytics"

    # ── Stage 4 — Analytics Planning ─────────────────────────────────────────

    def plan_analytics(self, state: AnalyticsState) -> dict:
        question_intent = state.get("question_intent")
        plan = self.planner.plan(
            state["dataset_context"], state["capability_profile"], state["discovered_kpis"],
            question_intent=question_intent,
        )
        plan = self._domain_plugin.enhance_plan(plan, state["dataset_context"], question_intent=question_intent)
        return {"analytics_plan": plan}

    # ── Stage 5 — Scheduling ─────────────────────────────────────────────────

    def schedule(self, state: AnalyticsState) -> dict:
        scheduled = self.scheduler.schedule(
            state["analytics_plan"], self.budget, question_intent=state.get("question_intent"),
        )
        return {"scheduled_plan": scheduled}

    # ── Stage 6+7 — Model Selection + Analyzer Execution ────────────────────

    def execute_analyses(self, state: AnalyticsState) -> dict:
        scheduled_plan = state.get("scheduled_plan") or []
        if not scheduled_plan:
            return {"response": "No analyses could be planned for this dataset and question."}

        question_intent = state.get("question_intent")
        filters = question_intent.filters if question_intent else {}
        resolved_kpi = state.get("resolved_kpi")
        dataset_context = state["dataset_context"]
        capability_profile = state["capability_profile"]

        df = self._fetch_filtered_dataset(filters)
        model_selector = ModelSelector(self.model_registry, self.budget)
        evidence_builder = EvidenceBuilder()

        for scheduled_analysis in scheduled_plan:
            analysis_type = scheduled_analysis.planned_analysis.analysis_type
            analyzer = self.analyzer_registry.get(analysis_type)
            if analyzer is None:
                logger.warning(f"No analyzer registered for analysis_type='{analysis_type}' — skipping")
                continue

            preferred_algorithm = self._domain_plugin.get_preferred_deterministic_strategy(analysis_type)
            selected_model = model_selector.select(
                scheduled_analysis, dataset_context, capability_profile, preferred_algorithm=preferred_algorithm,
            )
            capability = capability_for_analysis_type(analysis_type)
            ml_confidence = capability_profile.confidence_for(capability) if capability else None
            extra_context = self._build_extra_context(analysis_type, scheduled_analysis, resolved_kpi, filters, df)

            evidence = analyzer.execute(
                df, dataset_context, scheduled_analysis, selected_model,
                ml_confidence=ml_confidence, extra_context=extra_context,
            )

            if analysis_type == "root_cause":
                self._enrich_root_cause_evidence(evidence, df)

            evidence_builder.add(analysis_type, type(analyzer).__name__, evidence)

        return self._adapt_evidence_for_narration(evidence_builder)

    def _fetch_filtered_dataset(self, filters: dict) -> pd.DataFrame:
        """One shared, filtered fetch per request via SQLTool — every
        scheduled analysis (KPI-grounded or generic/structural alike)
        operates on slices/aggregates of this same DataFrame, replacing
        today's several bespoke per-handler queries. Same rows, same
        formulas as before — see the plan's Phase 4 Detailed Design,
        Decision 3."""
        return self.sql.get_kpi_data(["*"], filters=filters or None)

    def _build_extra_context(
        self, analysis_type: str, scheduled_analysis, resolved_kpi: dict | None, filters: dict, df: pd.DataFrame,
    ) -> dict:
        if analysis_type in ("kpi_summary", "kpi_variance"):
            return {"kpi": resolved_kpi, "filters": filters} if resolved_kpi else {}

        if analysis_type == "root_cause" and scheduled_analysis.planned_analysis.is_kpi_grounded and resolved_kpi:
            actual_col = resolved_kpi.get("actual_column")
            budget_col = resolved_kpi.get("budget_column")
            if actual_col and budget_col and actual_col in df.columns and budget_col in df.columns:
                unit = resolved_kpi.get("unit", "")
                method = "mean" if unit in ("%", "ratio", "rate") or resolved_kpi.get("category") == "ratio" else "sum"
                actual = self.analytics.aggregate(df, actual_col, method=method)
                budget = self.analytics.aggregate(df, budget_col, method=method)
                variance = self.analytics.variance_vs_budget(actual, budget, resolved_kpi.get("higher_is_better", True))
                return {"total_variance": variance["variance_amount"], "kpi": resolved_kpi}

        return {}

    def _enrich_root_cause_evidence(self, evidence: Evidence, df: pd.DataFrame) -> None:
        """Post-processing enrichment done outside the analyzer, exactly
        like the old handle_root_cause did it — driver descriptions and ML
        corroboration were never part of RootCauseTool.analyse() itself."""
        if evidence.evidence.get("mode") != "labeled_driver":
            return

        for d in evidence.evidence.get("driver_breakdown", []):
            d["description"] = self.rule_engine.get_variance_driver_description(d["driver"])

        try:
            ml_evidence = self.ml.get_root_cause_ml_evidence(df)
            if ml_evidence and "error" not in ml_evidence:
                evidence.evidence["ml_predicted_driver"] = ml_evidence["predicted_driver"]
                evidence.evidence["ml_confidence"] = ml_evidence["confidence"]
                evidence.evidence["ml_top_features"] = ml_evidence["top_features"]
        except Exception as e:
            logger.warning(f"ML root-cause evidence unavailable: {e}")

    def _adapt_evidence_for_narration(self, evidence_builder: EvidenceBuilder) -> dict:
        """Backward-compat adapter (plan's Phase 4 Decision 6): for the
        single-scheduled-analysis case — 100% of today's Insurance
        question-answering flow — populate state["intent"]/state["evidence"]
        exactly as the old handlers did, so graph.py's
        _build_execution_trace() needs zero logic changes. When that one
        analysis produced no evidence at all (an error/no-data condition),
        short-circuit straight to a response, bypassing narrate() — the
        same "early-exit" shape route_after_handler used to implement.
        Multi-analysis "report mode" (new capability) leaves intent/
        evidence unset; graph.py's trace builder branches on
        evidence_builder's entry count instead.
        """
        entries = evidence_builder.all()
        if len(entries) == 1:
            analysis_type, _analyzer_name, evidence = entries[0]
            if not evidence.evidence:
                reason = evidence.reasons[-1] if evidence.reasons else "No evidence was produced for this query."
                return {"evidence_builder": evidence_builder, "response": reason}
            return {"evidence_builder": evidence_builder, "intent": analysis_type, "evidence": evidence.flatten()}
        return {"evidence_builder": evidence_builder}

    # ── Narration / Memory (shared tail nodes) ──────────────────────────────

    def narrate(self, state: AnalyticsState) -> dict:
        evidence_builder = state.get("evidence_builder")
        narration_context = evidence_builder.to_narration_context() if evidence_builder else {}
        response = self.explanation.narrate(narration_context, state["business_question"])
        # last_engine_used is set as a side effect of the narrate() call
        # above — the only way to observe which path it actually took.
        # Surfaced into state for graph.py's execution trace.
        return {"response": response, "llm_engine_used": self.explanation.last_engine_used}

    def route_after_execute(self, state: AnalyticsState) -> str:
        """execute_analyses either wrote "response" directly (early-exit —
        nothing could be planned, or the single scheduled analysis
        produced no evidence) or "evidence_builder" for narrate() to turn
        into a response."""
        return "record_memory" if "response" in state else "narrate"

    def record_memory(self, state: AnalyticsState) -> dict:
        """Runs for every path — narrated or an early-exit error string —
        exactly like the old record_memory did for every branch."""
        response = state.get("response", "")
        intent = state.get("intent") or "report"
        kpi_name = state.get("kpi_name") or ""
        filters = state["question_intent"].filters if state.get("question_intent") else {}
        tools_used = self._build_tools_used(state)

        self.memory.add_turn(
            user_query       = state["business_question"],
            intent           = intent,
            kpi_name         = kpi_name,
            filters          = filters,
            tools_used       = tools_used,
            evidence         = {},
            response_summary = response[:300],
        )
        # Also surfaced into state so graph.py's execution-trace/summary
        # builder can reuse this exact list.
        return {"tools_used": tools_used}

    def _build_tools_used(self, state: AnalyticsState) -> list[str]:
        """Derived from what genuinely executed for this request (Phase 4
        Decision 7) rather than a static agent.yaml-declared chain — no
        existing test asserts the old chain's exact content."""
        tools = ["RuleEngine", "SQLTool", "AnalyticsTool"]
        evidence_builder = state.get("evidence_builder")
        if evidence_builder:
            for analysis_type, _analyzer_name, evidence in evidence_builder.all():
                if analysis_type == "root_cause":
                    tools.append("RootCauseTool")
                if evidence.model_metadata:
                    tools.append(f"MLTool→{evidence.model_metadata['algorithm']}")
        tools.append("ExplanationTool")

        seen: set[str] = set()
        deduped = []
        for t in tools:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped
