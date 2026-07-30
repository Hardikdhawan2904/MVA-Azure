"""Profiling graph nodes.

Mirrors ProfilingOrchestrator's exact service construction and step-by-step logic
(app/services/orchestration/profiling_orchestrator.py) — this file does not
reimplement any profiling/quality/chart logic, it only re-expresses the same
steps as graph nodes operating on shared state instead of local variables in
one linear method, so behavior stays identical to the pre-graph pipeline.

Two real new decision points that didn't exist before:
  - quality_gate_router: skips readiness/charts/suggestions on very poor data
  - schema_intelligence retry: retries once on a hard failure instead of
    silently accepting a fully-degraded result
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd

from app.core.config import Settings
from app.core.enums import (
    RunStatus, FileType, ColumnRole, HierarchyChainStatus, RuleSource,
)
from app.core.exceptions import UnsupportedDomainError
from app.core.logging import get_logger
from app.repositories.configuration_repository import ConfigurationRepository
from app.services.ingestion.file_validator import FileValidator
from app.services.ingestion.csv_loader import CSVLoader
from app.services.ingestion.xlsx_loader import XLSXLoader
from app.services.ingestion.temporary_storage import TemporaryStorage
from app.services.profiling.dataset_profiler import DatasetProfiler
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.profiling.type_refiner import TypeRefiner
from app.services.profiling.identifier_detector import IdentifierDetector
from app.services.profiling.semantic_candidate_generator import SemanticCandidateGenerator
from app.services.schema_intelligence.interface import SchemaIntelligenceProvider
from app.services.schema_intelligence.models import ColumnAnalysisInput, DomainContext
from app.services.domains.secondary_domain_classifier import SecondaryDomainClassifier
from app.services.classification.data_category_classifier import DataCategoryClassifier
from app.services.hierarchy.candidate_builder import HierarchyCandidateBuilder
from app.services.rules.rule_loader import RuleLoader
from app.services.rules.rule_engine import RuleEngine
from app.services.quality.completeness import assess_completeness
from app.services.quality.uniqueness import assess_uniqueness
from app.services.quality.validity import assess_validity
from app.services.quality.conformity import assess_conformity
from app.services.quality.consistency import assess_consistency
from app.services.quality.timeliness import assess_timeliness
from app.services.quality.integrity import assess_integrity
from app.services.quality.accuracy import assess_accuracy
from app.services.quality.business_rule_compliance import assess_business_rule_compliance
from app.services.quality.semantic_quality import assess_semantic_quality
from app.services.quality.overall_score import calculate_overall_score
from app.services.readiness.readiness_engine import ReadinessEngine
from app.services.charts.candidate_generator import ChartCandidateGenerator
from app.services.charts.aggregation_engine import AggregationEngine

from app.agents.data_profiling_agent.config import get_quality_gate_min_score
from app.agents.data_profiling_agent.state import ProfilingState

logger = get_logger(__name__)

# Quality score below this skips suggestions/readiness/charts/drill-down —
# there's no value profiling-in-depth or proposing rules from data already
# known to be poor. Declared in agent.yaml, not hardcoded here.
QUALITY_GATE_MIN_SCORE = get_quality_gate_min_score()


class ProfilingGraphNodes:
    """Builds every graph node as a bound method, sharing one set of service
    instances constructed exactly like ProfilingOrchestrator.__init__ did."""

    def __init__(
        self,
        settings: Settings,
        config_repo: ConfigurationRepository,
        schema_intelligence: SchemaIntelligenceProvider,
        temp_storage: TemporaryStorage,
    ):
        self._settings = settings
        self._config_repo = config_repo
        self._schema_intelligence = schema_intelligence
        self._temp_storage = temp_storage
        self._file_validator = FileValidator(settings)
        self._csv_loader = CSVLoader(settings)
        self._xlsx_loader = XLSXLoader(settings)
        self._dataset_profiler = DatasetProfiler()
        self._column_profiler = ColumnProfiler(settings)
        self._type_refiner = TypeRefiner()
        self._identifier_detector = IdentifierDetector()
        self._semantic_generator = SemanticCandidateGenerator()
        self._domain_classifier = SecondaryDomainClassifier(config_repo)
        self._category_classifier = DataCategoryClassifier(weights=config_repo.get_classification_weights())
        self._hierarchy_builder = HierarchyCandidateBuilder(config_repo)
        self._rule_loader = RuleLoader(config_repo)
        self._rule_engine = RuleEngine()
        self._readiness_engine = ReadinessEngine()
        self._chart_generator = ChartCandidateGenerator()
        self._agg_engine = AggregationEngine()

        self._llm_provider = schema_intelligence._llm if hasattr(schema_intelligence, "_llm") else None

    # ── Node 1: validate + load ─────────────────────────────────────────────

    def validate_and_load(self, state: ProfilingState) -> dict[str, Any]:
        run_id = state["run_id"]
        primary_domain = state["primary_domain"]

        supported = self._config_repo.get_supported_primary_domains()
        if primary_domain not in supported:
            raise UnsupportedDomainError(
                code="UNSUPPORTED_DOMAIN",
                message=f"Domain '{primary_domain}' is not supported.",
                details={"domain": primary_domain, "supported": supported},
            )

        file_type = self._file_validator.validate(state["filename"], len(state["file_content"]))
        file_path = self._temp_storage.save_upload(run_id, state["file_content"], state["filename"])

        if file_type == FileType.CSV:
            df = self._csv_loader.load(file_path)
        else:
            df = self._xlsx_loader.load(file_path, sheet_name=state.get("sheet_name"))

        return {"file_type": file_type, "file_path": file_path, "df": df}

    # ── Node 2: deterministic profiling stack ───────────────────────────────

    def profile_data(self, state: ProfilingState) -> dict[str, Any]:
        df = state["df"]
        ds_profile = self._dataset_profiler.profile(df)
        col_profiles = self._column_profiler.profile_all(df, ds_profile.normalized_keys)
        refined_types = [self._type_refiner.refine(p) for p in col_profiles]
        grain_result = self._identifier_detector.detect(col_profiles, refined_types)
        id_flags = [c.is_identifier for c in grain_result.identifier_candidates]
        sem_candidates = self._semantic_generator.generate_all(col_profiles, refined_types, id_flags)

        return {
            "dataset_profile_obj": ds_profile,
            "col_profiles": col_profiles,
            "refined_types": refined_types,
            "grain_result": grain_result,
            "sem_candidates": sem_candidates,
        }

    # ── Node 3: schema intelligence (LLM) ───────────────────────────────────

    def schema_intelligence(self, state: ProfilingState) -> dict[str, Any]:
        col_profiles = state["col_profiles"]
        refined_types = state["refined_types"]
        sem_candidates = state["sem_candidates"]
        primary_domain = state["primary_domain"]
        schema_metadata = state.get("schema_metadata")
        ds_profile = state["dataset_profile_obj"]

        si_inputs = [
            ColumnAnalysisInput(
                column_name=p.physical_name,
                normalized_key=p.normalized_key,
                description=_get_description(p.physical_name, schema_metadata),
                refined_physical_type=refined_types[i].value,
                statistics_summary=p.to_statistics_dict(),
                representative_sample_values=p.representative_values[:5],
                candidate_semantic_type=sem_candidates[i].candidate_semantic_type,
                candidate_column_role=sem_candidates[i].candidate_column_role.value,
                candidate_confidence=sem_candidates[i].candidate_confidence,
                primary_domain=primary_domain,
            )
            for i, p in enumerate(col_profiles)
        ]
        domain_ctx = DomainContext(
            primary_domain=primary_domain,
            row_count=ds_profile.row_count,
            column_count=ds_profile.column_count,
        )

        retry_count = state.get("schema_intelligence_retry_count", 0)
        try:
            si_result = self._schema_intelligence.analyze(si_inputs, domain_ctx)
            return {"si_result": si_result, "schema_intelligence_attempted": True}
        except Exception as e:
            logger.warning(
                "schema_intelligence_node_failed",
                run_id=str(state["run_id"]), retry_count=retry_count, error=str(e),
            )
            return {
                "si_result": None,
                "schema_intelligence_attempted": True,
                "schema_intelligence_retry_count": retry_count + 1,
            }

    def schema_intelligence_router(self, state: ProfilingState) -> Literal["retry", "continue"]:
        """Real conditional edge: retry once on a hard failure, then move on regardless."""
        if state.get("si_result") is None and state.get("schema_intelligence_retry_count", 0) < 1:
            return "retry"
        return "continue"

    # ── Node 4: classification + hierarchy ──────────────────────────────────

    def classify_and_build_hierarchy(self, state: ProfilingState) -> dict[str, Any]:
        df = state["df"]
        col_profiles = state["col_profiles"]
        sem_candidates = state["sem_candidates"]
        grain_result = state["grain_result"]
        primary_domain = state["primary_domain"]

        sec_domain = self._domain_classifier.classify(primary_domain, col_profiles, sem_candidates)
        classifications = self._category_classifier.classify_all(col_profiles, sem_candidates, primary_domain)
        hierarchy_result = self._hierarchy_builder.build(
            df, primary_domain, sec_domain.name, col_profiles, sem_candidates, grain_result.grain_columns,
        )

        return {
            "sec_domain": sec_domain,
            "classifications": classifications,
            "hierarchy_result": hierarchy_result,
        }

    # ── Node 5: business rules ──────────────────────────────────────────────

    def evaluate_business_rules(self, state: ProfilingState) -> dict[str, Any]:
        df = state["df"]
        primary_domain = state["primary_domain"]
        sec_domain = state["sec_domain"]
        sem_candidates = state["sem_candidates"]
        col_profiles = state["col_profiles"]
        request_rules = state.get("request_rules")
        approved_rule_dicts = state.get("approved_rule_dicts")

        rules = self._rule_loader.load_domain_rules(primary_domain, sec_domain.name)
        if request_rules:
            rules.extend(self._rule_loader.load_request_rules(request_rules, primary_domain))
        if approved_rule_dicts:
            rules.extend(self._rule_loader.load_request_rules(
                approved_rule_dicts, primary_domain, source=RuleSource.APPROVED_SUGGESTION,
            ))

        role_map = _build_role_map(sem_candidates, col_profiles)
        rule_results = self._rule_engine.evaluate(df, rules, role_map)

        return {"rules": rules, "role_map": role_map, "rule_results": rule_results}

    # ── Node 6: quality assessment ──────────────────────────────────────────

    def assess_quality(self, state: ProfilingState) -> dict[str, Any]:
        col_profiles = state["col_profiles"]
        schema_metadata = state.get("schema_metadata")
        rule_results = state["rule_results"]
        si_result = state.get("si_result")
        classifications = state["classifications"]
        sec_domain = state["sec_domain"]
        hierarchy_result = state["hierarchy_result"]

        mandatory_cols = _get_mandatory_columns(schema_metadata)
        unique_cols = _get_expected_unique_columns(schema_metadata)

        quality_dims: list[dict[str, Any]] = []
        comp = assess_completeness(col_profiles, mandatory_cols)
        quality_dims.append({
            "dimension": comp.dimension, "score": comp.score,
            "status": comp.status.value, "assessed_count": comp.assessed_count,
            "violation_count": comp.violation_count, "evidence": comp.evidence, "reason": comp.reason,
        })
        quality_dims.append(assess_uniqueness(col_profiles, unique_cols))
        quality_dims.append(assess_validity(rule_results))
        quality_dims.append(assess_conformity(rule_results))
        quality_dims.append(assess_consistency(rule_results))
        quality_dims.append(assess_business_rule_compliance(rule_results))
        quality_dims.append(assess_timeliness())
        quality_dims.append(assess_integrity())
        quality_dims.append(assess_accuracy())

        si_conf_avg = _avg_confidence(si_result.column_results) if si_result and si_result.column_results else None
        class_conf_avg = sum(c.confidence for c in classifications) / len(classifications) if classifications else None
        sec_conf = sec_domain.confidence if sec_domain.name else None
        hier_conf = hierarchy_result.average_confidence if hierarchy_result.status != HierarchyChainStatus.UNRESOLVED else None
        quality_dims.append(assess_semantic_quality(si_conf_avg, class_conf_avg, sec_conf, hier_conf))

        weights_config = self._config_repo.get_quality_weights()
        overall_quality = calculate_overall_score(quality_dims, weights_config.get("weights", {}))

        quality_gate_passed = overall_quality.get("overall_score") is None or overall_quality["overall_score"] >= QUALITY_GATE_MIN_SCORE

        return {
            "mandatory_cols": mandatory_cols,
            "unique_cols": unique_cols,
            "quality_dims": quality_dims,
            "overall_quality": overall_quality,
            "quality_gate_passed": quality_gate_passed,
        }

    def quality_gate_router(self, state: ProfilingState) -> list[str]:
        """Real conditional edge: skip suggestions/readiness/charts on very poor data.
        Returns actual node names (not abstract keys) so the "full" path can fan out
        to both independent ReAct sub-agents at once — see graph.py's docstring for
        why generate_suggestions/recommend_target_features have no edge between them."""
        if state.get("quality_gate_passed", True):
            return ["generate_suggestions", "recommend_target_features"]
        return ["finalize_lightweight"]

    # ── Node 7: rule suggestions (LLM, ReAct tool-calling) ──────────────────

    def generate_suggestions(self, state: ProfilingState) -> dict[str, Any]:
        if self._llm_provider is None:
            return {"rule_suggestions": []}

        from app.agents.rule_suggestion_agent import run_rule_suggestion_agent

        retry_count = state.get("rule_suggestion_retry_count", 0)
        if self._settings.llm_burst_cooldown_seconds > 0:
            time.sleep(self._settings.llm_burst_cooldown_seconds)

        existing_rule_keys = [r.rule_key for r in state["rules"]]
        try:
            suggestions = run_rule_suggestion_agent(
                llm_model=self._settings.azure_openai_deployment,
                api_key=self._settings.azure_openai_api_key,
                azure_endpoint=self._settings.azure_openai_endpoint,
                api_version=self._settings.azure_openai_api_version,
                # No [:30] cap -- same class of bug as feature_target_agent's
                # column-summary truncation (see that graph.py's comment):
                # a wide dataset's business-relevant columns can sit well
                # past position 30, and were structurally invisible to this
                # agent regardless of how well they'd fit a proposed rule.
                col_profiles=state["col_profiles"],
                df=state["df"],
                primary_domain=state["primary_domain"],
                secondary_domain=state["sec_domain"].name,
                existing_rule_keys=existing_rule_keys,
            )
            return {"rule_suggestions": suggestions}
        except Exception as e:
            logger.warning(
                "rule_suggestion_generation_failed",
                run_id=str(state["run_id"]), retry_count=retry_count, error=str(e),
            )
            return {"rule_suggestions": [], "rule_suggestion_retry_count": retry_count + 1}

    # ── Node 7b: target/feature/drop recommendation ─────────────────────────
    # Always produces a feature_recommendation — deterministically (no LLM
    # call) when no business_question/target_column was supplied, so every
    # run (including question-less ones) has something for assess_readiness
    # to use, and existing callers see identical readiness scores to before.

    def recommend_target_features(self, state: ProfilingState) -> dict[str, Any]:
        col_profiles = state["col_profiles"]
        sem_candidates = state["sem_candidates"]
        grain_result = state["grain_result"]
        business_question = state.get("business_question")
        target_override = state.get("target_column_override")

        if not business_question and not target_override:
            return {"feature_recommendation": _deterministic_feature_split(sem_candidates, grain_result)}

        if self._llm_provider is None:
            return {"feature_recommendation": _deterministic_feature_split(sem_candidates, grain_result)}

        # A caller-supplied override always wins. Otherwise, try to resolve
        # the target deterministically from the question text before ever
        # asking the LLM to guess -- this is the actual, reproducible source
        # of target-selection non-determinism for the common "question
        # names the column in plain English" case (e.g. "Forecast
        # underwriting result..." naming underwriting_result_actual): the
        # LLM's answer for that case shouldn't depend on chance, so don't
        # let it. Only returns a column on a single, unambiguous match --
        # anything murkier still goes to the LLM exactly as before.
        effective_override = target_override
        if not effective_override and business_question:
            effective_override = _infer_target_column_from_question(business_question, sem_candidates)

        from app.agents.feature_target_agent import run_feature_target_agent

        try:
            recommendation = run_feature_target_agent(
                llm_model=self._settings.azure_openai_deployment,
                api_key=self._settings.azure_openai_api_key,
                azure_endpoint=self._settings.azure_openai_endpoint,
                api_version=self._settings.azure_openai_api_version,
                col_profiles=col_profiles,
                sem_candidates=sem_candidates,
                grain_result=grain_result,
                df=state["df"],
                business_question=business_question,
                target_column_override=effective_override,
            )
            return {"feature_recommendation": recommendation}
        except Exception as e:
            logger.warning(
                "feature_target_recommendation_failed",
                run_id=str(state["run_id"]), error=str(e),
            )
            return {"feature_recommendation": _deterministic_feature_split(sem_candidates, grain_result)}

    # ── Node 8: readiness ────────────────────────────────────────────────────

    def assess_readiness(self, state: ProfilingState) -> dict[str, Any]:
        col_profiles = state["col_profiles"]
        sem_candidates = state["sem_candidates"]
        quality_dims = state["quality_dims"]
        grain_result = state["grain_result"]
        schema_metadata = state.get("schema_metadata")
        ds_profile = state["dataset_profile_obj"]
        feature_recommendation = state.get("feature_recommendation")

        has_temporal = any(c.candidate_column_role == ColumnRole.TEMPORAL_DIMENSION for c in sem_candidates)
        metric_count = sum(1 for c in sem_candidates if c.candidate_column_role == ColumnRole.METRIC)
        dim_count = sum(1 for c in sem_candidates if c.candidate_column_role == ColumnRole.DIMENSION)
        desc_coverage = _calc_description_coverage(col_profiles, schema_metadata)

        readiness_results = self._readiness_engine.assess_all(
            col_profiles, quality_dims, grain_result.grain_columns,
            has_temporal, metric_count, dim_count, desc_coverage, ds_profile.row_count,
            feature_recommendation=feature_recommendation,
        )
        return {"readiness_assessments": [r.to_dict() for r in readiness_results]}

    # ── Node 9: charts + drill-down ──────────────────────────────────────────

    def generate_charts(self, state: ProfilingState) -> dict[str, Any]:
        df = state["df"]
        col_profiles = state["col_profiles"]
        sem_candidates = state["sem_candidates"]
        quality_dims = state["quality_dims"]
        hierarchy_levels = state["hierarchy_result"].level_columns
        primary_domain = state["primary_domain"]

        domain_config = self._config_repo.get_domain_config(primary_domain)
        chart_templates = domain_config.get("chart_templates", [])
        charts = self._chart_generator.generate(
            col_profiles, sem_candidates, chart_templates, quality_dims, hierarchy_levels,
        )
        charts = self._agg_engine.aggregate_all(df, charts)

        drill_down_cubes = _generate_drill_down_cubes(
            self._settings, df, charts, hierarchy_levels, sem_candidates, col_profiles,
        )

        return {
            "charts": [c.to_dict() for c in charts],
            "drill_down_cubes": drill_down_cubes,
        }

    # ── Node 10a: finalize (full path) ──────────────────────────────────────

    def finalize(self, state: ProfilingState) -> dict[str, Any]:
        col_profiles = state["col_profiles"]
        refined_types = state["refined_types"]
        sem_candidates = state["sem_candidates"]
        si_result = state.get("si_result")
        grain_result = state["grain_result"]
        schema_metadata = state.get("schema_metadata")

        return {
            "column_output": _build_column_output(
                col_profiles, refined_types, sem_candidates, si_result, grain_result, schema_metadata,
            ),
            "status": RunStatus.COMPLETED.value,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Node 10b: finalize (lightweight path — poor quality gate) ──────────

    def finalize_lightweight(self, state: ProfilingState) -> dict[str, Any]:
        col_profiles = state["col_profiles"]
        refined_types = state["refined_types"]
        sem_candidates = state["sem_candidates"]
        si_result = state.get("si_result")
        grain_result = state["grain_result"]
        schema_metadata = state.get("schema_metadata")

        return {
            "column_output": _build_column_output(
                col_profiles, refined_types, sem_candidates, si_result, grain_result, schema_metadata,
            ),
            "feature_recommendation": _deterministic_feature_split(sem_candidates, grain_result),
            "rule_suggestions": [],
            "readiness_assessments": [],
            "charts": [],
            "drill_down_cubes": [],
            "status": RunStatus.COMPLETED.value,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "warnings": ["Quality gate below threshold — readiness, charts, and rule suggestions were skipped."],
        }


# ── Module-level helpers (pure functions, ported from ProfilingOrchestrator) ──

def _get_description(col_name: str, metadata: dict[str, Any] | None) -> str | None:
    if not metadata or "columns" not in metadata:
        return None
    for col in metadata["columns"]:
        if col.get("column_name") == col_name:
            return col.get("description")
    return None


def _build_column_output(profiles, refined_types, candidates, si_result, grain_result, metadata):
    output = []
    si_map = {r.column_name: r for r in si_result.column_results} if si_result else {}
    id_map = {c.column_name: c for c in grain_result.identifier_candidates}
    for i, p in enumerate(profiles):
        si = si_map.get(p.physical_name)
        id_c = id_map.get(p.physical_name)
        output.append({
            "physical_name": p.physical_name,
            "normalized_key": p.normalized_key,
            "pandas_dtype": p.pandas_dtype,
            "refined_data_type": refined_types[i].value,
            "statistics": p.to_statistics_dict(),
            "candidate_semantic_type": candidates[i].candidate_semantic_type,
            "candidate_column_role": candidates[i].candidate_column_role.value,
            "candidate_confidence": candidates[i].candidate_confidence,
            "confirmed_semantic_type": si.confirmed_semantic_type if si else None,
            "confirmed_column_role": si.confirmed_column_role if si else None,
            "schema_intelligence_decision": si.decision.value if si else None,
            "schema_confidence": si.confidence if si else None,
            "identifier_score": id_c.identifier_score if id_c else None,
            "is_grain_key": p.physical_name in grain_result.grain_columns,
        })
    return output


_FEATURE_ROLES = {ColumnRole.METRIC, ColumnRole.DIMENSION, ColumnRole.TEMPORAL_DIMENSION, ColumnRole.FLAG}

# Qualifier suffixes stripped from a metric column's normalized name before
# phrase-matching against the question -- same pattern Analytics-Agent's
# BusinessQuestionInterpreter._base_phrase already uses successfully for
# the identical problem (a column like "net_profit_actual" should match a
# question mentioning "net profit" even though the column itself also
# carries an "actual"/"budget"/etc. qualifier the question doesn't repeat).
_TARGET_QUALIFIER_SUFFIX_RE = re.compile(
    r"\b(actual|budget|forecast|prior[_ ]year|ytd|ratio|rate|amount|pct|percent)\b", re.IGNORECASE,
)
_MIN_TARGET_PHRASE_LEN = 3


def _infer_target_column_from_question(business_question: str, sem_candidates) -> str | None:
    """Deterministic, zero-LLM-cost target resolution for the common case
    where the question names the target in plain English (e.g. "Forecast
    underwriting result for next 6 months" naming
    underwriting_result_actual). Only returns a column when there's a
    single, unambiguous longest match -- anything murkier (no match, or a
    tie) is left to the LLM exactly as before, never guessed."""
    question_lower = business_question.lower()
    candidates: list[tuple[int, str]] = []
    for c in sem_candidates:
        if c.candidate_column_role != ColumnRole.METRIC:
            continue
        normalized = c.column_name.replace("_", " ").lower()
        phrase = _TARGET_QUALIFIER_SUFFIX_RE.sub("", normalized)
        phrase = re.sub(r"\s+", " ", phrase).strip()
        if len(phrase) >= _MIN_TARGET_PHRASE_LEN and phrase in question_lower:
            candidates.append((len(phrase), c.column_name))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None  # ambiguous tie -- don't guess
    return candidates[0][1]


def _deterministic_feature_split(sem_candidates, grain_result) -> dict[str, Any]:
    """No-LLM feature/drop split — used when no business_question/target_column
    was supplied, and as the fallback if the LLM call fails. Every non-identifier
    column with a usable semantic role is a candidate feature; grain columns are
    dropped. No target/problem_type/approach can be determined without a question."""
    grain_columns = set(grain_result.grain_columns)

    feature_columns = [
        {
            "column": c.column_name,
            "role": c.candidate_column_role.value,
            # "high" because these are genuinely-typed metric/dimension/temporal/flag
            # columns (a more precise filter than the old "every non-identifier
            # column" heuristic, not a weaker one) — "medium" would understate
            # ml_readiness's feature-coverage score for every question-less run,
            # which is the common case.
            "usefulness": "high",
            "suggested_aggregation": None,
            "reasoning": "Included by default — no business question was supplied to prioritize features.",
        }
        for c in sem_candidates
        if c.candidate_column_role in _FEATURE_ROLES and c.column_name not in grain_columns
    ]
    drop_columns = [
        {"column": name, "reason": "Identifier / grain key — no predictive value."}
        for name in grain_result.grain_columns
    ]

    return {
        "target_column": None,
        "problem_type": "unknown",
        "time_column": None,
        "recommended_approach": "unknown",
        "approach_reasoning": None,
        "feature_columns": feature_columns,
        "drop_columns": drop_columns,
        "confidence": 0.0,
    }


def _build_role_map(candidates, profiles) -> dict[str, str]:
    role_map: dict[str, str] = {}
    for i, c in enumerate(candidates):
        if c.candidate_semantic_type:
            role_map[c.candidate_semantic_type] = profiles[i].physical_name
    return role_map


def _get_mandatory_columns(metadata) -> list[str]:
    if metadata and "columns" in metadata:
        return [c["column_name"] for c in metadata["columns"] if c.get("mandatory")]
    return []


def _get_expected_unique_columns(metadata) -> list[str]:
    if metadata and "columns" in metadata:
        return [c["column_name"] for c in metadata["columns"] if c.get("expected_unique")]
    return []


def _calc_description_coverage(profiles, metadata) -> float:
    if not metadata or "columns" not in metadata:
        return 0.0
    described = sum(1 for c in metadata["columns"] if c.get("description"))
    return described / max(len(profiles), 1)


def _avg_confidence(results) -> float:
    if not results:
        return 0.0
    return sum(r.confidence for r in results) / len(results)


def _generate_drill_down_cubes(settings, df, charts, hierarchy_levels, sem_candidates, col_profiles) -> list[dict[str, Any]]:
    if not hierarchy_levels or len(hierarchy_levels) < 2:
        return []

    min_group = settings.min_cube_group_size
    cubes: list[dict[str, Any]] = []

    metric_cols = [
        col_profiles[i].physical_name
        for i, c in enumerate(sem_candidates)
        if c.candidate_column_role == ColumnRole.METRIC and col_profiles[i].physical_name in df.columns
    ]
    if not metric_cols:
        return []

    primary_metric = metric_cols[0]
    numeric_metric = pd.to_numeric(df[primary_metric], errors="coerce")

    for depth in range(len(hierarchy_levels)):
        level_col = hierarchy_levels[depth]
        if level_col not in df.columns:
            continue

        path_cols = hierarchy_levels[:depth]

        if not path_cols:
            grouped = df.assign(_metric=numeric_metric).groupby(level_col)["_metric"]
            agg_data = []
            for label, group_metric in grouped:
                count = int(group_metric.count())
                if count < min_group:
                    continue
                agg_data.append({
                    "label": str(label),
                    "value": round(float(group_metric.sum()), 2) if not group_metric.isna().all() else 0,
                    "count": count,
                })
            if agg_data:
                cubes.append({
                    "level_column": level_col, "level_depth": depth, "dimension_path_json": {},
                    "aggregated_data_json": agg_data[:100], "metric_column": primary_metric,
                    "aggregation": "sum", "record_count": len(df),
                })
        else:
            valid_path_cols = [c for c in path_cols if c in df.columns]
            if not valid_path_cols:
                continue
            parent_combos = df[valid_path_cols].drop_duplicates().head(50)
            for _, parent_row in parent_combos.iterrows():
                mask = pd.Series(True, index=df.index)
                path_dict: dict[str, str] = {}
                for pc in valid_path_cols:
                    mask &= df[pc] == parent_row[pc]
                    path_dict[pc] = str(parent_row[pc])
                subset = df[mask]
                if len(subset) < min_group:
                    continue
                sub_metric = pd.to_numeric(subset[primary_metric], errors="coerce")
                grouped = subset.assign(_m=sub_metric).groupby(level_col)["_m"]
                agg_data = []
                for label, group_metric in grouped:
                    count = int(group_metric.count())
                    if count < min_group:
                        continue
                    agg_data.append({
                        "label": str(label),
                        "value": round(float(group_metric.sum()), 2) if not group_metric.isna().all() else 0,
                        "count": count,
                    })
                if agg_data:
                    cubes.append({
                        "level_column": level_col, "level_depth": depth, "dimension_path_json": path_dict,
                        "aggregated_data_json": agg_data[:100], "metric_column": primary_metric,
                        "aggregation": "sum", "record_count": len(subset),
                    })

    return cubes
