"""app/services/question_interpreter/business_question_interpreter.py —
Stage 3 of the Agent 3 redesign (plan "zany-giggling-crayon").

Narrows, never replaces, never widens the Planner's (Stage 4) candidate
analysis types to what the business_question actually asks about —
"Forecast revenue" should mean forecasting/trend/anomaly_detection, not
also clustering/association_rules/ranking/segmentation just because
they're structurally possible.

Primary mechanism is the same deterministic keyword vocabulary Agent 3
already has today (`nodes/pipeline.py`'s `_INTENT_MAP` — "forecast",
"trend", "why", "anomaly", "segment" are already domain-neutral English,
not Insurance-specific; only `_KPI_ALIASES`/`_DIMENSION_KEYWORDS` were
Insurance-specific and those move into the Insurance plugin separately),
generalized to map onto a *set* of analysis types instead of one fixed
handler. Zero LLM cost, deterministic, fast — matches "keeps execution
fast" from the plan. An LLM-assisted fallback (same ReAct shape as Agent
2's feature_target_agent) is the extension point for questions no keyword
matches; not implemented in Phase 1 (offline, no live LLM wiring here
yet) — returns None, same as "no business_question at all," so the
Planner runs unrestricted rather than guessing.
"""

from __future__ import annotations

import re

from app.services.capability_resolution.models import CapabilityProfile
from app.services.dataset_context.models import (
    SEMANTIC_ROLE_DIMENSION, SEMANTIC_ROLE_METRIC, DatasetContext,
)
from app.services.kpi_discovery.models import DiscoveredKPI
from app.services.question_interpreter.models import QuestionIntent

# Qualifier suffixes stripped from a column's normalized name before
# phrase-matching against the question — "net_profit_actual" should match
# a question mentioning "net profit" even though the column itself also
# carries an "actual"/"budget"/"prior_year" qualifier the question doesn't
# repeat verbatim.
_QUALIFIER_SUFFIX_RE = re.compile(
    r"\b(actual|budget|forecast|prior[_ ]year|ytd|ratio|rate|amount|pct|percent)\b", re.IGNORECASE,
)
_MIN_PREFERRED_PHRASE_LEN = 3


def _base_phrase(column_name: str) -> str:
    normalized = column_name.replace("_", " ").lower()
    stripped = _QUALIFIER_SUFFIX_RE.sub("", normalized)
    return re.sub(r"\s+", " ", stripped).strip()

# keyword group -> candidate analysis types it implies. Order doesn't
# matter — every group is checked, matches are unioned. Deliberately not
# exhaustive: an unmapped phrasing just means _interpret_via_keywords
# returns nothing for that word, not an error, and the overall result
# falls through to the LLM-fallback stub (-> None -> unrestricted planning).
_KEYWORD_ANALYSIS_TYPES: list[tuple[list[str], set[str]]] = [
    (["forecast", "predict", "projection", "next quarter", "future", "expected"],
     {"forecast", "trend", "time_series_analysis", "anomaly_detection"}),
    (["anomaly", "anomalies", "irregular", "outlier", "unusual", "spike", "detect"],
     {"anomaly_detection", "outlier_detection"}),
    (["why", "reason", "cause", "driver", "drove", "driven", "factors", "contributing", "explain why"],
     {"root_cause", "feature_importance"}),
    (["variance", "vs budget", "vs prior", "vs plan", "deviation", "gap", "difference"],
     {"comparative_analysis", "root_cause"}),
    (["trend", "over time", "qoq", "yoy", "mom", "quarter over quarter", "growth"],
     {"trend", "time_series_analysis", "forecast"}),
    (["segment", "cluster", "group by risk", "risk tier", "portfolio risk"],
     {"clustering", "segmentation", "comparative_analysis", "ranking"}),
    (["correlat", "relationship between", "related to"],
     {"correlation"}),
    (["rank", "top ", "bottom ", "highest", "lowest"],
     {"ranking"}),
    (["distribution", "spread", "breakdown by"],
     {"distribution_analysis"}),
    (["associat", "co-occur", "pattern between"],
     {"association_rules"}),
    (["classify", "classification", "category prediction"],
     {"classification", "feature_importance"}),
    (["compar", "versus", " vs "],
     {"comparative_analysis"}),
]


class BusinessQuestionInterpreter:
    def interpret(
        self,
        business_question: str | None,
        capability_profile: CapabilityProfile,
        discovered_kpis: list[DiscoveredKPI],
        dataset_context: DatasetContext | None = None,
    ) -> QuestionIntent | None:
        if not business_question:
            return None

        question_lower = business_question.lower()
        matched_types: set[str] = set()
        matched_group_labels: list[str] = []
        for keywords, analysis_types in _KEYWORD_ANALYSIS_TYPES:
            if any(kw in question_lower for kw in keywords):
                matched_types |= analysis_types
                matched_group_labels.append(keywords[0])

        if not matched_types:
            return self._llm_fallback(business_question, capability_profile, discovered_kpis)

        # Intersect with structural possibility — can narrow within what's
        # possible, never widen past it. An analysis type with no
        # capability gate (root_cause, trend, ...) always passes through.
        final_types = {t for t in matched_types if capability_profile.is_analysis_type_possible(t)}
        if not final_types:
            # Every keyword-matched type turned out structurally
            # impossible — safer to not narrow at all than to hand the
            # Planner an empty candidate set.
            return None

        mentioned_kpis = [kpi.name for kpi in discovered_kpis if kpi.name.lower() in question_lower]
        preferred_metrics = self._match_preferred_columns(question_lower, dataset_context, SEMANTIC_ROLE_METRIC)
        preferred_dimensions = self._match_preferred_columns(question_lower, dataset_context, SEMANTIC_ROLE_DIMENSION)

        return QuestionIntent(
            candidate_analysis_types=final_types,
            mentioned_kpis=mentioned_kpis,
            rationale=f"Matched keyword group(s): {', '.join(matched_group_labels)}",
            preferred_metrics=preferred_metrics,
            preferred_dimensions=preferred_dimensions,
        )

    @staticmethod
    def _match_preferred_columns(
        question_lower: str, dataset_context: DatasetContext | None, role: str,
    ) -> list[str]:
        """Which columns of this role does the question actually name?
        Ranked by matched-phrase length (longest/most specific wins),
        original column order breaks ties — deterministic, no LLM cost on
        the common case (see module docstring)."""
        if dataset_context is None:
            return []
        candidates: list[tuple[int, str]] = []
        for column in dataset_context.columns_with_role(role):
            phrase = _base_phrase(column.name)
            if len(phrase) >= _MIN_PREFERRED_PHRASE_LEN and phrase in question_lower:
                candidates.append((len(phrase), column.name))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return [name for _length, name in candidates]

    def _llm_fallback(
        self,
        business_question: str,
        capability_profile: CapabilityProfile,
        discovered_kpis: list[DiscoveredKPI],
    ) -> QuestionIntent | None:
        """Extension point for a later phase: an LLM-assisted
        interpretation (same ReAct shape as Agent 2's feature_target_agent)
        for questions no keyword group matches. Not implemented in Phase 1
        — this stage is offline/pure, not yet wired to a live LLM call —
        returns None so the Planner runs unrestricted rather than
        guessing."""
        return None
