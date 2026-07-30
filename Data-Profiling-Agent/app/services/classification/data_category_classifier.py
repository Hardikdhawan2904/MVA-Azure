"""Column data-category classification — assigns primary and secondary categories."""

import re
from typing import Any

from app.core.enums import DataCategory, ColumnRole, RefinedDataType
from app.core.logging import get_logger
from app.services.profiling.column_profiler import ColumnProfileResult
from app.services.profiling.semantic_candidate_generator import SemanticCandidate

logger = get_logger(__name__)


class ColumnCategoryResult:
    """Classification result for a single column."""

    def __init__(
        self,
        column_name: str,
        primary_category: DataCategory,
        secondary_categories: list[DataCategory],
        confidence: float,
        evidence: list[dict[str, Any]],
    ):
        self.column_name = column_name
        self.primary_category = primary_category
        self.secondary_categories = secondary_categories
        self.confidence = confidence
        self.evidence = evidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "primary_category": self.primary_category.value,
            "secondary_categories": [c.value for c in self.secondary_categories],
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
        }


# Deterministic classification rules mapping semantic types/roles/names to categories
_TRANSACTION_KEYWORDS = re.compile(
    r"(transaction|txn|payment|purchase|order|invoice|transfer|settlement|clearing)",
    re.IGNORECASE,
)
_FINANCIAL_KEYWORDS = re.compile(
    r"(amount|amt|price|revenue|cost|balance|fee|charge|salary|income|expense|"
    r"profit|budget|forecast|actual|variance|margin)",
    re.IGNORECASE,
)
_DEMOGRAPHIC_KEYWORDS = re.compile(
    r"(age|gender|sex|birth|dob|ethnicity|race|nationality|marital|education)",
    re.IGNORECASE,
)
_BEHAVIORAL_KEYWORDS = re.compile(
    r"(click|visit|session|page_view|engagement|frequency|recency|churn|retention)",
    re.IGNORECASE,
)
_OPERATIONAL_KEYWORDS = re.compile(
    r"(status|state|stage|queue|batch|process|workflow|pipeline|system|error|retry)",
    re.IGNORECASE,
)
_INTERACTION_KEYWORDS = re.compile(
    r"(call|email|chat|ticket|support|feedback|complaint|survey|nps|csat|contact)",
    re.IGNORECASE,
)
_MASTER_KEYWORDS = re.compile(
    r"(customer_id|employee_id|product_id|account_id|merchant_id|vendor_id|"
    r"name|title|department|position|segment|tier|category|code)",
    re.IGNORECASE,
)
_GEOGRAPHIC_KEYWORDS = re.compile(
    r"(country|region|state|city|zip|postal|address|location|lat|lon|geo|branch|office)",
    re.IGNORECASE,
)
_RISK_KEYWORDS = re.compile(
    r"(risk|fraud|score|alert|suspicious|flag|violation|compliance|penalty)",
    re.IGNORECASE,
)
_CUSTOMER_EXP_KEYWORDS = re.compile(
    r"(satisfaction|nps|csat|rating|review|loyalty|points|reward|tier|membership)",
    re.IGNORECASE,
)


# Default weights (used when no config is provided)
_DEFAULT_WEIGHTS = {
    "name_keyword_match": 0.4,
    "semantic_type_match": 0.5,
    "column_role_match": 0.3,
    "domain_context_boost": 0.2,
    "domain_context_boost_weak": 0.1,
    "datetime_parse_match": 0.4,
}

_DEFAULT_SELECTION = {
    "secondary_min_ratio_of_primary": 0.5,
    "secondary_min_absolute": 0.3,
    "max_secondary_categories": 3,
    "confidence_gap_threshold": 0.3,
    "confidence_gap_boost": 0.1,
}


class DataCategoryClassifier:
    """Classifies columns into data categories using configurable weights."""

    def __init__(self, weights: dict[str, Any] | None = None):
        """
        Initialize with optional weights from configuration.

        Args:
            weights: Dict from classification_weights.yaml with keys
                     'signal_weights' and 'selection'. If None, uses defaults.
        """
        if weights:
            self._signal_weights = weights.get("signal_weights", _DEFAULT_WEIGHTS)
            self._selection = weights.get("selection", _DEFAULT_SELECTION)
        else:
            self._signal_weights = _DEFAULT_WEIGHTS
            self._selection = _DEFAULT_SELECTION

    @property
    def _w_name(self) -> float:
        return self._signal_weights.get("name_keyword_match", 0.4)

    @property
    def _w_semantic(self) -> float:
        return self._signal_weights.get("semantic_type_match", 0.5)

    @property
    def _w_role(self) -> float:
        return self._signal_weights.get("column_role_match", 0.3)

    @property
    def _w_domain(self) -> float:
        return self._signal_weights.get("domain_context_boost", 0.2)

    @property
    def _w_domain_weak(self) -> float:
        return self._signal_weights.get("domain_context_boost_weak", 0.1)

    @property
    def _w_datetime(self) -> float:
        return self._signal_weights.get("datetime_parse_match", 0.4)

    def classify(
        self,
        profile: ColumnProfileResult,
        semantic_candidate: SemanticCandidate,
        primary_domain: str,
    ) -> ColumnCategoryResult:
        """
        Classify a single column into primary + secondary categories.

        Uses configurable weights loaded from YAML.
        """
        scores: dict[DataCategory, float] = {}
        evidence: list[dict[str, Any]] = []
        name = profile.normalized_key

        # Score each category
        self._score_by_name(name, scores, evidence)
        self._score_by_semantic_type(semantic_candidate, scores, evidence)
        self._score_by_role(semantic_candidate, scores, evidence)
        self._score_by_domain(primary_domain, semantic_candidate, scores, evidence)
        self._score_temporal(profile, semantic_candidate, scores, evidence)

        if not scores:
            default = self._default_category(semantic_candidate)
            return ColumnCategoryResult(
                column_name=profile.physical_name,
                primary_category=default,
                secondary_categories=[],
                confidence=0.50,
                evidence=[{"type": "default", "value": "no_strong_signals"}],
            )

        # Select primary (highest score)
        sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_categories[0][0]
        primary_score = sorted_categories[0][1]

        # Select secondary using configurable thresholds
        min_ratio = self._selection.get("secondary_min_ratio_of_primary", 0.5)
        min_abs = self._selection.get("secondary_min_absolute", 0.3)
        max_secondary = self._selection.get("max_secondary_categories", 3)

        secondary: list[DataCategory] = []
        for cat, score in sorted_categories[1:]:
            if score >= primary_score * min_ratio and score >= min_abs:
                secondary.append(cat)

        # Confidence based on primary score and separation from second
        confidence = min(1.0, primary_score)
        gap_threshold = self._selection.get("confidence_gap_threshold", 0.3)
        gap_boost = self._selection.get("confidence_gap_boost", 0.1)
        if len(sorted_categories) > 1:
            gap = primary_score - sorted_categories[1][1]
            if gap > gap_threshold:
                confidence = min(1.0, confidence + gap_boost)

        return ColumnCategoryResult(
            column_name=profile.physical_name,
            primary_category=primary,
            secondary_categories=secondary[:max_secondary],
            confidence=round(confidence, 3),
            evidence=evidence,
        )

    def classify_all(
        self,
        profiles: list[ColumnProfileResult],
        semantic_candidates: list[SemanticCandidate],
        primary_domain: str,
    ) -> list[ColumnCategoryResult]:
        """Classify all columns."""
        results = []
        for i, profile in enumerate(profiles):
            candidate = semantic_candidates[i] if i < len(semantic_candidates) else None
            if candidate is None:
                candidate = SemanticCandidate(
                    column_name=profile.physical_name,
                    normalized_key=profile.normalized_key,
                    refined_type=RefinedDataType.UNKNOWN,
                    candidate_semantic_type=None,
                    candidate_column_role=ColumnRole.UNKNOWN,
                    candidate_confidence=0.0,
                    evidence=[],
                )
            results.append(self.classify(profile, candidate, primary_domain))
        return results

    def _score_by_name(
        self, name: str, scores: dict[DataCategory, float], evidence: list[dict[str, Any]]
    ) -> None:
        """Score categories based on column name patterns. Weight from config."""
        w = self._w_name
        checks = [
            (_TRANSACTION_KEYWORDS, DataCategory.TRANSACTION, "transaction_keyword"),
            (_FINANCIAL_KEYWORDS, DataCategory.FINANCIAL, "financial_keyword"),
            (_DEMOGRAPHIC_KEYWORDS, DataCategory.DEMOGRAPHIC, "demographic_keyword"),
            (_BEHAVIORAL_KEYWORDS, DataCategory.BEHAVIORAL, "behavioral_keyword"),
            (_OPERATIONAL_KEYWORDS, DataCategory.OPERATIONAL, "operational_keyword"),
            (_INTERACTION_KEYWORDS, DataCategory.INTERACTION, "interaction_keyword"),
            (_MASTER_KEYWORDS, DataCategory.MASTER, "master_keyword"),
            (_GEOGRAPHIC_KEYWORDS, DataCategory.GEOGRAPHIC, "geographic_keyword"),
            (_RISK_KEYWORDS, DataCategory.RISK, "risk_keyword"),
            (_CUSTOMER_EXP_KEYWORDS, DataCategory.CUSTOMER_EXPERIENCE, "customer_exp_keyword"),
        ]

        for pattern, category, signal in checks:
            if pattern.search(name):
                scores[category] = scores.get(category, 0) + w
                evidence.append({"type": "name_match", "value": signal, "column": name})

    def _score_by_semantic_type(
        self,
        candidate: SemanticCandidate,
        scores: dict[DataCategory, float],
        evidence: list[dict[str, Any]],
    ) -> None:
        """Score based on semantic type. Weight from config."""
        sem_type = (candidate.candidate_semantic_type or "").lower()
        if not sem_type:
            return

        w = self._w_semantic
        mappings: list[tuple[list[str], DataCategory]] = [
            (["monetary_amount", "transaction_amount", "payment"], DataCategory.TRANSACTION),
            (["revenue", "expense", "profit", "margin", "budget", "forecast"], DataCategory.FINANCIAL),
            (["country", "region", "city", "geographic", "state_province"], DataCategory.GEOGRAPHIC),
            (["fraud", "risk", "risk_indicator"], DataCategory.RISK),
            (["satisfaction", "nps", "loyalty", "rating"], DataCategory.CUSTOMER_EXPERIENCE),
            (["identifier", "customer_identifier", "employee_identifier"], DataCategory.MASTER),
        ]

        for keywords, category in mappings:
            if any(k in sem_type for k in keywords):
                scores[category] = scores.get(category, 0) + w
                evidence.append({"type": "semantic_type", "value": sem_type})
                break

    def _score_by_role(
        self,
        candidate: SemanticCandidate,
        scores: dict[DataCategory, float],
        evidence: list[dict[str, Any]],
    ) -> None:
        """Score based on column role. Weight from config."""
        w = self._w_role
        role = candidate.candidate_column_role
        if role == ColumnRole.TEMPORAL_DIMENSION:
            scores[DataCategory.TIME_SERIES] = scores.get(DataCategory.TIME_SERIES, 0) + w
        elif role == ColumnRole.IDENTIFIER:
            scores[DataCategory.MASTER] = scores.get(DataCategory.MASTER, 0) + w

    def _score_by_domain(
        self,
        primary_domain: str,
        candidate: SemanticCandidate,
        scores: dict[DataCategory, float],
        evidence: list[dict[str, Any]],
    ) -> None:
        """Boost scores based on primary domain context. Weights from config."""
        domain_lower = primary_domain.lower()
        w_strong = self._w_domain
        w_weak = self._w_domain_weak

        if domain_lower == "payments":
            scores[DataCategory.TRANSACTION] = scores.get(DataCategory.TRANSACTION, 0) + w_strong
        elif domain_lower == "finance":
            scores[DataCategory.FINANCIAL] = scores.get(DataCategory.FINANCIAL, 0) + w_strong
        elif domain_lower == "customer":
            scores[DataCategory.CUSTOMER_EXPERIENCE] = scores.get(DataCategory.CUSTOMER_EXPERIENCE, 0) + w_weak
        elif domain_lower == "hr":
            scores[DataCategory.MASTER] = scores.get(DataCategory.MASTER, 0) + w_weak

    def _score_temporal(
        self,
        profile: ColumnProfileResult,
        candidate: SemanticCandidate,
        scores: dict[DataCategory, float],
        evidence: list[dict[str, Any]],
    ) -> None:
        """Score for time-series data. Weight from config."""
        if profile.datetime_parse_ratio >= 0.9:
            w = self._w_datetime
            scores[DataCategory.TIME_SERIES] = scores.get(DataCategory.TIME_SERIES, 0) + w
            evidence.append({"type": "datetime_ratio", "value": profile.datetime_parse_ratio})

    def _default_category(self, candidate: SemanticCandidate) -> DataCategory:
        """Provide a default category when no strong signals exist."""
        if candidate.candidate_column_role == ColumnRole.METRIC:
            return DataCategory.OPERATIONAL
        if candidate.candidate_column_role == ColumnRole.TEMPORAL_DIMENSION:
            return DataCategory.TIME_SERIES
        if candidate.candidate_column_role == ColumnRole.IDENTIFIER:
            return DataCategory.MASTER
        return DataCategory.OPERATIONAL
