"""Deterministic semantic candidate generation — before Schema Intelligence."""

import re
from typing import Any

from app.core.enums import RefinedDataType, ColumnRole
from app.services.profiling.column_profiler import ColumnProfileResult


# Name patterns for metric detection
_METRIC_PATTERNS = re.compile(
    r"(amount|amt|price|revenue|cost|total|balance|sum|fee|charge|"
    r"salary|compensation|pay|income|expense|profit|loss|value|"
    r"quantity|qty|count|volume|weight|score|rate|ratio|tenure|"
    r"premium|claim)",
    re.IGNORECASE,
)

# Name patterns for temporal detection
_TEMPORAL_PATTERNS = re.compile(
    r"(date|time|timestamp|datetime|created|updated|modified|"
    r"_at$|_dt$|_date$|_time$|start|end|due|expire|birth|hire|"
    r"effective|posted|processed|settled|authorized|cleared)",
    re.IGNORECASE,
)

# Name patterns for status/flag dimensions
_STATUS_PATTERNS = re.compile(
    r"(status|state|flag|type|category|class|kind|level|tier|"
    r"grade|priority|severity|outcome|result|decision|stage|"
    r"segment|channel|department|dept)",
    re.IGNORECASE,
)

# Name patterns for geographic
_GEO_PATTERNS = re.compile(
    r"(country|region|state|city|zip|postal|address|location|"
    r"latitude|longitude|lat|lng|lon|geo|province|county|district)",
    re.IGNORECASE,
)


class SemanticCandidate:
    """Result of deterministic semantic analysis for one column."""

    def __init__(
        self,
        column_name: str,
        normalized_key: str,
        refined_type: RefinedDataType,
        candidate_semantic_type: str | None,
        candidate_column_role: ColumnRole,
        candidate_confidence: float,
        evidence: list[dict[str, Any]],
    ):
        self.column_name = column_name
        self.normalized_key = normalized_key
        self.refined_type = refined_type
        self.candidate_semantic_type = candidate_semantic_type
        self.candidate_column_role = candidate_column_role
        self.candidate_confidence = candidate_confidence
        self.evidence = evidence


class SemanticCandidateGenerator:
    """Generates deterministic semantic type and role candidates."""

    def generate(
        self,
        profile: ColumnProfileResult,
        refined_type: RefinedDataType,
        is_identifier: bool,
    ) -> SemanticCandidate:
        """
        Generate semantic candidate for a single column.

        Uses:
        - Refined data type
        - Column name patterns
        - Statistics (cardinality, value range)
        - Detected patterns
        - Identifier status
        """
        # Identifiers
        if is_identifier:
            name = profile.normalized_key.lower()
            semantic_type = "identifier"
            if "transaction" in name or "txn" in name:
                semantic_type = "transaction_identifier"
            elif "employee" in name or "emp" in name:
                semantic_type = "employee_identifier"
            elif "interaction" in name:
                semantic_type = "interaction_identifier"
            elif "policy" in name:
                semantic_type = "policy_identifier"
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type=semantic_type,
                candidate_column_role=ColumnRole.IDENTIFIER,
                candidate_confidence=0.90,
                evidence=[{"type": "identifier_detection", "value": "grain_key"}],
            )

        # Route based on refined type
        if refined_type in (RefinedDataType.DATE, RefinedDataType.DATETIME):
            return self._temporal_candidate(profile, refined_type)

        if refined_type == RefinedDataType.BOOLEAN:
            return self._flag_candidate(profile, refined_type)

        if refined_type == RefinedDataType.EMAIL:
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type="email_address",
                candidate_column_role=ColumnRole.DIMENSION,
                candidate_confidence=0.92,
                evidence=[{"type": "refined_type", "value": "email"}],
            )

        if refined_type == RefinedDataType.PHONE:
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type="phone_number",
                candidate_column_role=ColumnRole.DIMENSION,
                candidate_confidence=0.88,
                evidence=[{"type": "refined_type", "value": "phone"}],
            )

        if refined_type == RefinedDataType.CURRENCY_CODE:
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type="currency_code",
                candidate_column_role=ColumnRole.DIMENSION,
                candidate_confidence=0.93,
                evidence=[{"type": "refined_type", "value": "currency_code"}],
            )

        if refined_type == RefinedDataType.COUNTRY_CODE:
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type="country_code",
                candidate_column_role=ColumnRole.DIMENSION,
                candidate_confidence=0.91,
                evidence=[{"type": "refined_type", "value": "country_code"}],
            )

        if refined_type == RefinedDataType.PERCENTAGE:
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type="percentage",
                candidate_column_role=ColumnRole.METRIC,
                candidate_confidence=0.85,
                evidence=[{"type": "refined_type", "value": "percentage"}],
            )

        # Numeric columns — check if metric or dimension
        if refined_type in (RefinedDataType.INTEGER, RefinedDataType.DECIMAL):
            return self._numeric_candidate(profile, refined_type)

        # Categorical
        if refined_type == RefinedDataType.CATEGORICAL:
            return self._categorical_candidate(profile, refined_type)

        # Text
        if refined_type == RefinedDataType.TEXT:
            return self._text_candidate(profile, refined_type)

        # Fallback
        return SemanticCandidate(
            column_name=profile.physical_name,
            normalized_key=profile.normalized_key,
            refined_type=refined_type,
            candidate_semantic_type=None,
            candidate_column_role=ColumnRole.UNKNOWN,
            candidate_confidence=0.30,
            evidence=[{"type": "fallback", "value": "no_deterministic_match"}],
        )

    def generate_all(
        self,
        profiles: list[ColumnProfileResult],
        refined_types: list[RefinedDataType],
        identifier_flags: list[bool],
    ) -> list[SemanticCandidate]:
        """Generate semantic candidates for all columns."""
        results: list[SemanticCandidate] = []
        for i, profile in enumerate(profiles):
            rt = refined_types[i] if i < len(refined_types) else RefinedDataType.UNKNOWN
            is_id = identifier_flags[i] if i < len(identifier_flags) else False
            results.append(self.generate(profile, rt, is_id))
        return results

    def _temporal_candidate(
        self, profile: ColumnProfileResult, refined_type: RefinedDataType
    ) -> SemanticCandidate:
        """Generate candidate for temporal columns."""
        semantic_type = "date" if refined_type == RefinedDataType.DATE else "datetime"

        # Check if it's a specific kind of date from name
        name = profile.normalized_key.lower()
        if any(h in name for h in ("created", "create", "creation")):
            semantic_type = "creation_date"
        elif any(h in name for h in ("updated", "update", "modified")):
            semantic_type = "modification_date"
        elif "birth" in name:
            semantic_type = "birth_date"
        elif "hire" in name:
            semantic_type = "hire_date"
        elif "settle" in name:
            semantic_type = "settlement_date"
        elif "auth" in name:
            semantic_type = "authorization_date"
        elif "expire" in name or "expir" in name:
            semantic_type = "expiration_date"
        elif "start" in name:
            semantic_type = "start_date"
        elif "end" in name:
            semantic_type = "end_date"

        return SemanticCandidate(
            column_name=profile.physical_name,
            normalized_key=profile.normalized_key,
            refined_type=refined_type,
            candidate_semantic_type=semantic_type,
            candidate_column_role=ColumnRole.TEMPORAL_DIMENSION,
            candidate_confidence=0.88,
            evidence=[
                {"type": "refined_type", "value": refined_type.value},
                {"type": "datetime_parse_ratio", "value": profile.datetime_parse_ratio},
            ],
        )

    def _flag_candidate(
        self, profile: ColumnProfileResult, refined_type: RefinedDataType
    ) -> SemanticCandidate:
        """Generate candidate for boolean/flag columns."""
        semantic_type = "flag"
        name = profile.normalized_key.lower()
        if "fraud" in name:
            semantic_type = "fraud_flag"
        elif "active" in name:
            semantic_type = "active_flag"
        elif "deleted" in name or "removed" in name:
            semantic_type = "deletion_flag"

        return SemanticCandidate(
            column_name=profile.physical_name,
            normalized_key=profile.normalized_key,
            refined_type=refined_type,
            candidate_semantic_type=semantic_type,
            candidate_column_role=ColumnRole.FLAG,
            candidate_confidence=0.85,
            evidence=[{"type": "refined_type", "value": "boolean"}],
        )

    def _numeric_candidate(
        self, profile: ColumnProfileResult, refined_type: RefinedDataType
    ) -> SemanticCandidate:
        """Generate candidate for numeric columns — metric or categorical."""
        name = profile.normalized_key.lower()
        evidence: list[dict[str, Any]] = []

        # Check if it's a metric (monetary amount, count, score)
        if _METRIC_PATTERNS.search(name):
            # Determine specific semantic type — business-labeled keywords checked
            # first so they win over the generic buckets below.
            # Business-concept words (what the column IS) checked before
            # actual/forecast/variance modifiers (WHICH data point it is) — most
            # insurance/finance columns in variance-reporting datasets carry an
            # _actual/_budget/_variance suffix on every underlying concept, so if
            # the modifier won here, premium/claims/revenue/expense columns would
            # all collapse into generic actual_amount/forecast_amount and never
            # be recognizable as their real business concept.
            semantic_type = "monetary_amount"
            if "revenue" in name:
                semantic_type = "revenue_amount"
            elif "premium" in name:
                semantic_type = "premium_amount"
            elif "claim" in name:
                semantic_type = "claims_amount"
            elif "expense" in name:
                semantic_type = "expense_amount"
            elif "variance" in name:
                semantic_type = "variance_amount"
            elif "actual" in name:
                semantic_type = "actual_amount"
            elif "forecast" in name or "budget" in name:
                semantic_type = "forecast_amount"
            elif any(h in name for h in ("salary", "compensation", "wage")):
                semantic_type = "salary_amount"
            elif "tenure" in name:
                semantic_type = "tenure"
            elif "satisfaction" in name:
                semantic_type = "satisfaction_score"
            elif any(h in name for h in ("count", "qty", "quantity", "volume")):
                semantic_type = "count_metric"
            elif any(h in name for h in ("score", "rating")):
                semantic_type = "score"
            elif any(h in name for h in ("rate", "ratio")):
                semantic_type = "rate"
            elif any(h in name for h in ("weight",)):
                semantic_type = "measurement"

            evidence.append({"type": "name_pattern", "value": "metric_keyword"})
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type=semantic_type,
                candidate_column_role=ColumnRole.METRIC,
                candidate_confidence=0.85,
                evidence=evidence,
            )

        # Check if low cardinality numeric → categorical/dimension
        if profile.cardinality_ratio <= 0.05 and profile.distinct_count <= 20:
            evidence.append({"type": "low_cardinality_numeric", "value": profile.distinct_count})
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type="numeric_category",
                candidate_column_role=ColumnRole.DIMENSION,
                candidate_confidence=0.70,
                evidence=evidence,
            )

        # Default numeric → metric
        evidence.append({"type": "numeric_continuous", "value": "high_cardinality"})
        return SemanticCandidate(
            column_name=profile.physical_name,
            normalized_key=profile.normalized_key,
            refined_type=refined_type,
            candidate_semantic_type="numeric_measure",
            candidate_column_role=ColumnRole.METRIC,
            candidate_confidence=0.65,
            evidence=evidence,
        )

    def _categorical_candidate(
        self, profile: ColumnProfileResult, refined_type: RefinedDataType
    ) -> SemanticCandidate:
        """Generate candidate for categorical columns."""
        name = profile.normalized_key.lower()
        evidence: list[dict[str, Any]] = [
            {"type": "refined_type", "value": "categorical"},
            {"type": "distinct_count", "value": profile.distinct_count},
        ]

        # Check for decline/reject reason codes before the generic description catch —
        # these are small fixed code sets meant to be charted as a dimension, not free text.
        if "reason" in name and ("decline" in name or "reject" in name):
            evidence.append({"type": "name_pattern", "value": "decline_reason_keyword"})
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type="decline_reason",
                candidate_column_role=ColumnRole.DIMENSION,
                candidate_confidence=0.80,
                evidence=evidence,
            )

        # Check for description/text fields that happen to have low cardinality
        if any(h in name for h in ("desc", "description", "note", "comment", "remark", "reason")):
            evidence.append({"type": "name_pattern", "value": "description_keyword"})
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type="description",
                candidate_column_role=ColumnRole.TEXT_FIELD,
                candidate_confidence=0.78,
                evidence=evidence,
            )

        # Check for status/flag
        if _STATUS_PATTERNS.search(name):
            semantic_type = "status"
            if "expense" in name:
                semantic_type = "expense_category"
            elif "segment" in name:
                semantic_type = "customer_segment"
            elif "loyalty" in name:
                semantic_type = "loyalty_tier"
            elif "department" in name or "dept" in name:
                semantic_type = "department"
            elif "channel" in name:
                semantic_type = "channel_dimension"
            elif "type" in name or "category" in name:
                semantic_type = "category"
            elif "tier" in name or "level" in name or "grade" in name:
                semantic_type = "tier"
            evidence.append({"type": "name_pattern", "value": "status_keyword"})
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type=semantic_type,
                candidate_column_role=ColumnRole.DIMENSION,
                candidate_confidence=0.82,
                evidence=evidence,
            )

        # Check for geographic
        if _GEO_PATTERNS.search(name):
            semantic_type = "geographic"
            if "country" in name:
                semantic_type = "country"
            elif "region" in name:
                semantic_type = "region"
            elif "city" in name:
                semantic_type = "city"
            elif "state" in name or "province" in name:
                semantic_type = "state_province"
            evidence.append({"type": "name_pattern", "value": "geographic_keyword"})
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type=semantic_type,
                candidate_column_role=ColumnRole.DIMENSION,
                candidate_confidence=0.80,
                evidence=evidence,
            )

        # Generic dimension
        return SemanticCandidate(
            column_name=profile.physical_name,
            normalized_key=profile.normalized_key,
            refined_type=refined_type,
            candidate_semantic_type="dimension",
            candidate_column_role=ColumnRole.DIMENSION,
            candidate_confidence=0.70,
            evidence=evidence,
        )

    def _text_candidate(
        self, profile: ColumnProfileResult, refined_type: RefinedDataType
    ) -> SemanticCandidate:
        """Generate candidate for text columns."""
        name = profile.normalized_key.lower()

        # Check for decline/reject reason codes before the generic description catch —
        # these are small fixed code sets meant to be charted as a dimension, not free text.
        if "reason" in name and ("decline" in name or "reject" in name):
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type="decline_reason",
                candidate_column_role=ColumnRole.DIMENSION,
                candidate_confidence=0.80,
                evidence=[{"type": "name_pattern", "value": "decline_reason_keyword"}],
            )

        # Check if it's a description or note
        if any(h in name for h in ("desc", "description", "note", "comment", "remark", "reason")):
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type="description",
                candidate_column_role=ColumnRole.TEXT_FIELD,
                candidate_confidence=0.78,
                evidence=[{"type": "name_pattern", "value": "description_keyword"}],
            )

        # Check if it's a name
        if any(h in name for h in ("name", "title", "label")):
            return SemanticCandidate(
                column_name=profile.physical_name,
                normalized_key=profile.normalized_key,
                refined_type=refined_type,
                candidate_semantic_type="name",
                candidate_column_role=ColumnRole.DIMENSION,
                candidate_confidence=0.75,
                evidence=[{"type": "name_pattern", "value": "name_keyword"}],
            )

        # Generic text
        return SemanticCandidate(
            column_name=profile.physical_name,
            normalized_key=profile.normalized_key,
            refined_type=refined_type,
            candidate_semantic_type="text",
            candidate_column_role=ColumnRole.TEXT_FIELD,
            candidate_confidence=0.50,
            evidence=[{"type": "default_text", "value": "no_specific_pattern"}],
        )
