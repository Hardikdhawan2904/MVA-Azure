"""Identifier and dataset grain detection — uses behavior, not names alone."""

import re
from typing import Any

from app.core.constants import IDENTIFIER_CARDINALITY_THRESHOLD
from app.core.enums import RefinedDataType
from app.services.profiling.column_profiler import ColumnProfileResult


# Name hints that support (but do not prove) identifier status
_ID_NAME_PATTERNS = re.compile(
    r"(_id|_key|_uuid|_guid|_code|_number|_num|_no|_ref)$|"
    r"^(id|key|uuid|guid|code|number)$",
    re.IGNORECASE,
)


class IdentifierCandidate:
    """Result of identifier analysis for one column."""

    def __init__(
        self,
        column_name: str,
        normalized_key: str,
        is_identifier: bool,
        identifier_score: float,
        reason: str,
        evidence: list[dict[str, Any]],
    ):
        self.column_name = column_name
        self.normalized_key = normalized_key
        self.is_identifier = is_identifier
        self.identifier_score = identifier_score
        self.reason = reason
        self.evidence = evidence


class GrainDetectionResult:
    """Result of dataset grain detection."""

    def __init__(
        self,
        grain_columns: list[str],
        identifier_candidates: list[IdentifierCandidate],
    ):
        self.grain_columns = grain_columns
        self.identifier_candidates = identifier_candidates

    @property
    def inferred_grain(self) -> str | None:
        """Return a string representation of the grain."""
        if not self.grain_columns:
            return None
        return " + ".join(self.grain_columns)


class IdentifierDetector:
    """Detects identifiers and dataset grain using behavioral analysis."""

    def __init__(self, cardinality_threshold: float = IDENTIFIER_CARDINALITY_THRESHOLD):
        self._threshold = cardinality_threshold

    def detect(
        self,
        profiles: list[ColumnProfileResult],
        refined_types: list[RefinedDataType],
    ) -> GrainDetectionResult:
        """
        Analyze all columns to detect identifiers and infer dataset grain.

        Uses:
        - Cardinality ratio (primary signal)
        - Null ratio (identifiers should have low nulls)
        - Duplicate behavior
        - Value patterns (UUID, sequential integers)
        - Name hints (supporting evidence only)
        - Refined type
        """
        candidates: list[IdentifierCandidate] = []

        for i, profile in enumerate(profiles):
            refined_type = refined_types[i] if i < len(refined_types) else RefinedDataType.UNKNOWN
            candidate = self._analyze_column(profile, refined_type)
            candidates.append(candidate)

        # Determine grain columns (identifiers with highest scores)
        grain_columns = [
            c.column_name for c in candidates
            if c.is_identifier
        ]

        return GrainDetectionResult(
            grain_columns=grain_columns,
            identifier_candidates=candidates,
        )

    def _analyze_column(
        self, profile: ColumnProfileResult, refined_type: RefinedDataType
    ) -> IdentifierCandidate:
        """Analyze a single column for identifier characteristics."""
        evidence: list[dict[str, Any]] = []
        score = 0.0

        # Primary signal: cardinality ratio
        if profile.non_null_count > 0:
            cardinality = profile.cardinality_ratio

            if cardinality >= self._threshold:
                score += 0.50
                evidence.append({
                    "signal": "high_cardinality",
                    "value": round(cardinality, 4),
                    "threshold": self._threshold,
                })
            elif cardinality >= 0.90:
                score += 0.25
                evidence.append({
                    "signal": "moderate_cardinality",
                    "value": round(cardinality, 4),
                })

        # Null ratio — identifiers should have low nulls
        if profile.null_ratio <= 0.01:
            score += 0.10
            evidence.append({"signal": "low_null_ratio", "value": round(profile.null_ratio, 4)})
        elif profile.null_ratio > 0.10:
            score -= 0.15
            evidence.append({"signal": "high_null_ratio_penalty", "value": round(profile.null_ratio, 4)})

        # Duplicate behavior — true identifiers have zero or near-zero duplicates
        if profile.duplicate_count == 0 and profile.non_null_count > 1:
            score += 0.15
            evidence.append({"signal": "zero_duplicates"})

        # Pattern hints
        if "UUID" in profile.dominant_patterns:
            score += 0.20
            evidence.append({"signal": "uuid_pattern"})

        # Refined type already detected as identifier
        if refined_type == RefinedDataType.IDENTIFIER:
            score += 0.10
            evidence.append({"signal": "refined_type_identifier"})

        # Name hint (supporting only, not primary)
        if _ID_NAME_PATTERNS.search(profile.normalized_key):
            score += 0.05
            evidence.append({"signal": "name_hint", "pattern": profile.normalized_key})

        # Negative signals: low cardinality means NOT an identifier
        if profile.cardinality_ratio < 0.50 and profile.non_null_count > 10:
            score -= 0.30
            evidence.append({
                "signal": "low_cardinality_disqualifies",
                "value": round(profile.cardinality_ratio, 4),
            })

        # Continuous decimal values are essentially never real identifiers —
        # financial amounts, rates, and measurements naturally land close to 100%
        # unique without being an identifier, so cardinality/duplicate behavior
        # alone can't tell the two apart. Real identifiers are virtually always
        # integer/string/UUID; require a supporting name hint for decimals.
        if refined_type == RefinedDataType.DECIMAL and not _ID_NAME_PATTERNS.search(profile.normalized_key):
            score -= 0.40
            evidence.append({
                "signal": "decimal_without_id_name_penalty",
                "value": profile.pandas_dtype,
            })

        # Same reasoning applies to long free-text values (commentary/narrative
        # fields) — every row happening to be a unique sentence doesn't make it
        # an identifier. A genuine ID/code is short (same 40-char threshold used
        # in TypeRefiner's identifier check).
        if (
            profile.avg_string_length is not None
            and profile.avg_string_length > 40
            and not _ID_NAME_PATTERNS.search(profile.normalized_key)
        ):
            score -= 0.40
            evidence.append({
                "signal": "long_text_without_id_name_penalty",
                "value": round(profile.avg_string_length, 1),
            })

        # Cap score to [0, 1]
        score = max(0.0, min(1.0, score))

        # Decision
        is_identifier = score >= 0.60 and profile.cardinality_ratio >= self._threshold

        reason = "identifier" if is_identifier else "not_identifier"
        if is_identifier:
            reason = f"Identified as grain key (score={score:.2f}, cardinality={profile.cardinality_ratio:.4f})"
        else:
            reason = f"Not identifier (score={score:.2f}, cardinality={profile.cardinality_ratio:.4f})"

        return IdentifierCandidate(
            column_name=profile.physical_name,
            normalized_key=profile.normalized_key,
            is_identifier=is_identifier,
            identifier_score=round(score, 3),
            reason=reason,
            evidence=evidence,
        )
