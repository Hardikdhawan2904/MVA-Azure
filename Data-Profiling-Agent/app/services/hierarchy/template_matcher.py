"""Hierarchy template matching — maps configured template concepts to actual columns."""

import re
from typing import Any

from app.core.logging import get_logger
from app.services.profiling.column_profiler import ColumnProfileResult
from app.services.profiling.semantic_candidate_generator import SemanticCandidate

logger = get_logger(__name__)


class TemplateMatchResult:
    """Result of matching a template to dataset columns."""

    def __init__(
        self,
        template_key: str,
        priority: int,
        matched_levels: list[tuple[str, str]],  # (template_concept, column_name)
        missing_concepts: list[str],
        match_ratio: float,
    ):
        self.template_key = template_key
        self.priority = priority
        self.matched_levels = matched_levels
        self.missing_concepts = missing_concepts
        self.match_ratio = match_ratio


class TemplateMatcher:
    """Matches hierarchy templates to actual dataset columns."""

    def match_templates(
        self,
        templates: list[dict[str, Any]],
        profiles: list[ColumnProfileResult],
        semantic_candidates: list[SemanticCandidate],
        grain_columns: list[str],
    ) -> list[TemplateMatchResult]:
        """
        Match configured templates against available columns.

        - Exclude grain keys from candidate levels.
        - Match template concepts to columns by name similarity and semantic type.
        - Remove absent concepts.
        - Return sorted by match quality.
        """
        # Build column lookup excluding identifiers
        available_columns: dict[str, tuple[ColumnProfileResult, SemanticCandidate]] = {}
        for i, profile in enumerate(profiles):
            if profile.physical_name in grain_columns:
                continue
            candidate = semantic_candidates[i] if i < len(semantic_candidates) else None
            if candidate:
                available_columns[profile.normalized_key] = (profile, candidate)

        results: list[TemplateMatchResult] = []

        for template in templates:
            key = template.get("key", "unknown")
            priority = template.get("priority", 0)
            levels = template.get("levels", [])

            matched: list[tuple[str, str]] = []
            missing: list[str] = []
            used_columns: set[str] = set()  # Prevent reuse of same column

            for concept in levels:
                col_name = self._find_matching_column(concept, available_columns)
                if col_name and col_name not in used_columns:
                    matched.append((concept, col_name))
                    used_columns.add(col_name)
                else:
                    missing.append(concept)

            match_ratio = len(matched) / len(levels) if levels else 0.0

            results.append(TemplateMatchResult(
                template_key=key,
                priority=priority,
                matched_levels=matched,
                missing_concepts=missing,
                match_ratio=match_ratio,
            ))

        # Sort: best match ratio first, then by priority
        results.sort(key=lambda r: (-r.match_ratio, -r.priority))
        return results

    def _find_matching_column(
        self,
        concept: str,
        available: dict[str, tuple[ColumnProfileResult, SemanticCandidate]],
    ) -> str | None:
        """
        Find a column matching a template concept.

        Matching strategies:
        1. Exact normalized key match
        2. Concept contained in column name
        3. Column name contained in concept
        4. Semantic type match
        """
        concept_lower = concept.lower().replace(" ", "_").replace("-", "_")

        # Strategy 1: Exact match
        if concept_lower in available:
            return available[concept_lower][0].physical_name

        # Strategy 2 & 3: Substring match
        for key, (profile, candidate) in available.items():
            if concept_lower in key or key in concept_lower:
                return profile.physical_name

        # Strategy 4: Semantic type contains concept
        for key, (profile, candidate) in available.items():
            sem_type = (candidate.candidate_semantic_type or "").lower()
            if concept_lower in sem_type:
                return profile.physical_name

        # Strategy 5: Partial word match (concept words appear in column)
        concept_words = set(re.split(r"[_\s]+", concept_lower))
        for key, (profile, candidate) in available.items():
            col_words = set(re.split(r"[_\s]+", key))
            if concept_words & col_words:
                return profile.physical_name

        return None
