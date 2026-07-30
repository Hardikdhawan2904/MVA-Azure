"""Functional dependency validation for hierarchy edges."""

from typing import Any

import pandas as pd

from app.core.constants import (
    HIERARCHY_ACCEPTED_CONSISTENCY,
    HIERARCHY_WARNING_CONSISTENCY,
    HIERARCHY_MIN_MAPPING_COVERAGE,
    MAX_CONFLICT_SAMPLES,
)
from app.core.enums import HierarchyEdgeStatus
from app.core.logging import get_logger

logger = get_logger(__name__)


class FDValidationResult:
    """Result of functional-dependency validation for one edge."""

    def __init__(
        self,
        parent_column: str,
        child_column: str,
        distinct_child_count: int,
        mapped_child_count: int,
        violating_child_count: int,
        fd_consistency: float,
        mapping_coverage: float,
        edge_confidence: float,
        status: HierarchyEdgeStatus,
        conflict_samples: list[dict[str, Any]],
    ):
        self.parent_column = parent_column
        self.child_column = child_column
        self.distinct_child_count = distinct_child_count
        self.mapped_child_count = mapped_child_count
        self.violating_child_count = violating_child_count
        self.fd_consistency = fd_consistency
        self.mapping_coverage = mapping_coverage
        self.edge_confidence = edge_confidence
        self.status = status
        self.conflict_samples = conflict_samples

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_column": self.parent_column,
            "child_column": self.child_column,
            "distinct_child_count": self.distinct_child_count,
            "mapped_child_count": self.mapped_child_count,
            "violating_child_count": self.violating_child_count,
            "fd_consistency": round(self.fd_consistency, 4),
            "mapping_coverage": round(self.mapping_coverage, 4),
            "edge_confidence": round(self.edge_confidence, 4),
            "status": self.status.value,
            "conflict_count": self.violating_child_count,
            "conflict_samples": self.conflict_samples,
        }


class FunctionalDependencyValidator:
    """
    Validates functional dependencies between hierarchy levels.

    For an edge Parent → Child, validates: Child → Parent
    (each distinct child maps to exactly one parent).
    """

    def __init__(
        self,
        accepted_consistency: float = HIERARCHY_ACCEPTED_CONSISTENCY,
        warning_consistency: float = HIERARCHY_WARNING_CONSISTENCY,
        min_mapping_coverage: float = HIERARCHY_MIN_MAPPING_COVERAGE,
        max_conflict_samples: int = MAX_CONFLICT_SAMPLES,
    ):
        self._accepted = accepted_consistency
        self._warning = warning_consistency
        self._min_coverage = min_mapping_coverage
        self._max_samples = max_conflict_samples

    def validate_edge(
        self, df: pd.DataFrame, parent_column: str, child_column: str
    ) -> FDValidationResult:
        """
        Validate the functional dependency: child → parent.

        Null child values are excluded from the FD denominator.
        Non-null child values with missing (null) parents reduce mapping coverage
        but do not count as multi-parent violations.
        """
        # Get non-null child values
        mask = df[child_column].notna()
        subset = df.loc[mask, [child_column, parent_column]]

        distinct_child_count = int(subset[child_column].nunique())

        if distinct_child_count == 0:
            return FDValidationResult(
                parent_column=parent_column,
                child_column=child_column,
                distinct_child_count=0,
                mapped_child_count=0,
                violating_child_count=0,
                fd_consistency=1.0,
                mapping_coverage=0.0,
                edge_confidence=0.0,
                status=HierarchyEdgeStatus.REJECTED,
                conflict_samples=[],
            )

        # For each distinct child, find its parent values
        # Exclude rows where parent is null (for mapping coverage)
        both_non_null = subset[subset[parent_column].notna()]
        mapped_child_count = int(both_non_null[child_column].nunique())

        # Find children mapping to multiple parents
        child_parent_groups = both_non_null.groupby(child_column)[parent_column].nunique()
        violating_children = child_parent_groups[child_parent_groups > 1]
        violating_child_count = len(violating_children)

        # Calculate metrics
        if distinct_child_count > 0:
            fd_consistency = 1.0 - (violating_child_count / distinct_child_count)
        else:
            fd_consistency = 1.0

        if distinct_child_count > 0:
            mapping_coverage = mapped_child_count / distinct_child_count
        else:
            mapping_coverage = 0.0

        edge_confidence = fd_consistency * mapping_coverage

        # Determine status
        status = self._determine_status(fd_consistency, mapping_coverage)

        # Collect conflict samples
        conflict_samples = self._collect_conflict_samples(
            both_non_null, child_column, parent_column, violating_children
        )

        return FDValidationResult(
            parent_column=parent_column,
            child_column=child_column,
            distinct_child_count=distinct_child_count,
            mapped_child_count=mapped_child_count,
            violating_child_count=violating_child_count,
            fd_consistency=fd_consistency,
            mapping_coverage=mapping_coverage,
            edge_confidence=edge_confidence,
            status=status,
            conflict_samples=conflict_samples,
        )

    def _determine_status(
        self, fd_consistency: float, mapping_coverage: float
    ) -> HierarchyEdgeStatus:
        """Determine edge status from consistency and coverage."""
        if mapping_coverage < self._min_coverage:
            return HierarchyEdgeStatus.REJECTED

        if fd_consistency >= self._accepted:
            return HierarchyEdgeStatus.ACCEPTED

        if fd_consistency >= self._warning:
            return HierarchyEdgeStatus.RETAINED_WITH_WARNING

        return HierarchyEdgeStatus.REJECTED

    def _collect_conflict_samples(
        self,
        df: pd.DataFrame,
        child_column: str,
        parent_column: str,
        violating_children: pd.Series,
    ) -> list[dict[str, Any]]:
        """Collect bounded conflict samples for reporting."""
        samples: list[dict[str, Any]] = []

        for child_value in violating_children.index[: self._max_samples]:
            parent_values = (
                df[df[child_column] == child_value][parent_column]
                .unique()
                .tolist()
            )
            samples.append({
                "child_value": str(child_value),
                "parent_values": [str(p) for p in parent_values[:5]],
            })

        return samples
