"""Chain selector — selects the best single hierarchy chain from candidates."""

from typing import Any

import pandas as pd

from app.core.enums import HierarchyEdgeStatus, HierarchyChainStatus
from app.core.logging import get_logger
from app.services.hierarchy.functional_dependency import (
    FunctionalDependencyValidator,
    FDValidationResult,
)
from app.services.hierarchy.template_matcher import TemplateMatchResult

logger = get_logger(__name__)


class HierarchyChainResult:
    """Final result of hierarchy chain selection."""

    def __init__(
        self,
        status: HierarchyChainStatus,
        template_key: str | None,
        level_columns: list[str],
        edges: list[FDValidationResult],
        average_confidence: float,
        warnings: list[str],
        rejected_edges: list[FDValidationResult],
    ):
        self.status = status
        self.template_key = template_key
        self.level_columns = level_columns
        self.edges = edges
        self.average_confidence = average_confidence
        self.warnings = warnings
        self.rejected_edges = rejected_edges

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "template_key": self.template_key,
            "level_columns": self.level_columns,
            "edge_count": len(self.edges),
            "edges": [e.to_dict() for e in self.edges],
            "average_confidence": round(self.average_confidence, 4),
            "warnings": self.warnings,
            "rejected_edge_count": len(self.rejected_edges),
        }


class ChainSelector:
    """
    Selects the best single hierarchy chain.

    Policy:
    1. Reject invalid edges.
    2. Find valid contiguous chain segments.
    3. Prefer the longest valid chain.
    4. Tie-break using highest average edge confidence.
    5. Final tie-break using template priority.
    6. Return unresolved if no chain of at least 2 levels.
    7. A single dimension is NOT a hierarchy.
    """

    def __init__(self, fd_validator: FunctionalDependencyValidator | None = None):
        self._fd_validator = fd_validator or FunctionalDependencyValidator()

    def select(
        self,
        df: pd.DataFrame,
        template_matches: list[TemplateMatchResult],
    ) -> HierarchyChainResult:
        """
        Select the best hierarchy chain from template matches.

        Validates each candidate chain's edges using functional dependencies.
        """
        if not template_matches:
            return self._unresolved("No hierarchy templates matched")

        best_chain: HierarchyChainResult | None = None

        for match in template_matches:
            # Need at least 2 matched levels to form a hierarchy
            if len(match.matched_levels) < 2:
                continue

            # Get ordered column names from matched levels
            columns = [col_name for _, col_name in match.matched_levels]

            # Validate all adjacent edges
            chain_result = self._validate_chain(df, columns, match.template_key)

            if chain_result.status == HierarchyChainStatus.UNRESOLVED:
                continue

            # Compare with current best
            if best_chain is None:
                best_chain = chain_result
            else:
                best_chain = self._compare_chains(best_chain, chain_result, match)

        if best_chain is None:
            return self._unresolved("No valid chain of 2+ levels found")

        return best_chain

    def _validate_chain(
        self, df: pd.DataFrame, columns: list[str], template_key: str
    ) -> HierarchyChainResult:
        """Validate a candidate chain by checking all adjacent edges."""
        all_edges: list[FDValidationResult] = []
        rejected_edges: list[FDValidationResult] = []
        warnings: list[str] = []

        # Validate each adjacent pair (parent → child means child → parent FD)
        for i in range(len(columns) - 1):
            parent_col = columns[i]
            child_col = columns[i + 1]

            # Check columns exist in dataframe
            if parent_col not in df.columns or child_col not in df.columns:
                rejected_edges.append(FDValidationResult(
                    parent_column=parent_col,
                    child_column=child_col,
                    distinct_child_count=0,
                    mapped_child_count=0,
                    violating_child_count=0,
                    fd_consistency=0.0,
                    mapping_coverage=0.0,
                    edge_confidence=0.0,
                    status=HierarchyEdgeStatus.REJECTED,
                    conflict_samples=[],
                ))
                continue

            edge = self._fd_validator.validate_edge(df, parent_col, child_col)
            all_edges.append(edge)

            if edge.status == HierarchyEdgeStatus.REJECTED:
                rejected_edges.append(edge)
            elif edge.status == HierarchyEdgeStatus.RETAINED_WITH_WARNING:
                warnings.append(
                    f"Edge {parent_col}→{child_col}: fd_consistency={edge.fd_consistency:.3f}, "
                    f"{edge.violating_child_count} conflicting child values"
                )

        # Find longest contiguous valid segment
        valid_edges, segment_columns = self._find_longest_valid_segment(
            all_edges, columns, rejected_edges
        )

        if len(segment_columns) < 2:
            return HierarchyChainResult(
                status=HierarchyChainStatus.UNRESOLVED,
                template_key=template_key,
                level_columns=[],
                edges=[],
                average_confidence=0.0,
                warnings=["No valid contiguous chain of 2+ levels"],
                rejected_edges=rejected_edges,
            )

        # Calculate average confidence
        avg_confidence = (
            sum(e.edge_confidence for e in valid_edges) / len(valid_edges)
            if valid_edges
            else 0.0
        )

        # Determine status
        if any(e.status == HierarchyEdgeStatus.RETAINED_WITH_WARNING for e in valid_edges):
            status = HierarchyChainStatus.PARTIAL
        else:
            status = HierarchyChainStatus.ACCEPTED

        return HierarchyChainResult(
            status=status,
            template_key=template_key,
            level_columns=segment_columns,
            edges=valid_edges,
            average_confidence=avg_confidence,
            warnings=warnings,
            rejected_edges=rejected_edges,
        )

    def _find_longest_valid_segment(
        self,
        all_edges: list[FDValidationResult],
        columns: list[str],
        rejected_edges: list[FDValidationResult],
    ) -> tuple[list[FDValidationResult], list[str]]:
        """Find the longest contiguous segment of valid edges."""
        rejected_indices: set[int] = set()
        for i, edge in enumerate(all_edges):
            if edge.status == HierarchyEdgeStatus.REJECTED:
                rejected_indices.add(i)

        # Find contiguous segments
        best_segment_edges: list[FDValidationResult] = []
        best_segment_cols: list[str] = []

        current_edges: list[FDValidationResult] = []
        current_start = 0

        for i, edge in enumerate(all_edges):
            if i in rejected_indices:
                # End current segment
                segment_cols = columns[current_start: current_start + len(current_edges) + 1]
                if len(segment_cols) > len(best_segment_cols):
                    best_segment_edges = current_edges[:]
                    best_segment_cols = segment_cols
                current_edges = []
                current_start = i + 1
            else:
                current_edges.append(edge)

        # Don't forget the last segment
        segment_cols = columns[current_start: current_start + len(current_edges) + 1]
        if len(segment_cols) > len(best_segment_cols):
            best_segment_edges = current_edges[:]
            best_segment_cols = segment_cols

        return best_segment_edges, best_segment_cols

    def _compare_chains(
        self,
        current_best: HierarchyChainResult,
        candidate: HierarchyChainResult,
        match: TemplateMatchResult,
    ) -> HierarchyChainResult:
        """Compare two chains and return the better one."""
        # Prefer longer chain
        if len(candidate.level_columns) > len(current_best.level_columns):
            return candidate
        if len(candidate.level_columns) < len(current_best.level_columns):
            return current_best

        # Tie-break: higher average confidence
        if candidate.average_confidence > current_best.average_confidence:
            return candidate

        return current_best

    def _unresolved(self, reason: str) -> HierarchyChainResult:
        """Return an unresolved result."""
        return HierarchyChainResult(
            status=HierarchyChainStatus.UNRESOLVED,
            template_key=None,
            level_columns=[],
            edges=[],
            average_confidence=0.0,
            warnings=[reason],
            rejected_edges=[],
        )
