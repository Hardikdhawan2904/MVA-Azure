"""app/services/dataset_context/models.py — the canonical internal
representation everything downstream of Stage 0 consumes (Agent 3
redesign, Phase 0).

Decouples the rest of Agent 3 from Agent 2's exact response shape — only
DatasetContextBuilder needs to change if that shape evolves. Plain
dataclasses, not Pydantic: these are built once per request (potentially
one ColumnContext per column, up to ~140 on the reference dataset) and
never cross a network boundary themselves, so validation overhead buys
nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ContextSource = Literal["agent2", "local_fallback"]

# Mirrors Agent 2's ColumnRole vocabulary (app/models/enums.py equivalent
# in Data-Profiling-Agent) — kept as plain strings here rather than an
# imported enum since Agent 3 must not import Agent 2's code (separate
# service, separate deployable).
SEMANTIC_ROLE_METRIC = "metric"
SEMANTIC_ROLE_DIMENSION = "dimension"
SEMANTIC_ROLE_IDENTIFIER = "identifier"
SEMANTIC_ROLE_TEMPORAL = "temporal_dimension"
SEMANTIC_ROLE_TEXT = "text_field"
SEMANTIC_ROLE_FLAG = "flag"
SEMANTIC_ROLE_UNKNOWN = "unknown"


@dataclass
class ColumnContext:
    name: str
    physical_type: str | None = None
    semantic_type: str | None = None
    semantic_role: str = SEMANTIC_ROLE_UNKNOWN
    cardinality_ratio: float | None = None
    null_ratio: float | None = None
    is_identifier: bool = False
    is_temporal: bool = False
    sample_values: list[Any] = field(default_factory=list)
    confidence: float | None = None  # confirmed/candidate confidence, whichever was used


@dataclass
class HierarchyInfo:
    status: str | None = None
    template_key: str | None = None
    level_columns: list[str] = field(default_factory=list)
    average_confidence: float | None = None


@dataclass
class DatasetContext:
    row_count: int
    column_count: int
    columns: list[ColumnContext]
    context_source: ContextSource
    hierarchy: HierarchyInfo | None = None
    detected_domain: str | None = None
    feature_recommendation: dict[str, Any] | None = None
    charts: list[dict[str, Any]] = field(default_factory=list)

    def column(self, name: str) -> ColumnContext | None:
        return next((c for c in self.columns if c.name == name), None)

    def columns_with_role(self, role: str) -> list[ColumnContext]:
        return [c for c in self.columns if c.semantic_role == role]

    def columns_with_semantic_type(self, *semantic_types: str) -> list[ColumnContext]:
        wanted = set(semantic_types)
        return [c for c in self.columns if c.semantic_type in wanted]
