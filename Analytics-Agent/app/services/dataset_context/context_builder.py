"""app/services/dataset_context/context_builder.py — Stage 0 of the Agent 3
redesign: build a DatasetContext, the canonical internal model everything
downstream consumes.

Two paths:
  - Rich: Agent 2's column_profiles/hierarchy/charts were forwarded by the
    Orchestrator (Decision A1) — map them directly onto ColumnContext.
  - Fallback: nothing was forwarded (a direct /analyze call outside the
    Orchestrator, or an Orchestrator that hasn't been updated yet) —
    delegate to LocalSchemaInferer. This is what makes "Agent 3 must still
    generate analytics even if Agent 1 can't classify the domain" true:
    the generic path never *requires* Agent 2's data, it's just richer
    with it.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.services.dataset_context.local_schema_inferer import LocalSchemaInferer
from app.services.dataset_context.models import ColumnContext, DatasetContext, HierarchyInfo

logger = logging.getLogger(__name__)


class DatasetContextBuilder:
    def build(
        self,
        df: pd.DataFrame,
        column_profiles: list[dict[str, Any]] | None = None,
        hierarchy: dict[str, Any] | None = None,
        charts: list[dict[str, Any]] | None = None,
        full_feature_recommendation: dict[str, Any] | None = None,
        detected_domain: str | None = None,
    ) -> DatasetContext:
        if column_profiles:
            return self._build_from_agent2(
                df, column_profiles, hierarchy, charts, full_feature_recommendation, detected_domain,
            )
        logger.info("No column_profiles forwarded — building DatasetContext via LocalSchemaInferer.")
        return LocalSchemaInferer().infer(df, detected_domain=detected_domain)

    def _build_from_agent2(
        self,
        df: pd.DataFrame,
        column_profiles: list[dict[str, Any]],
        hierarchy: dict[str, Any] | None,
        charts: list[dict[str, Any]] | None,
        full_feature_recommendation: dict[str, Any] | None,
        detected_domain: str | None,
    ) -> DatasetContext:
        columns = [self._column_context_from_profile(p) for p in column_profiles]

        hierarchy_info = None
        if hierarchy and hierarchy.get("status") not in (None, "unresolved"):
            hierarchy_info = HierarchyInfo(
                status=hierarchy.get("status"),
                template_key=hierarchy.get("template_key"),
                level_columns=hierarchy.get("level_columns") or [],
                average_confidence=hierarchy.get("average_confidence"),
            )

        return DatasetContext(
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            context_source="agent2",
            hierarchy=hierarchy_info,
            detected_domain=detected_domain,
            feature_recommendation=full_feature_recommendation,
            charts=charts or [],
        )

    @staticmethod
    def _column_context_from_profile(profile: dict[str, Any]) -> ColumnContext:
        # confirmed_* (Schema Intelligence's LLM confirm/override) wins when
        # present; candidate_* (pure deterministic classification) is the
        # fallback — same precedence Agent 2 itself documents.
        semantic_type = profile.get("confirmed_semantic_type") or profile.get("candidate_semantic_type")
        semantic_role = profile.get("confirmed_column_role") or profile.get("candidate_column_role") or "unknown"
        confidence = profile.get("schema_confidence")
        if confidence is None:
            confidence = profile.get("candidate_confidence")

        statistics = profile.get("statistics") or {}
        sample_values = statistics.get("sample_values") or statistics.get("representative_values") or []

        # Agent 2 assigns column *role* by a priority order where a
        # high-cardinality date column (e.g. one reporting_date per row —
        # the common case for any daily-grain time series) loses to
        # "identifier"/grain-key before "temporal_dimension" is ever
        # considered. Trusting semantic_role alone here would make Agent 3
        # blind to temporal columns on exactly the datasets most likely to
        # need trend/forecast analysis. refined_data_type ("date"/
        # "datetime", from Agent 2's own type refiner — see
        # Data-Profiling-Agent/app/core/enums.py::RefinedDataType) is a
        # physical-type signal Agent 2 always sets correctly regardless of
        # role classification, so it's used as a fallback — the same
        # temporal-wins-over-identifier precedence LocalSchemaInferer's own
        # _is_temporal()/is_identifier already apply on the no-Agent-2 path.
        physical_type = profile.get("refined_data_type")
        is_temporal = semantic_role == "temporal_dimension" or physical_type in ("date", "datetime")

        return ColumnContext(
            name=profile.get("physical_name", ""),
            physical_type=physical_type,
            semantic_type=semantic_type,
            semantic_role=semantic_role,
            cardinality_ratio=statistics.get("cardinality_ratio"),
            null_ratio=statistics.get("null_ratio"),
            is_identifier=bool(profile.get("is_grain_key")),
            is_temporal=is_temporal,
            sample_values=list(sample_values)[:5],
            confidence=confidence,
        )
