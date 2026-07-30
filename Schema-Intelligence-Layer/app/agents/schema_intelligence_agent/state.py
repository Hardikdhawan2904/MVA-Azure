"""Typed state for the Schema Intelligence Agent's LangGraph — threads pipeline
data between nodes, replacing the old linear route-handler function body.
"""

from typing import Any, TypedDict

import pandas as pd

from app.models.schemas import DatasetMetadata


class SchemaIntelligenceState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    filename: str
    content: bytes
    force_reclassify: bool

    # ── File loading ────────────────────────────────────────────────────────
    file_extension: str
    df: pd.DataFrame

    # ── Quality gate ────────────────────────────────────────────────────────
    quality_report: dict[str, Any]
    quality_decision: str

    # ── Existing-dataset lookup ─────────────────────────────────────────────
    existing_metadata: DatasetMetadata | None
    is_new: bool
    profiling_df: pd.DataFrame

    # ── Metadata / classification ──────────────────────────────────────────
    metadata: DatasetMetadata
    column_descriptions: dict[str, str]
    business_domain: str
    sub_domain: str
    dataset_summary: str
    processing_status: str

    # ── Output ──────────────────────────────────────────────────────────────
    response: dict[str, Any]
    error_status_code: int
    error_content: dict[str, Any]
