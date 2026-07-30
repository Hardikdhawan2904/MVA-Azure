"""Data models for Schema Intelligence inputs and outputs."""

from typing import Any
from pydantic import BaseModel, Field

from app.core.enums import SchemaIntelligenceDecision


class ColumnAnalysisInput(BaseModel):
    """Input for Schema Intelligence analysis of a single column."""
    column_name: str
    normalized_key: str
    description: str | None = None
    refined_physical_type: str
    statistics_summary: dict[str, Any] = {}
    representative_sample_values: list[Any] = []
    candidate_semantic_type: str | None = None
    candidate_column_role: str | None = None
    candidate_confidence: float = 0.0
    primary_domain: str = ""


class ColumnAnalysisResult(BaseModel):
    """Output of Schema Intelligence analysis for a single column."""
    column_name: str
    decision: SchemaIntelligenceDecision
    confirmed_semantic_type: str | None = None
    confirmed_column_role: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    recommended_mandatory: bool | None = None
    recommended_expected_unique: bool | None = None


class DomainContext(BaseModel):
    """Domain context passed to Schema Intelligence."""
    primary_domain: str
    secondary_domains: list[str] = []
    row_count: int = 0
    column_count: int = 0


class SchemaIntelligenceResult(BaseModel):
    """Complete result from Schema Intelligence analysis."""
    column_results: list[ColumnAnalysisResult]
    model_name: str = ""
    prompt_version: str = ""
    success: bool = True
    fallback_used: bool = False
    error: str | None = None
