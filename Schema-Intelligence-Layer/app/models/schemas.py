"""
Pydantic schemas for request/response models and internal data structures.
"""

from pydantic import BaseModel, Field
from typing import Optional


class LLMClassification(BaseModel):
    """Schema for LLM classification output."""
    business_domain: str = "Other"
    sub_domain: str = "General"
    dataset_summary: str = ""
    confidence: Optional[float] = None
    reason: Optional[str] = None


class QualitySummary(BaseModel):
    """Summary of data quality analysis."""
    rows_analyzed: int
    columns: int


class QualityReport(BaseModel):
    """Data quality validation report."""
    dataset_score: float
    passing_score: float
    decision: str
    summary: QualitySummary
    checks: dict[str, float]
    warnings: list[str]


class DatasetMetadata(BaseModel):
    """Complete metadata record for a dataset."""
    dataset_id: str
    dataset_name: str
    file_type: str
    upload_timestamp: str
    business_domain: str = "Pending"
    sub_domain: str = "General"
    dataset_summary: str = ""
    row_count: int
    column_count: int
    column_names: list[str]
    column_data_types: list[str]
    column_descriptions: dict[str, str] = {}
    sample_data: list[dict] = []
    processing_status: str = "Processing"
    quality_score: Optional[float] = None
    quality_report: Optional[QualityReport] = None


class UploadResponse(BaseModel):
    """API response returned after successful dataset upload."""
    dataset_id: str
    dataset_name: str
    business_domain: str
    sub_domain: str
    dataset_summary: str
    row_count: int
    column_count: int
    column_descriptions: dict[str, str]
    status: str
    dataframe_records: list[dict] = Field(
        default=[],
        description="Full dataset as a list of row records (dict per row). "
                    "Intended for downstream agent consumption."
    )
    quality_report: Optional[QualityReport] = None


class DatasetListItem(BaseModel):
    """Summary item for dataset listing endpoint."""
    dataset_id: str
    dataset_name: str
    business_domain: str
    sub_domain: str
    row_count: int
    column_count: int
    upload_timestamp: str
    processing_status: str
