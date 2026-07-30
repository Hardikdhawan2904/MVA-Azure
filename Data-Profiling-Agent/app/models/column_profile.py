"""Column-level profile model."""

import uuid
from datetime import datetime

from sqlalchemy import String, Float, Boolean, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ColumnProfile(Base):
    __tablename__ = "column_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profile_runs.id"), nullable=False
    )
    physical_name: Mapped[str] = mapped_column(String(500), nullable=False)
    normalized_key: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    pandas_dtype: Mapped[str] = mapped_column(String(50), nullable=False)
    refined_data_type: Mapped[str] = mapped_column(String(50), nullable=False)
    statistics_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    candidate_semantic_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    candidate_column_role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    candidate_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confirmed_semantic_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    confirmed_column_role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    schema_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    identifier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_grain_key: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mandatory: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mandatory_source: Mapped[str | None] = mapped_column(String(30), nullable=True)
    expected_unique: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    expected_unique_source: Mapped[str | None] = mapped_column(String(30), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
