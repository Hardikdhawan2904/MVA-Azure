"""Pre-aggregated drill-down cube model."""

import uuid
from datetime import datetime

from sqlalchemy import String, Integer, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DrillDownCube(Base):
    __tablename__ = "drill_down_cubes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profile_runs.id"), nullable=False
    )
    chart_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chart_specifications.id"), nullable=False
    )
    hierarchy_chain_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("hierarchy_chains.id"), nullable=True
    )
    level_column: Mapped[str] = mapped_column(String(500), nullable=False)
    level_depth: Mapped[int] = mapped_column(Integer, nullable=False)
    dimension_path_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    aggregated_data_json: Mapped[list] = mapped_column(JSONB, nullable=False)
    metric_column: Mapped[str] = mapped_column(String(500), nullable=False)
    aggregation: Mapped[str] = mapped_column(String(20), nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
