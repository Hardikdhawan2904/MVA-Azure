"""Hierarchy chain and edge models."""

import uuid
from datetime import datetime

from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class HierarchyChain(Base):
    __tablename__ = "hierarchy_chains"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profile_runs.id"), nullable=False
    )
    template_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    average_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    level_columns_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    warnings_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    algorithm_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class HierarchyEdge(Base):
    __tablename__ = "hierarchy_edges"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chain_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("hierarchy_chains.id"), nullable=False
    )
    parent_column: Mapped[str] = mapped_column(String(500), nullable=False)
    child_column: Mapped[str] = mapped_column(String(500), nullable=False)
    distinct_child_count: Mapped[int] = mapped_column(Integer, nullable=False)
    mapped_child_count: Mapped[int] = mapped_column(Integer, nullable=False)
    violating_child_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fd_consistency: Mapped[float] = mapped_column(Float, nullable=False)
    mapping_coverage: Mapped[float] = mapped_column(Float, nullable=False)
    edge_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    conflict_samples_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
