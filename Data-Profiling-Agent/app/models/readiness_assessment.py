"""AI readiness assessment model."""

import uuid

from sqlalchemy import String, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ReadinessAssessment(Base):
    __tablename__ = "readiness_assessments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profile_runs.id"), nullable=False
    )
    assessment_type: Mapped[str] = mapped_column(String(30), nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    strengths_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    blockers_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    recommendations_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    evidence_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    weight_profile_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    dataset_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    task_compatibility_score: Mapped[float | None] = mapped_column(Float, nullable=True)
