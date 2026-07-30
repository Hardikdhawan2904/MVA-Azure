"""Target/feature/drop recommendation model — output of the feature_target_agent
sub-agent (see app/agents/feature_target_agent/), persisted so a downstream
consumer can read it without re-deriving it."""

import uuid

from sqlalchemy import String, Float, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class FeatureRecommendation(Base):
    __tablename__ = "feature_recommendations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profile_runs.id"), nullable=False
    )
    business_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    problem_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    time_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    recommended_approach: Mapped[str | None] = mapped_column(String(20), nullable=True)
    approach_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_columns_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    drop_columns_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
