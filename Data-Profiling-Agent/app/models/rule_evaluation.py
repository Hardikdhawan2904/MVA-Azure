"""Rule evaluation result model."""

import uuid

from sqlalchemy import Integer, Float, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RuleEvaluation(Base):
    __tablename__ = "rule_evaluations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("profile_runs.id"), nullable=False
    )
    # Only approved-suggestion-sourced rules ever get a rule_definitions row —
    # YAML domain rules and ad-hoc request rules don't, so this stays nullable
    # and rule_key/rule_type/source below carry identity for every row instead.
    rule_definition_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rule_definitions.id"), nullable=True
    )
    rule_key: Mapped[str] = mapped_column(String(200), nullable=False)
    rule_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(30), nullable=False)
    records_checked: Mapped[int] = mapped_column(Integer, nullable=False)
    pass_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fail_count: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
