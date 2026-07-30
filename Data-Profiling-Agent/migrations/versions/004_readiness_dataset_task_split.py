"""Add dataset_score/task_compatibility_score to readiness_assessments

Splits ml_readiness/llm_readiness's single `score` into an explicit
question-independent component (dataset_score) and question-dependent
component (task_compatibility_score) — analytics_readiness sets
dataset_score = score with task_compatibility_score left null (it has no
task-dependent input at all). `score` itself is unchanged; these two are
purely additive transparency into how it was built.

Revision ID: 004
Revises: 003
Create Date: 2026-07-23 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = 'agent2'


def upgrade() -> None:
    op.add_column('readiness_assessments', sa.Column('dataset_score', sa.Float(), nullable=True), schema=SCHEMA)
    op.add_column('readiness_assessments', sa.Column('task_compatibility_score', sa.Float(), nullable=True), schema=SCHEMA)


def downgrade() -> None:
    op.drop_column('readiness_assessments', 'task_compatibility_score', schema=SCHEMA)
    op.drop_column('readiness_assessments', 'dataset_score', schema=SCHEMA)
