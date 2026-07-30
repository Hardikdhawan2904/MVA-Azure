"""Add feature_recommendations table

Target/feature/drop classification produced by the feature_target_agent
sub-agent (app/agents/feature_target_agent/) — one row per profile run,
mirroring readiness_assessments' JSON-columns-plus-FK shape.

Revision ID: 003
Revises: 002
Create Date: 2026-07-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Same reasoning as 002: the connecting role's search_path doesn't include
# agent2, so every op call here must be explicitly schema-qualified.
SCHEMA = 'agent2'


def upgrade() -> None:
    op.create_table('feature_recommendations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('business_question', sa.Text(), nullable=True),
        sa.Column('target_column', sa.String(255), nullable=True),
        sa.Column('problem_type', sa.String(30), nullable=True),
        sa.Column('time_column', sa.String(255), nullable=True),
        sa.Column('recommended_approach', sa.String(20), nullable=True),
        sa.Column('approach_reasoning', sa.Text(), nullable=True),
        sa.Column('feature_columns_json', postgresql.JSONB(), nullable=True),
        sa.Column('drop_columns_json', postgresql.JSONB(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], [f'{SCHEMA}.profile_runs.id']),
        sa.PrimaryKeyConstraint('id'),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table('feature_recommendations', schema=SCHEMA)
