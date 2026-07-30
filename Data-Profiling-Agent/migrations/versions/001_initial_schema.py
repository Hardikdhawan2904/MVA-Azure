"""Initial schema

Revision ID: 001
Revises: None
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('profile_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('primary_domain', sa.String(50), nullable=False),
        sa.Column('dominant_secondary_domain', sa.String(100), nullable=True),
        sa.Column('source_filename', sa.String(255), nullable=False),
        sa.Column('source_file_type', sa.String(10), nullable=False),
        sa.Column('row_count', sa.Integer(), nullable=True),
        sa.Column('column_count', sa.Integer(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('failed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_code', sa.String(100), nullable=True),
        sa.Column('error_message', sa.String(1000), nullable=True),
        sa.Column('configuration_version', sa.String(50), nullable=True),
        sa.Column('pipeline_version', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('dataset_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('row_count', sa.Integer(), nullable=False),
        sa.Column('column_count', sa.Integer(), nullable=False),
        sa.Column('duplicate_row_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('memory_estimate_bytes', sa.Integer(), nullable=True),
        sa.Column('inferred_grain', sa.String(500), nullable=True),
        sa.Column('profile_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['profile_runs.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('column_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('physical_name', sa.String(500), nullable=False),
        sa.Column('normalized_key', sa.String(500), nullable=False),
        sa.Column('description', sa.String(2000), nullable=True),
        sa.Column('pandas_dtype', sa.String(50), nullable=False),
        sa.Column('refined_data_type', sa.String(50), nullable=False),
        sa.Column('statistics_json', postgresql.JSONB(), nullable=True),
        sa.Column('candidate_semantic_type', sa.String(100), nullable=True),
        sa.Column('candidate_column_role', sa.String(50), nullable=True),
        sa.Column('candidate_confidence', sa.Float(), nullable=True),
        sa.Column('confirmed_semantic_type', sa.String(100), nullable=True),
        sa.Column('confirmed_column_role', sa.String(50), nullable=True),
        sa.Column('schema_confidence', sa.Float(), nullable=True),
        sa.Column('identifier_score', sa.Float(), nullable=True),
        sa.Column('is_grain_key', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('mandatory', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('mandatory_source', sa.String(30), nullable=True),
        sa.Column('expected_unique', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('expected_unique_source', sa.String(30), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['profile_runs.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('secondary_domain_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('domain_name', sa.String(100), nullable=True),
        sa.Column('rank', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('evidence_json', postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['profile_runs.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('column_classifications',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('column_profile_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('primary_category', sa.String(100), nullable=False),
        sa.Column('secondary_categories_json', postgresql.JSONB(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('evidence_json', postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['profile_runs.id']),
        sa.ForeignKeyConstraint(['column_profile_id'], ['column_profiles.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('hierarchy_chains',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('template_key', sa.String(200), nullable=True),
        sa.Column('status', sa.String(30), nullable=False),
        sa.Column('average_confidence', sa.Float(), nullable=True),
        sa.Column('level_columns_json', postgresql.JSONB(), nullable=True),
        sa.Column('warnings_json', postgresql.JSONB(), nullable=True),
        sa.Column('algorithm_version', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['profile_runs.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('hierarchy_edges',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chain_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('parent_column', sa.String(500), nullable=False),
        sa.Column('child_column', sa.String(500), nullable=False),
        sa.Column('distinct_child_count', sa.Integer(), nullable=False),
        sa.Column('mapped_child_count', sa.Integer(), nullable=False),
        sa.Column('violating_child_count', sa.Integer(), nullable=False),
        sa.Column('fd_consistency', sa.Float(), nullable=False),
        sa.Column('mapping_coverage', sa.Float(), nullable=False),
        sa.Column('edge_confidence', sa.Float(), nullable=False),
        sa.Column('status', sa.String(30), nullable=False),
        sa.Column('conflict_samples_json', postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(['chain_id'], ['hierarchy_chains.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('quality_assessments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('dimension', sa.String(50), nullable=False),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('display_score', sa.Float(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('assessed_count', sa.Integer(), nullable=True),
        sa.Column('violation_count', sa.Integer(), nullable=True),
        sa.Column('weight', sa.Float(), nullable=True),
        sa.Column('evidence_json', postgresql.JSONB(), nullable=True),
        sa.Column('reason', sa.String(500), nullable=True),
        sa.Column('formula_version', sa.String(20), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['profile_runs.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('readiness_assessments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('assessment_type', sa.String(30), nullable=False),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('strengths_json', postgresql.JSONB(), nullable=True),
        sa.Column('blockers_json', postgresql.JSONB(), nullable=True),
        sa.Column('recommendations_json', postgresql.JSONB(), nullable=True),
        sa.Column('evidence_json', postgresql.JSONB(), nullable=True),
        sa.Column('weight_profile_version', sa.String(20), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['profile_runs.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('chart_specifications',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chart_key', sa.String(200), nullable=False),
        sa.Column('category', sa.String(20), nullable=False),
        sa.Column('chart_type', sa.String(20), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('specification_json', postgresql.JSONB(), nullable=True),
        sa.Column('aggregated_data_json', postgresql.JSONB(), nullable=True),
        sa.Column('rank', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['profile_runs.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('rule_definitions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rule_key', sa.String(200), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('domain', sa.String(50), nullable=False),
        sa.Column('secondary_domain', sa.String(100), nullable=True),
        sa.Column('definition_json', postgresql.JSONB(), nullable=False),
        sa.Column('source', sa.String(30), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='active'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('rule_suggestions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('suggested_definition_json', postgresql.JSONB(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='proposed'),
        sa.Column('evidence_json', postgresql.JSONB(), nullable=True),
        sa.Column('comment', sa.String(1000), nullable=True),
        sa.Column('rejection_reason', sa.String(1000), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['profile_runs.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('rule_evaluations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rule_definition_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('records_checked', sa.Integer(), nullable=False),
        sa.Column('pass_count', sa.Integer(), nullable=False),
        sa.Column('fail_count', sa.Integer(), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('evidence_json', postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['profile_runs.id']),
        sa.ForeignKeyConstraint(['rule_definition_id'], ['rule_definitions.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table('drill_down_cubes',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chart_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('hierarchy_chain_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('level_column', sa.String(500), nullable=False),
        sa.Column('level_depth', sa.Integer(), nullable=False),
        sa.Column('dimension_path_json', postgresql.JSONB(), nullable=True),
        sa.Column('aggregated_data_json', postgresql.JSONB(), nullable=False),
        sa.Column('metric_column', sa.String(500), nullable=False),
        sa.Column('aggregation', sa.String(20), nullable=False),
        sa.Column('record_count', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['profile_runs.id']),
        sa.ForeignKeyConstraint(['chart_id'], ['chart_specifications.id']),
        sa.ForeignKeyConstraint(['hierarchy_chain_id'], ['hierarchy_chains.id']),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('drill_down_cubes')
    op.drop_table('rule_evaluations')
    op.drop_table('rule_suggestions')
    op.drop_table('rule_definitions')
    op.drop_table('chart_specifications')
    op.drop_table('readiness_assessments')
    op.drop_table('quality_assessments')
    op.drop_table('hierarchy_edges')
    op.drop_table('hierarchy_chains')
    op.drop_table('column_classifications')
    op.drop_table('secondary_domain_results')
    op.drop_table('column_profiles')
    op.drop_table('dataset_profiles')
    op.drop_table('profile_runs')
