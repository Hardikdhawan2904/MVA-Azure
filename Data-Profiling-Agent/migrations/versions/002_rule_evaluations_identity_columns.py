"""Rule evaluations: add identity columns, make rule_definition_id optional

Only approved-suggestion-sourced rules ever get a row in rule_definitions —
YAML domain rules and ad-hoc request rules never did and never will. The
original NOT NULL FK made it impossible to persist evaluations for those,
which is why nothing ever wrote to this table. rule_key/rule_type/source are
added directly on the row so evaluations remain identifiable and queryable
without requiring a join to rule_definitions.

Revision ID: 002
Revises: 001
Create Date: 2026-07-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# The connecting role's search_path doesn't include agent2 (this project's tables were
# originally created via pg_restore + ALTER TABLE ... SET SCHEMA, not by literally running
# 001 against this search_path) — every op call here must be explicitly schema-qualified.
SCHEMA = 'agent2'


def upgrade() -> None:
    op.add_column('rule_evaluations', sa.Column('rule_key', sa.String(200), nullable=False, server_default=''), schema=SCHEMA)
    op.add_column('rule_evaluations', sa.Column('rule_type', sa.String(50), nullable=False, server_default=''), schema=SCHEMA)
    op.add_column('rule_evaluations', sa.Column('source', sa.String(30), nullable=False, server_default=''), schema=SCHEMA)
    op.alter_column('rule_evaluations', 'rule_key', server_default=None, schema=SCHEMA)
    op.alter_column('rule_evaluations', 'rule_type', server_default=None, schema=SCHEMA)
    op.alter_column('rule_evaluations', 'source', server_default=None, schema=SCHEMA)
    op.alter_column('rule_evaluations', 'rule_definition_id', nullable=True, schema=SCHEMA)


def downgrade() -> None:
    op.alter_column('rule_evaluations', 'rule_definition_id', nullable=False, schema=SCHEMA)
    op.drop_column('rule_evaluations', 'source', schema=SCHEMA)
    op.drop_column('rule_evaluations', 'rule_type', schema=SCHEMA)
    op.drop_column('rule_evaluations', 'rule_key', schema=SCHEMA)
