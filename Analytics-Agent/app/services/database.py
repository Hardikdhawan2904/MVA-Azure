"""
app/services/database.py — PostgreSQL persistence for conversation memory.

Mirrors Schema-Intelligence-Layer/app/services/database.py's raw-psycopg2
style (same shared Postgres instance, one schema per agent) rather than
Agent 2's SQLAlchemy+Alembic — a single table with no evolving relational
structure doesn't need a migration framework.

Unlike Agent 1's init_db() (which raises and crashes startup on a DB
error), failures here are always non-fatal — conversation memory is an
enhancement, not core to answering a single question. Callers (init_db()
from app/main.py's lifespan, MemoryManager) catch and log rather than
propagate, so a Postgres outage degrades to in-process-only, non-persistent
memory instead of taking the service down or failing every /analyze call.
"""

import json
import logging
from typing import Any

import psycopg2
import psycopg2.extras

from app.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id SERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    user_query TEXT NOT NULL,
    intent_detected TEXT,
    kpi_name TEXT,
    filters_applied TEXT,
    tools_used TEXT,
    key_evidence TEXT,
    agent_response_summary TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_conversation_turns_cid ON conversation_turns(conversation_id);
"""

INSERT_TURN_SQL = """
INSERT INTO conversation_turns (
    conversation_id, turn_id, user_query, intent_detected, kpi_name,
    filters_applied, tools_used, key_evidence, agent_response_summary
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

SELECT_TURNS_SQL = """
SELECT turn_id, user_query, intent_detected, kpi_name, filters_applied,
       tools_used, key_evidence, agent_response_summary, created_at
FROM conversation_turns
WHERE conversation_id = %s
ORDER BY turn_id DESC
LIMIT %s;
"""


def _get_connection() -> psycopg2.extensions.connection:
    """Create a new PostgreSQL connection scoped to the agent3 schema."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        cursor_factory=psycopg2.extras.RealDictCursor,
        options="-c search_path=agent3",
    )


def init_db() -> None:
    """Create the agent3 schema/table if they don't exist. Never raises —
    logs and returns on failure, so a Postgres outage doesn't take the
    whole service down at startup."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute("CREATE SCHEMA IF NOT EXISTS agent3")
        cur.execute(CREATE_TABLE_SQL)
        cur.execute(CREATE_INDEX_SQL)
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Conversation memory DB initialised at {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB} (agent3)")
    except Exception as e:
        logger.error(f"Failed to initialise conversation memory DB — persistence disabled for this run: {e}")


def insert_turn(conversation_id: str, turn: dict[str, Any]) -> None:
    """Persist one conversation turn. Raises on failure — callers
    (MemoryManager.add_turn) decide how to handle it, since they already
    have their own try/except around every DB call."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(INSERT_TURN_SQL, (
            conversation_id,
            turn["turn_id"],
            turn["user_query"],
            turn.get("intent_detected"),
            turn.get("kpi_name"),
            json.dumps(turn.get("filters_applied") or {}),
            json.dumps(turn.get("tools_used") or []),
            json.dumps(turn.get("key_evidence") or {}),
            turn.get("agent_response_summary"),
        ))
        conn.commit()
        cur.close()
    finally:
        conn.close()


def load_turns(conversation_id: str, limit: int) -> list[dict[str, Any]]:
    """Load the most recent `limit` turns for conversation_id, oldest
    first (matching MemoryManager._history's append-order convention).
    Raises on failure — callers decide how to handle it."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(SELECT_TURNS_SQL, (conversation_id, limit))
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    turns = []
    for row in reversed(rows):  # DESC-fetched (newest first) -> chronological
        turns.append({
            "turn_id": row["turn_id"],
            "user_query": row["user_query"],
            "intent_detected": row["intent_detected"],
            "kpi_name": row["kpi_name"],
            "filters_applied": json.loads(row["filters_applied"]) if row["filters_applied"] else {},
            "tools_used": json.loads(row["tools_used"]) if row["tools_used"] else [],
            "key_evidence": json.loads(row["key_evidence"]) if row["key_evidence"] else {},
            "agent_response_summary": row["agent_response_summary"],
            "timestamp": row["created_at"].isoformat(),
        })
    return turns
