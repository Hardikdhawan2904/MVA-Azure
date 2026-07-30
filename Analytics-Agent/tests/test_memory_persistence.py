"""tests/test_memory_persistence.py — Tests for Postgres-backed conversation
memory (app/services/database.py + app/services/memory.py).

Runs against the real shared Postgres instance, not a mock — matching this
codebase's established practice (test_ml_persistence.py does the same
against real persisted models). Each test uses its own uuid4()
conversation_id so tests can't collide with each other or with real
conversations.

Skips cleanly if Postgres isn't reachable — this is a real infra
dependency (the native Shared-Postgres instance, see
Shared-Postgres/README.md), not something to fake.
"""

import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import pytest

from app.services import database
from app.services.memory import MemoryManager
from app.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD


def _postgres_reachable() -> bool:
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST, port=POSTGRES_PORT, dbname=POSTGRES_DB,
            user=POSTGRES_USER, password=POSTGRES_PASSWORD, connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Shared Postgres not reachable — start the native instance, see Shared-Postgres/README.md",
)


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    database.init_db()


def _sample_turn(turn_id: int, **overrides) -> dict:
    turn = {
        "turn_id": turn_id,
        "user_query": f"query {turn_id}",
        "intent_detected": "show_kpi",
        "kpi_name": "gross_written_premium",
        "filters_applied": {"fiscal_year": "FY2025"},
        "tools_used": ["RuleEngine", "SQLTool"],
        "key_evidence": {"actual": 100.0 * turn_id},
        "agent_response_summary": f"response {turn_id}",
    }
    turn.update(overrides)
    return turn


# ── app/services/database.py ─────────────────────────────────────────────────

def test_load_turns_empty_for_unknown_conversation():
    assert database.load_turns(str(uuid.uuid4()), limit=10) == []


def test_insert_and_load_round_trip_preserves_fields():
    cid = str(uuid.uuid4())
    database.insert_turn(cid, _sample_turn(1, user_query="Show GWP for FY2025", kpi_name="gross_written_premium"))

    turns = database.load_turns(cid, limit=10)
    assert len(turns) == 1
    assert turns[0]["user_query"] == "Show GWP for FY2025"
    assert turns[0]["kpi_name"] == "gross_written_premium"
    assert turns[0]["filters_applied"] == {"fiscal_year": "FY2025"}
    assert turns[0]["tools_used"] == ["RuleEngine", "SQLTool"]
    assert turns[0]["key_evidence"] == {"actual": 100.0}


def test_load_turns_returns_chronological_order():
    cid = str(uuid.uuid4())
    for i in (1, 2, 3):
        database.insert_turn(cid, _sample_turn(i))

    turns = database.load_turns(cid, limit=10)
    assert [t["turn_id"] for t in turns] == [1, 2, 3]


def test_load_turns_respects_limit_keeping_most_recent():
    cid = str(uuid.uuid4())
    for i in range(1, 6):
        database.insert_turn(cid, _sample_turn(i))

    turns = database.load_turns(cid, limit=2)
    assert [t["turn_id"] for t in turns] == [4, 5]


def test_different_conversations_do_not_leak_into_each_other():
    cid_a, cid_b = str(uuid.uuid4()), str(uuid.uuid4())
    database.insert_turn(cid_a, _sample_turn(1, user_query="conversation A"))
    database.insert_turn(cid_b, _sample_turn(1, user_query="conversation B"))

    turns_a = database.load_turns(cid_a, limit=10)
    turns_b = database.load_turns(cid_b, limit=10)
    assert [t["user_query"] for t in turns_a] == ["conversation A"]
    assert [t["user_query"] for t in turns_b] == ["conversation B"]


# ── app/services/memory.py — MemoryManager ────────────────────────────────────

def test_new_conversation_starts_empty():
    mm = MemoryManager(str(uuid.uuid4()))
    assert mm.get_history() == []
    assert mm.get_last_kpi() is None
    assert mm.get_last_filters() == {}


def test_add_turn_persists_and_is_visible_to_a_fresh_instance():
    """The core guarantee this whole feature exists for: two separate
    MemoryManager instances (i.e. two separate HTTP requests) sharing a
    conversation_id see the same history — this could never work before
    Postgres persistence, since run_analytics_graph() builds a fresh,
    empty MemoryManager on every single request."""
    cid = str(uuid.uuid4())

    first = MemoryManager(cid)
    first.add_turn(
        user_query="Show GWP for FY2025", intent="show_kpi", kpi_name="gross_written_premium",
        filters={"fiscal_year": "FY2025"}, tools_used=["RuleEngine"], evidence={"actual": 100.0},
        response_summary="GWP was 100",
    )

    second = MemoryManager(cid)  # simulates the next HTTP request
    assert second.get_last_kpi() == "gross_written_premium"
    assert second.get_last_filters() == {"fiscal_year": "FY2025"}
    assert len(second.get_history()) == 1


def test_filter_carryover_across_simulated_requests():
    """Reproduces the "what about EMEA?" follow-up scenario: the second
    request's filters should merge the first request's filters with
    whatever the new query explicitly mentions — this is
    main.py::detect_intent_and_filters's existing merge logic
    (nodes/pipeline.py), verified here at the memory layer it depends on."""
    cid = str(uuid.uuid4())

    first = MemoryManager(cid)
    first.add_turn(
        user_query="Show GWP for FY2025", intent="show_kpi", kpi_name="gross_written_premium",
        filters={"fiscal_year": "FY2025"}, tools_used=["RuleEngine"], evidence={},
        response_summary="...",
    )

    second = MemoryManager(cid)
    carried_filters = {**second.get_last_filters(), "region": "EMEA"}
    assert carried_filters == {"fiscal_year": "FY2025", "region": "EMEA"}


def test_get_context_for_llm_survives_a_fresh_instance():
    cid = str(uuid.uuid4())
    first = MemoryManager(cid)
    first.add_turn(
        user_query="Show GWP for FY2025", intent="show_kpi", kpi_name="gross_written_premium",
        filters={}, tools_used=[], evidence={}, response_summary="GWP was 100",
    )

    second = MemoryManager(cid)
    context = second.get_context_for_llm()
    assert "Show GWP for FY2025" in context
    assert "GWP was 100" in context


def test_db_failure_degrades_to_empty_history_not_a_crash(monkeypatch):
    def _boom(*args, **kwargs):
        raise psycopg2.OperationalError("simulated connection failure")

    monkeypatch.setattr(database, "load_turns", _boom)
    mm = MemoryManager(str(uuid.uuid4()))  # must not raise
    assert mm.get_history() == []


def test_add_turn_survives_db_failure(monkeypatch):
    def _boom(*args, **kwargs):
        raise psycopg2.OperationalError("simulated connection failure")

    monkeypatch.setattr(database, "insert_turn", _boom)
    mm = MemoryManager(str(uuid.uuid4()))
    turn_id = mm.add_turn(
        user_query="q", intent="show_kpi", kpi_name="x",
        filters={}, tools_used=[], evidence={}, response_summary="a",
    )  # must not raise
    assert turn_id == 1
    assert len(mm.get_history()) == 1  # still recorded in-process
