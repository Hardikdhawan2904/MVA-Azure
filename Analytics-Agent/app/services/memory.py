"""
app/services/memory.py — Conversation History & Memory Manager

Implements sliding window memory for the Analytics Agent, backed by
Postgres (app/services/database.py) instead of an in-process-only list.
This matters more than it sounds: run_analytics_graph() builds a fresh
AnalyticsGraphNodes (and therefore a fresh MemoryManager) on every single
HTTP request, so without real persistence there was never any way for two
separate POST /analyze calls to share history — restart or no restart.
conversation_id is what ties multiple requests back to the same
MemoryManager's history.

Config is read from agent.yaml → memory section:
  - max_turns: 10
  - include_evidence: true
  - summarise_after_turns: 10

DB failures (load or save) are caught and logged, never raised — a
Postgres outage degrades this to in-process-only, non-persistent memory
for the life of the request, rather than failing the question being asked.
"""

import logging
from datetime import datetime, timezone

from app.config import MEMORY_CONFIG
from app.services import database

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Sliding window conversation history manager, persisted per
    conversation_id in Postgres (agent3.conversation_turns).

    Each turn stores:
    - turn_id
    - user_query
    - intent_detected
    - kpi_name
    - filters_applied
    - tools_used
    - key_evidence  (numbers only — no DataFrames)
    - agent_response_summary
    - timestamp
    """

    def __init__(self, conversation_id: str):
        self.conversation_id    = conversation_id
        self.max_turns          = MEMORY_CONFIG.get("max_turns", 10)
        self.include_evidence   = MEMORY_CONFIG.get("include_evidence", True)
        self.summarise_after    = MEMORY_CONFIG.get("summarise_after_turns", 10)
        self._history: list[dict] = []
        self._turn_counter      = 0

        try:
            self._history = database.load_turns(conversation_id, limit=self.max_turns)
            self._turn_counter = self._history[-1]["turn_id"] if self._history else 0
            logger.info(f"Memory: loaded {len(self._history)} past turns for conversation={conversation_id}")
        except Exception as e:
            logger.warning(f"Memory: could not load history for conversation={conversation_id}: {e} — starting fresh.")

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_turn(
        self,
        user_query: str,
        intent: str,
        kpi_name: str,
        filters: dict,
        tools_used: list[str],
        evidence: dict,
        response_summary: str,
    ) -> int:
        """
        Record a completed Q&A turn.

        Returns the turn_id assigned.
        """
        self._turn_counter += 1

        # Strip non-serialisable items from evidence (e.g. DataFrames)
        safe_evidence = self._sanitise_evidence(evidence)

        turn = {
            "turn_id":               self._turn_counter,
            "user_query":            user_query,
            "intent_detected":       intent,
            "kpi_name":              kpi_name,
            "filters_applied":       filters,
            "tools_used":            tools_used,
            "key_evidence":          safe_evidence if self.include_evidence else {},
            "agent_response_summary":response_summary[:500],  # Truncate for context efficiency
            "timestamp":             datetime.now(timezone.utc).isoformat(),
        }

        self._history.append(turn)
        logger.info(f"Memory: turn {self._turn_counter} recorded — intent='{intent}', kpi='{kpi_name}'")

        # Sliding window — remove oldest if over limit
        if len(self._history) > self.max_turns:
            dropped = self._history.pop(0)
            logger.debug(f"Memory: dropped turn {dropped['turn_id']} (window full)")

        # Auto-summarise if threshold reached
        if len(self._history) >= self.summarise_after:
            self._summarise()

        try:
            database.insert_turn(self.conversation_id, turn)
        except Exception as e:
            logger.warning(f"Memory: could not persist turn {self._turn_counter} for conversation="
                            f"{self.conversation_id}: {e} — this turn only lives in-process.")

        return self._turn_counter

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_history(self) -> list[dict]:
        """Return full conversation history (within window)."""
        return self._history.copy()

    def get_context_for_llm(self) -> str:
        """
        Format history as a compact string for LLM context injection.
        Only includes fields relevant to the LLM narrator.
        """
        if not self._history:
            return "No prior conversation history."

        lines = ["=== Prior Conversation Context ==="]
        for turn in self._history[-5:]:   # Last 5 turns only for LLM context
            lines.append(
                f"\nTurn {turn['turn_id']} | {turn['timestamp'][:10]}"
                f"\nQ: {turn['user_query']}"
                f"\nIntent: {turn['intent_detected']} | KPI: {turn['kpi_name']}"
            )
            if turn.get("key_evidence"):
                ev = turn["key_evidence"]
                ev_str = ", ".join(
                    f"{k}={v:,.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in list(ev.items())[:6]
                )
                lines.append(f"Evidence: {ev_str}")
            lines.append(f"A: {turn['agent_response_summary'][:200]}...")

        return "\n".join(lines)

    def get_last_kpi(self) -> str | None:
        """Return the KPI from the most recent turn (for context carryover)."""
        return self._history[-1]["kpi_name"] if self._history else None

    def get_last_filters(self) -> dict:
        """Return filters from the most recent turn."""
        return self._history[-1]["filters_applied"] if self._history else {}

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitise_evidence(evidence: dict) -> dict:
        """Remove non-JSON-serialisable objects from evidence dict."""
        safe = {}
        for k, v in evidence.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                safe[k] = v
            elif isinstance(v, dict):
                safe[k] = MemoryManager._sanitise_evidence(v)
            elif isinstance(v, list) and all(isinstance(i, (str, int, float)) for i in v):
                safe[k] = v
        return safe

    def _summarise(self):
        """Compress old history into a brief summary (placeholder for LLM summarisation)."""
        logger.info(f"Memory: compressing {len(self._history)} turns into summary")
        # Future enhancement: call the LLM to summarise old turns into one summary entry
        # For now, just keep the sliding window approach
