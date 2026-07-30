"""tests/test_knowledge_update_service.py — KnowledgeUpdateTool._validate()
regression test.

Real bug, found live during a handover code review: _validate() imported
`from tools.schemas import KPIDefinition` — a module that doesn't exist
anywhere in this repo (the real path is `app.services.schemas`). Every
call raised ModuleNotFoundError, silently swallowed by the surrounding
`except Exception`, so auto-KPI-generation was completely dead for every
domain and every query that named an unrecognized KPI alias — with zero
test coverage to catch it. This file exists so it can never regress
silently again.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.tools.knowledge_update_service import KnowledgeUpdateTool


def _tool():
    return KnowledgeUpdateTool(rule_engine=None)  # rule_engine unused by _validate()


def test_validate_accepts_a_structurally_sound_candidate():
    tool = _tool()
    candidate = {
        "label": "Net Promoter Score (NPS)",
        "formula": "% Promoters - % Detractors",
        "unit": "score",
        "category": "customer",
        "higher_is_better": True,
        "dependencies": [],
    }
    is_valid, reason = tool._validate(candidate)
    assert is_valid is True, f"expected valid, got: {reason}"
    assert reason == "OK"


def test_validate_rejects_unknown_category_with_a_clear_reason():
    tool = _tool()
    candidate = {
        "label": "Some KPI", "formula": "...", "unit": "%",
        "category": "not_a_real_category", "higher_is_better": True, "dependencies": [],
    }
    is_valid, reason = tool._validate(candidate)
    assert is_valid is False
    assert "category must be one of" in reason


def test_validate_rejects_missing_required_field():
    tool = _tool()
    candidate = {"formula": "...", "unit": "%", "category": "premium", "higher_is_better": True}  # no label
    is_valid, reason = tool._validate(candidate)
    assert is_valid is False
    # The real regression check: this must be a Pydantic validation error,
    # never "No module named 'tools'" (the bug this test locks in against).
    assert "tools" not in reason.lower() or "no module named" not in reason.lower()


def test_static_fallback_kpi_passes_validation_end_to_end():
    """The exact no-LLM path _validate() is actually exercised through in
    production — update() with no Azure OpenAI client configured. Forced
    deterministically (not relying on AZURE_OPENAI_API_KEY being unset in
    this environment) so this test's outcome never depends on real env state."""
    tool = KnowledgeUpdateTool(rule_engine=_FakeRuleEngine())
    tool._client = None
    result = tool.update("net_promoter_score")
    assert result["success"] is True, result["reason"]
    assert result["kpi_key"] == "net_promoter_score"


class _FakeRuleEngine:
    def __init__(self):
        self._kpis = {}

    def add_kpi(self, key, definition):
        if key in self._kpis:
            return False
        self._kpis[key] = definition
        return True
