"""
Data Profiling Agent config loader — reads agent.yaml once (LRU-cached).

agent.yaml is the single source of truth for this pipeline's declarative
metadata: persona/role/constraints, and the quality-gate threshold that
decides whether the expensive downstream stages (suggestions, readiness,
charts) run at all. LLM model/credentials stay in Settings (.env) since
they're shared infrastructure config, not agent-specific — see
rule_suggestion_agent/config.py for the one sub-agent with its own
LLM-tuning knobs (temperature, max_tokens).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def load_agent_config() -> dict[str, Any]:
    path = Path(__file__).parent / "agent.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_entry_point() -> str:
    return load_agent_config()["agent"]["langgraph"]["pipeline"]["entry_point"]


def get_quality_gate_min_score() -> float:
    return load_agent_config()["agent"]["quality_gate"]["min_overall_score"]
