"""
Rule Suggestion Agent config loader — reads agent.yaml once (LRU-cached).

agent.yaml is the single source of truth for:
  - LLM settings      (temperature, max_tokens, recursion_limit)
  - Entry point       (which node the graph starts at)
  - Agent tools       (names only — callables resolved via TOOL_REGISTRY)
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


def _pipeline() -> dict[str, Any]:
    return load_agent_config()["agent"]["langgraph"]["pipeline"]


def get_entry_point() -> str:
    return _pipeline()["entry_point"]


def get_agent_tool_names() -> list[str]:
    return _pipeline()["agent_tools"]


def get_llm_config() -> dict[str, Any]:
    return load_agent_config()["agent"]["llm"]
