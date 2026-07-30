"""
Orchestration Agent config loader — reads agent.yaml once (LRU-cached).

agent.yaml is the single source of truth for this agent's declarative
metadata: persona/role/constraints and the graph's entry point. Downstream
service URLs and request timeouts stay in app/config.py (Settings) since
they're deployment/infra config, not agent persona — that separation
mirrors how agent.yaml never encodes secrets or environment-specific values.
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
