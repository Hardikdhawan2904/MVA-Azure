"""Shared LangChain tool factory used by both feature_target_agent and
rule_suggestion_agent — get_column_statistics was defined identically
(same docstring, same body) in both agents' own tools.py. Built per-run
via a factory function since each run's column profiles differ — there's
no shared global state a plain module-level @tool could safely close over.

check_value_distribution is deliberately NOT here despite being structurally
similar in both agents — its docstring (the LLM-facing guidance on when to
call it) is genuinely different per agent (feature_target_agent frames it
around confirming a problem_type; rule_suggestion_agent frames it around
proposing allowed_values rules), so each agent keeps its own copy rather
than sharing a description that would blur that framing.
"""

from typing import Any

from langchain_core.tools import tool


def build_get_column_statistics_tool(col_profiles: list):
    """Column-statistics lookup tool — identical behavior in both agents."""
    profiles_by_name = {p.physical_name: p for p in col_profiles}

    @tool
    def get_column_statistics(column_name: str) -> str:
        """Get the full statistics for a specific column by its physical name."""
        profile = profiles_by_name.get(column_name)
        if profile is None:
            return f"Column '{column_name}' not found. Available columns: {list(profiles_by_name.keys())}"
        return str(profile.to_statistics_dict())

    return get_column_statistics


def tools_to_registry(tools: list) -> dict[str, Any]:
    """Build a name->tool lookup so agent.yaml's agent_tools list (names only)
    can be resolved into actual callables at graph-build time."""
    return {t.name: t for t in tools}
