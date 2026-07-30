"""Real LangChain tools for the feature-target agent. Built per-run via a
factory function since each run's column profiles/dataframe differ — there's
no shared global state a plain module-level @tool could safely close over.
"""

import pandas as pd
from langchain_core.tools import tool

from app.agents._shared.column_tools import build_get_column_statistics_tool, tools_to_registry

__all__ = ["build_feature_target_tools", "tools_to_registry"]


def build_feature_target_tools(col_profiles: list, df: pd.DataFrame) -> list:
    """
    Tools available to the feature-target LLM. get_column_statistics mirrors
    the equivalent rule_suggestion_agent tool exactly (shared implementation);
    check_value_distribution is agent-specific — the agent already receives
    every column's role/semantic-type/statistics summary up front, this is
    only for drilling into a specific candidate target column before
    committing to a problem_type (e.g. confirming it's genuinely
    binary/categorical rather than continuous).
    """
    @tool
    def check_value_distribution(column_name: str) -> str:
        """Get the full distinct-value distribution (value counts) for a
        column — useful before deciding a candidate target's problem_type,
        to see whether it's genuinely binary/categorical (classification) or
        continuous (regression) rather than guessing from samples alone."""
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in the dataset."
        counts = df[column_name].value_counts(dropna=True).head(20)
        return f"Value distribution for '{column_name}' (top 20): {counts.to_dict()}"

    return [build_get_column_statistics_tool(col_profiles), check_value_distribution]
