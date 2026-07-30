"""Real LangChain tools for the rule-suggestion agent. Built per-run via a
factory function since each run's column profiles/dataframe/active rules
differ — there's no shared global state a plain module-level @tool could
safely close over.
"""

import pandas as pd
from langchain_core.tools import tool

from app.agents._shared.column_tools import build_get_column_statistics_tool, tools_to_registry

__all__ = ["build_rule_suggestion_tools", "tools_to_registry"]


def build_rule_suggestion_tools(col_profiles: list, df: pd.DataFrame, existing_rule_keys: list[str]) -> list:
    """
    Tools available to the rule-suggestion LLM. get_column_statistics mirrors
    the equivalent feature_target_agent tool exactly (shared implementation).
    check_existing_rules is a genuinely new capability this pipeline didn't
    have before: checking which rules already exist (YAML-configured +
    previously approved) so it doesn't propose duplicates of rules that are
    already active for this domain.
    """
    @tool
    def check_value_distribution(column_name: str) -> str:
        """Get the full distinct-value distribution (value counts) for a column
        — useful before proposing an allowed_values rule, to see every distinct
        value rather than just the first few samples."""
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in the dataset."
        counts = df[column_name].value_counts(dropna=True).head(20)
        return f"Value distribution for '{column_name}' (top 20): {counts.to_dict()}"

    @tool
    def check_existing_rules() -> str:
        """List rule_keys that are already active for this dataset's domain
        (from YAML config or previously-approved suggestions). Never propose a
        new rule that duplicates one already in this list."""
        if not existing_rule_keys:
            return "No rules are currently active for this domain."
        return f"Already-active rule keys (do not duplicate these): {existing_rule_keys}"

    return [build_get_column_statistics_tool(col_profiles), check_value_distribution, check_existing_rules]
