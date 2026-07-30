"""Rule-suggestion sub-agent — genuine ReAct tool-calling LangGraph, unlike the
single-shot pre-computed-context call schema intelligence still uses.

Topology:
  agent ──(tool calls?)──► tools ──► agent   (loop)
  agent ──(no tool calls)─► finalize ──► END

See nodes/pipeline.py for the three node implementations, tools.py for what
the agent can call, and agent.yaml/config.py for its declarative persona and
LLM/tool config.
"""

from __future__ import annotations

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.agents.rule_suggestion_agent.config import get_entry_point, get_llm_config
from app.agents.rule_suggestion_agent.nodes.pipeline import RuleSuggestionAgentNodes
from app.agents.rule_suggestion_agent.state import RuleSuggestionAgentState

SYSTEM_PROMPT = """You are a business rule discovery system with tools available to \
investigate the dataset before proposing rules.

Supported rule types — the ONLY types the evaluation engine can execute:
  - non_null: requires target_column
  - expected_unique: requires target_column
  - regex_match: requires target_column, pattern
  - allowed_values: requires target_column, values (list of strings)
  - numeric_range: requires target_column, and at least one of min_value/max_value
  - column_comparison: requires left_column, operator (>=, >, <=, <, ==, !=), right_column
  - conditional_required: requires condition_column, condition_value, required_column

Rules:
- target_column / left_column / right_column / condition_column / required_column MUST
  be exact column names from the dataset. Never invent a column name.
- Before finalizing, call check_existing_rules to see what's already active — never
  propose a rule that duplicates one already in that list.
- Use get_column_statistics or check_value_distribution if a column's summary alone
  isn't enough to confidently propose a rule for it.
- rule_key must be a short, unique, snake_case identifier.
- Propose at most 5 rules. Confidence between 0.0 and 1.0.
- When you have enough information, respond with ONLY a JSON object (no markdown
  fences, no extra text) matching this exact schema and STOP calling tools:
{
  "suggestions": [
    {
      "rule_key": "...", "type": "...", "description": "...", "reasoning": "...",
      "confidence": 0.0, "severity": "low|medium|high", "target_columns": ["..."],
      "target_column": null, "pattern": null, "values": [], "min_value": null,
      "max_value": null, "inclusive_min": true, "inclusive_max": true,
      "left_column": null, "operator": null, "right_column": null,
      "condition_column": null, "condition_value": null, "required_column": null
    }
  ]
}
Only populate fields relevant to the chosen type."""


def _build_columns_summary(col_profiles: list) -> str:
    """No truncation -- see feature_target_agent/graph.py's
    _build_columns_summary for why a positional cap here silently hides
    real columns from the agent on wide datasets."""
    lines = []
    for p in col_profiles:
        lines.append(
            f"- {p.physical_name}: type={p.pandas_dtype}, "
            f"nulls={p.null_ratio:.2%}, distinct={p.distinct_count}, "
            f"samples={p.representative_values[:3]}"
        )
    return "\n".join(lines)


def build_rule_suggestion_graph(
    llm_model: str, api_key: str, azure_endpoint: str, api_version: str,
    col_profiles: list, df, existing_rule_keys: list[str],
):
    """Compile the rule-suggestion StateGraph. Built fresh per run (its nodes
    close over this run's tools), unlike the top-level pipeline graph which
    is stateless and compiled once at import time."""
    nodes = RuleSuggestionAgentNodes(
        llm_model, api_key, azure_endpoint, api_version, col_profiles, df, existing_rule_keys
    )

    g = StateGraph(RuleSuggestionAgentState)
    g.add_node("agent", nodes.call_model)
    g.add_node("tools", ToolNode(nodes.tools))
    g.add_node("finalize", nodes.finalize)
    g.set_entry_point(get_entry_point())
    g.add_conditional_edges("agent", nodes.should_continue, {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)

    return g.compile()


def run_rule_suggestion_agent(
    llm_model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    col_profiles: list,
    df: pd.DataFrame,
    primary_domain: str,
    secondary_domain: str | None,
    existing_rule_keys: list[str],
) -> list[dict]:
    """Run the ReAct rule-suggestion agent and return SuggestedRule-shaped dicts,
    matching the exact output shape RuleRepository.persist_suggestions expects."""
    compiled = build_rule_suggestion_graph(
        llm_model, api_key, azure_endpoint, api_version, col_profiles, df, existing_rule_keys
    )

    columns_summary = _build_columns_summary(col_profiles)
    human_content = (
        f"Primary Domain: {primary_domain}\n"
        f"Secondary Domain: {secondary_domain or 'unknown'}\n\n"
        f"Column profiles:\n{columns_summary}\n\n"
        "Investigate as needed using your tools, then propose up to 5 business rules."
    )

    initial_state: RuleSuggestionAgentState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_content)],
        "suggestions": [],
    }

    llm_config = get_llm_config()
    result = compiled.invoke(initial_state, config={"recursion_limit": llm_config["recursion_limit"]})
    return result.get("suggestions", [])
