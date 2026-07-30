"""Feature-target sub-agent — genuine ReAct tool-calling LangGraph, same
shape as rule_suggestion_agent's. Given a business question (and optionally
an explicit target-column override), classifies every column into target /
feature / drop and recommends whether the question is better answered by an
ML or an LLM approach.

Topology:
  agent ──(tool calls?)──► tools ──► agent   (loop)
  agent ──(no tool calls)─► finalize ──► END

See nodes/pipeline.py for the three node implementations, tools.py for what
the agent can call, and agent.yaml/config.py for its declarative persona and
LLM/tool config.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.agents.feature_target_agent.config import get_entry_point, get_llm_config
from app.agents.feature_target_agent.nodes.pipeline import FeatureTargetAgentNodes
from app.agents.feature_target_agent.state import FeatureTargetAgentState

SYSTEM_PROMPT = """You are a data-analysis planning system. Given a business question and a \
dataset's column profiles, decide exactly how the dataset should be used to answer it.

Rules:
- target_column MUST be an exact physical column name from the dataset, or null if the
  question doesn't imply a single target column to predict/measure. Never invent a column name.
- time_column MUST be an exact physical column name (usually role=temporal_dimension), or null.
- problem_type is one of: classification, regression, unknown. Use the target column's role,
  cardinality, and value distribution (use check_value_distribution if unsure) to decide —
  a low-cardinality target is classification, a continuous numeric target is regression.
- recommended_approach is one of: ml, llm, hybrid, unknown. Choose "ml" when the question maps
  onto predicting/explaining a structured numeric or categorical target from structured
  features. Choose "llm" when the question is interpretive/open-ended (summarize, explain why,
  compare narratively) or depends heavily on text_field columns rather than a clean numeric
  target. Choose "hybrid" when both apply.
- Every entry in feature_columns and drop_columns MUST use an exact physical column name from
  the dataset. Identifier/grain-key columns (role=identifier, or is_grain_key=true) almost
  always belong in drop_columns, not feature_columns.
- suggested_aggregation (for feature_columns) is one of: sum, mean, max, min, count, null —
  only meaningful when the feature is at a lower grain than the target and needs aggregating up.
- Use get_column_statistics or check_value_distribution if a column's summary alone isn't
  enough to confidently classify it.
- confidence is a float between 0.0 and 1.0 reflecting how directly the business question maps
  onto this dataset's columns.
- When you have enough information, respond with ONLY a JSON object (no markdown fences, no
  extra text) matching this exact schema and STOP calling tools:
{
  "target_column": null, "problem_type": "unknown", "time_column": null,
  "recommended_approach": "unknown", "approach_reasoning": "...",
  "feature_columns": [
    {"column": "...", "role": "...", "usefulness": "high|medium|low",
     "suggested_aggregation": null, "reasoning": "..."}
  ],
  "drop_columns": [{"column": "...", "reason": "..."}],
  "confidence": 0.0
}"""


def _build_columns_summary(col_profiles: list, sem_candidates: list, grain_result) -> str:
    """One compact line per column, no truncation -- a silent [:60] cap
    here previously meant the LLM could never see (let alone pick) a
    target column past the 60th, regardless of how well it matched the
    question. Confirmed live: on a real 141-column dataset,
    underwriting_result_actual sits at index 88 and was invisible to the
    agent every time it didn't happen to reach for a lookup tool on its
    own initiative -- a real, reproducible cause of target selection
    failing, not generic LLM non-determinism. A few hundred short lines
    is well within normal context limits; nothing here needs capping."""
    grain_columns = set(grain_result.grain_columns)
    id_scores = {c.column_name: c.identifier_score for c in grain_result.identifier_candidates}
    lines = []
    for i, p in enumerate(col_profiles):
        cand = sem_candidates[i] if i < len(sem_candidates) else None
        role = cand.candidate_column_role.value if cand else "unknown"
        semantic_type = cand.candidate_semantic_type if cand else None
        lines.append(
            f"- {p.physical_name}: role={role}, semantic_type={semantic_type}, "
            f"dtype={p.pandas_dtype}, nulls={p.null_ratio:.2%}, distinct={p.distinct_count}, "
            f"is_grain_key={p.physical_name in grain_columns}, "
            f"identifier_score={id_scores.get(p.physical_name, 0.0):.2f}, "
            f"samples={p.representative_values[:3]}"
        )
    return "\n".join(lines)


def build_feature_target_graph(
    llm_model: str, api_key: str, azure_endpoint: str, api_version: str, col_profiles: list, df
):
    """Compile the feature-target StateGraph. Built fresh per run (its nodes
    close over this run's tools), unlike the top-level pipeline graph which
    is stateless and compiled once at import time."""
    nodes = FeatureTargetAgentNodes(llm_model, api_key, azure_endpoint, api_version, col_profiles, df)

    g = StateGraph(FeatureTargetAgentState)
    g.add_node("agent", nodes.call_model)
    g.add_node("tools", ToolNode(nodes.tools))
    g.add_node("finalize", nodes.finalize)
    g.set_entry_point(get_entry_point())
    g.add_conditional_edges("agent", nodes.should_continue, {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)

    return g.compile()


def run_feature_target_agent(
    llm_model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    col_profiles: list,
    sem_candidates: list,
    grain_result,
    df: pd.DataFrame,
    business_question: str | None,
    target_column_override: str | None = None,
) -> dict[str, Any]:
    """Run the ReAct feature-target agent and return a target/feature/drop
    recommendation dict, matching the shape ReadinessEngine and the API
    response expect."""
    compiled = build_feature_target_graph(llm_model, api_key, azure_endpoint, api_version, col_profiles, df)

    columns_summary = _build_columns_summary(col_profiles, sem_candidates, grain_result)
    override_line = (
        f"\nThe caller has already specified the target column: '{target_column_override}'. "
        "Use it as target_column verbatim (don't second-guess it) and focus your reasoning on "
        "problem_type, time_column, recommended_approach, feature_columns, and drop_columns."
        if target_column_override
        else ""
    )
    human_content = (
        f"Business question: {business_question or '(none provided — classify generically)'}\n"
        f"{override_line}\n\n"
        f"Column profiles:\n{columns_summary}\n\n"
        "Investigate as needed using your tools, then respond with the recommendation JSON."
    )

    initial_state: FeatureTargetAgentState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_content)],
        "recommendation": {},
    }

    llm_config = get_llm_config()
    result = compiled.invoke(initial_state, config={"recursion_limit": llm_config["recursion_limit"]})
    recommendation = result.get("recommendation") or {}

    if target_column_override and target_column_override in {p.physical_name for p in col_profiles}:
        recommendation["target_column"] = target_column_override

    return recommendation
