"""Feature-target agent graph nodes — the ReAct loop's three steps (agent,
router, finalize) as bound methods on FeatureTargetAgentNodes, mirroring
RuleSuggestionAgentNodes' per-build construction in the sibling sub-agent.
Bound methods (not plain module functions) because each run's LLM+tools
binding is per-run data (tools close over this run's col_profiles/df) that a
stateless module-level function has nowhere to hold.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI

from app.core.logging import get_logger
from app.agents.feature_target_agent.config import get_agent_tool_names, get_llm_config
from app.agents.feature_target_agent.state import FeatureTargetAgentState
from app.agents.feature_target_agent.tools import build_feature_target_tools, tools_to_registry

logger = get_logger(__name__)

VALID_PROBLEM_TYPES = {"classification", "regression", "unknown"}
VALID_APPROACHES = {"ml", "llm", "hybrid", "unknown"}
VALID_USEFULNESS = {"high", "medium", "low"}


def _build_chat_model(
    llm_model: str, api_key: str, azure_endpoint: str, api_version: str, temperature: float, max_tokens: int
):
    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=llm_model,
        api_key=api_key,
        api_version=api_version,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _extract_json(content: str) -> dict[str, Any] | None:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    try:
        return json.loads(content.strip())
    except (json.JSONDecodeError, TypeError):
        return None


class FeatureTargetAgentNodes:
    """Bound methods sharing one LLM+tools binding for a single recommendation
    run. Constructed fresh per run_feature_target_agent() call, same lifecycle
    as RuleSuggestionAgentNodes being constructed fresh per run."""

    def __init__(
        self,
        llm_model: str,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        col_profiles: list,
        df,
    ):
        llm_config = get_llm_config()

        self._valid_columns = {p.physical_name for p in col_profiles}

        all_tools = build_feature_target_tools(col_profiles, df)
        registry = tools_to_registry(all_tools)
        self.tools = [registry[name] for name in get_agent_tool_names() if name in registry]

        llm = _build_chat_model(
            llm_model, api_key, azure_endpoint, api_version, llm_config["temperature"], llm_config["max_tokens"]
        )
        self._llm_with_tools = llm.bind_tools(self.tools)

    def call_model(self, state: FeatureTargetAgentState) -> dict:
        response = self._llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(self, state: FeatureTargetAgentState) -> Literal["tools", "finalize"]:
        last: AIMessage = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "finalize"

    def finalize(self, state: FeatureTargetAgentState) -> dict:
        last = state["messages"][-1]
        raw = last.content if hasattr(last, "content") else ""
        parsed = _extract_json(raw) if raw else None
        if not parsed:
            logger.warning("feature_target_agent_parse_failed", raw=str(raw)[:200])
            return {"recommendation": _empty_recommendation()}

        return {"recommendation": self._validate(parsed)}

    def _validate(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """Never trust a column name the LLM didn't earn — every column
        reference must be an exact physical name from this dataset, same
        rigor as RuleSuggestionAgentNodes.finalize's target_column handling."""
        target_column = parsed.get("target_column")
        if target_column is not None and target_column not in self._valid_columns:
            logger.warning("feature_target_agent_invalid_target", target_column=target_column)
            target_column = None

        time_column = parsed.get("time_column")
        if time_column is not None and time_column not in self._valid_columns:
            time_column = None

        problem_type = parsed.get("problem_type")
        if problem_type not in VALID_PROBLEM_TYPES:
            problem_type = "unknown"

        recommended_approach = parsed.get("recommended_approach")
        if recommended_approach not in VALID_APPROACHES:
            recommended_approach = "unknown"

        feature_columns = []
        for f in parsed.get("feature_columns", []) or []:
            col = f.get("column")
            if col not in self._valid_columns:
                logger.warning("feature_target_agent_invalid_feature_column", column=col)
                continue
            # Normalize case before checking -- an LLM returning "High"/"Medium"
            # (plausible prompt-adherence drift; not schema-constrained) would
            # otherwise silently downgrade to the "medium" default instead of
            # being recognized, since readiness_engine.py's usefulness_weight
            # lookup is a case-sensitive exact match too.
            raw_usefulness = (f.get("usefulness") or "").strip().lower()
            usefulness = raw_usefulness if raw_usefulness in VALID_USEFULNESS else "medium"
            feature_columns.append({
                "column": col,
                "role": f.get("role"),
                "usefulness": usefulness,
                "suggested_aggregation": f.get("suggested_aggregation"),
                "reasoning": f.get("reasoning"),
            })

        drop_columns = []
        for d in parsed.get("drop_columns", []) or []:
            col = d.get("column")
            if col not in self._valid_columns:
                logger.warning("feature_target_agent_invalid_drop_column", column=col)
                continue
            drop_columns.append({"column": col, "reason": d.get("reason")})

        try:
            confidence = round(float(parsed.get("confidence", 0.0)), 4)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return {
            "target_column": target_column,
            "problem_type": problem_type,
            "time_column": time_column,
            "recommended_approach": recommended_approach,
            "approach_reasoning": parsed.get("approach_reasoning"),
            "feature_columns": feature_columns,
            "drop_columns": drop_columns,
            "confidence": confidence,
        }


def _empty_recommendation() -> dict[str, Any]:
    return {
        "target_column": None,
        "problem_type": "unknown",
        "time_column": None,
        "recommended_approach": "unknown",
        "approach_reasoning": None,
        "feature_columns": [],
        "drop_columns": [],
        "confidence": 0.0,
    }
