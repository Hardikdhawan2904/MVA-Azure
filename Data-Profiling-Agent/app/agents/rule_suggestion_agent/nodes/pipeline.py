"""Rule-suggestion agent graph nodes — the ReAct loop's three steps (agent,
router, finalize) as bound methods on RuleSuggestionAgentNodes, mirroring
ProfilingGraphNodes' per-build construction in the main pipeline graph.
Bound methods (not plain module functions) because each run's LLM+tools
binding is per-run data (tools close over this run's col_profiles/df/
existing_rule_keys) that a stateless module-level function has nowhere to hold.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI

from app.core.enums import RuleSuggestionStatus, RuleType
from app.core.logging import get_logger
from app.agents.rule_suggestion_agent.config import get_agent_tool_names, get_llm_config
from app.agents.rule_suggestion_agent.state import RuleSuggestionAgentState
from app.agents.rule_suggestion_agent.tools import build_rule_suggestion_tools, tools_to_registry

logger = get_logger(__name__)

# Same set RuleEngine actually evaluates — see suggestion_generator.py for why
# date_range / cross_field_equality / cross_field_inequality are excluded.
ENGINE_SUPPORTED_RULE_TYPES = {
    RuleType.NON_NULL, RuleType.EXPECTED_UNIQUE, RuleType.REGEX_MATCH,
    RuleType.ALLOWED_VALUES, RuleType.NUMERIC_RANGE,
    RuleType.COLUMN_COMPARISON, RuleType.CONDITIONAL_REQUIRED,
}


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


class RuleSuggestionAgentNodes:
    """Bound methods sharing one LLM+tools binding for a single suggestion
    run. Constructed fresh per run_rule_suggestion_agent() call, same
    lifecycle as ProfilingGraphNodes being constructed fresh per graph build."""

    def __init__(
        self,
        llm_model: str,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        col_profiles: list,
        df,
        existing_rule_keys: list[str],
    ):
        llm_config = get_llm_config()

        all_tools = build_rule_suggestion_tools(col_profiles, df, existing_rule_keys)
        registry = tools_to_registry(all_tools)
        self.tools = [registry[name] for name in get_agent_tool_names() if name in registry]

        llm = _build_chat_model(
            llm_model, api_key, azure_endpoint, api_version, llm_config["temperature"], llm_config["max_tokens"]
        )
        self._llm_with_tools = llm.bind_tools(self.tools)

    def call_model(self, state: RuleSuggestionAgentState) -> dict:
        response = self._llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(self, state: RuleSuggestionAgentState) -> Literal["tools", "finalize"]:
        last: AIMessage = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "finalize"

    def finalize(self, state: RuleSuggestionAgentState) -> dict:
        last = state["messages"][-1]
        raw = last.content if hasattr(last, "content") else ""
        parsed = _extract_json(raw) if raw else None
        if not parsed or "suggestions" not in parsed:
            logger.warning("rule_suggestion_agent_parse_failed", raw=str(raw)[:200])
            return {"suggestions": []}

        suggestions: list[dict[str, Any]] = []
        for s in parsed["suggestions"][:5]:
            rule_type_str = s.get("type", "")
            try:
                engine_compatible = RuleType(rule_type_str) in ENGINE_SUPPORTED_RULE_TYPES
            except ValueError:
                engine_compatible = False
                logger.warning("rule_suggestion_invalid_type", rule_key=s.get("rule_key"), type=rule_type_str)

            suggestions.append({
                "suggestion_id": str(uuid.uuid4()),
                **s,
                "confidence": round(float(s.get("confidence", 0.0)), 4),
                "engine_compatible": engine_compatible,
                "status": RuleSuggestionStatus.PROPOSED.value,
            })
        return {"suggestions": suggestions}
