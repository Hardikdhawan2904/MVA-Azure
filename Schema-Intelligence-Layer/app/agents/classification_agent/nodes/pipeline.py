"""Classification agent graph nodes — the ReAct loop's three steps (agent,
router, finalize) as bound methods on ClassificationAgentNodes, mirroring
ProfilingGraphNodes' per-build construction in Agent 2's top-level graph.
Bound methods (not plain module functions) because each run's LLM+tools
binding is per-run data (tools close over this run's DatasetMetadata) that a
stateless module-level function has nowhere to hold.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI

from app.config import settings
from app.agents.classification_agent.config import get_agent_tool_names, get_llm_config
from app.agents.classification_agent.state import ClassificationAgentState
from app.agents.classification_agent.tools import build_classification_tools, tools_to_registry
from app.models.schemas import DatasetMetadata, LLMClassification

logger = logging.getLogger(__name__)

FALLBACK_DESCRIPTIONS_NOTE = "Column description was not generated for this run."
FALLBACK_CLASSIFICATION = LLMClassification(
    business_domain="Other",
    sub_domain="General",
    dataset_summary="Classification could not be completed for this run.",
    confidence=0.0,
    reason="The classification agent did not produce a result.",
)


def _build_chat_model(temperature: float, max_tokens: int):
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _extract_tool_result(messages: list, tool_name: str) -> str | None:
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == tool_name:
            return msg.content
    return None


class ClassificationAgentNodes:
    """Bound methods sharing one LLM+tools binding for a single classification
    run. Constructed fresh per run_classification_agent() call, same lifecycle
    as ProfilingGraphNodes being constructed fresh per graph build."""

    def __init__(self, metadata: DatasetMetadata):
        self._metadata = metadata
        llm_config = get_llm_config()

        all_tools = build_classification_tools(metadata)
        registry = tools_to_registry(all_tools)
        self.tools = [registry[name] for name in get_agent_tool_names() if name in registry]

        llm = _build_chat_model(llm_config["temperature"], llm_config["max_tokens"])
        self._llm_with_tools = llm.bind_tools(self.tools)

    def call_model(self, state: ClassificationAgentState) -> dict:
        response = self._llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(self, state: ClassificationAgentState) -> Literal["tools", "finalize"]:
        last: AIMessage = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "finalize"

    def finalize(self, state: ClassificationAgentState) -> dict:
        metadata = self._metadata
        descriptions_raw = _extract_tool_result(state["messages"], "generate_column_descriptions")
        classification_raw = _extract_tool_result(state["messages"], "classify_dataset")

        if descriptions_raw:
            try:
                descriptions = json.loads(descriptions_raw)
            except json.JSONDecodeError:
                descriptions = {}
        else:
            logger.warning("classification_agent_descriptions_tool_not_called")
            descriptions = {col: FALLBACK_DESCRIPTIONS_NOTE for col in metadata.column_names}

        if classification_raw:
            try:
                classification = json.loads(classification_raw)
            except json.JSONDecodeError:
                classification = FALLBACK_CLASSIFICATION.model_dump()
        else:
            logger.warning("classification_agent_classify_tool_not_called")
            classification = FALLBACK_CLASSIFICATION.model_dump()

        return {"column_descriptions": descriptions, "classification": classification}
