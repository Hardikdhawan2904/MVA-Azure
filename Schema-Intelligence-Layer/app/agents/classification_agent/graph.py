"""Dataset-classification sub-agent — genuine ReAct tool-calling LangGraph.

Topology:
  agent ──(tool calls?)──► tools ──► agent   (loop)
  agent ──(no tool calls)─► finalize ──► END

See nodes/pipeline.py for the three node implementations, tools.py for what
the agent can call, and agent.yaml/config.py for its declarative persona and
LLM/tool config.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.agents.classification_agent.config import get_entry_point, get_llm_config
from app.agents.classification_agent.nodes.pipeline import ClassificationAgentNodes, FALLBACK_CLASSIFICATION
from app.agents.classification_agent.state import ClassificationAgentState
from app.models.schemas import DatasetMetadata, LLMClassification

SYSTEM_PROMPT = """You are a data-intake coordinator with two tools available:
generate_column_descriptions and classify_dataset. Call generate_column_descriptions
first, then classify_dataset — descriptions improve classification accuracy. Once both
have returned results, stop calling tools and reply with a short confirmation."""


def build_classification_graph(metadata: DatasetMetadata):
    """Compile the classification StateGraph. Built fresh per run (its nodes
    close over this run's metadata-bound tools), unlike the top-level Schema
    Intelligence graph which is stateless and compiled once at import time."""
    nodes = ClassificationAgentNodes(metadata)

    g = StateGraph(ClassificationAgentState)
    g.add_node("agent", nodes.call_model)
    g.add_node("tools", ToolNode(nodes.tools))
    g.add_node("finalize", nodes.finalize)
    g.set_entry_point(get_entry_point())
    g.add_conditional_edges("agent", nodes.should_continue, {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)

    return g.compile()


def run_classification_agent(metadata: DatasetMetadata) -> tuple[dict[str, str], LLMClassification]:
    """Run the ReAct classification agent and return (column_descriptions,
    LLMClassification) — the exact same shapes the old direct llm_service.py
    calls produced, so callers don't need to know this is now agentic."""
    compiled = build_classification_graph(metadata)

    human_content = (
        f"Dataset: {metadata.dataset_name} ({metadata.file_type}, {metadata.row_count} rows, "
        f"{metadata.column_count} columns)\n"
        f"Columns: {metadata.column_names}\n"
        f"Types: {metadata.column_data_types}\n\n"
        "Generate column descriptions, then classify this dataset."
    )
    initial_state: ClassificationAgentState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_content)],
        "column_descriptions": {},
        "classification": {},
    }

    llm_config = get_llm_config()
    result = compiled.invoke(initial_state, config={"recursion_limit": llm_config["recursion_limit"]})

    descriptions = result.get("column_descriptions") or {}
    classification_dict = result.get("classification") or FALLBACK_CLASSIFICATION.model_dump()
    classification = LLMClassification(**classification_dict)
    return descriptions, classification
