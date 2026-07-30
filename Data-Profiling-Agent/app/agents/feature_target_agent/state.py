"""Typed state for the feature-target ReAct sub-agent."""

from typing import Any

from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class FeatureTargetAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    recommendation: dict[str, Any]
