"""Typed state for the dataset-classification ReAct sub-agent."""

from typing import Any

from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class ClassificationAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    column_descriptions: dict[str, str]
    classification: dict[str, Any]
