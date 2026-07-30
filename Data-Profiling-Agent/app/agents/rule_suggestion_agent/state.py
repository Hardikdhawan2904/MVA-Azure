"""Typed state for the rule-suggestion ReAct sub-agent."""

from typing import Any

from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class RuleSuggestionAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    suggestions: list[dict[str, Any]]
