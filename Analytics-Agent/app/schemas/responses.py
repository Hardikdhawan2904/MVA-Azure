"""app/schemas/responses.py — API response contract for POST /analyze."""

from typing import Any

from pydantic import BaseModel


class AnalysisResponse(BaseModel):
    """Response body for POST /analyze. Field names match what
    Agent-Orchestrator's call_agent3 node already expects from the
    subprocess-era CLI output, so the orchestrator-side change needed to
    consume this over HTTP instead is minimal."""

    status: str
    query: str
    response: str
    conversation_id: str | None = None
    ml_readiness_score_used: float | None = None
    llm_readiness_score_used: float | None = None
    # Step-by-step decision log (intent -> ML gate/engine -> LLM gate/engine)
    # and a compact rollup of the same, built once from the graph's final
    # state — see app/agents/analytics_agent/graph.py::_build_execution_trace.
    # None on a genuine unhandled failure (status="error") rather than
    # fabricating an explanation for a crash.
    execution_trace: list[dict[str, Any]] | None = None
    execution_summary: dict[str, Any] | None = None
