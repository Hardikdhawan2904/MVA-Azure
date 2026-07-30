"""Orchestrator pipeline as a LangGraph StateGraph.

Topology:

  resolve_dataset_identity ─(cache_hit)─────────────────────────────────────→ call_agent3
      │(cache_miss)
      ▼
  call_agent1 ─(stop)──────────────────────────────────────────────→ END (error)
      │(continue)
      ▼
  extract_domain_and_metadata ─(stop)──────────────────────────────→ END (error)
      │(continue)
      ▼
  call_agent2 ─(stop)───────────────────────────────────────────────→ END (error)
      │(continue)
      ▼
  fetch_agent2_result ─(stop)───────────────────────────────────────→ END (error)
      │(continue)
      ▼
  persist_master_dataset (best-effort — never routes to "stop")
      │
      ▼
  call_agent3 (best-effort — never routes to "stop")
      │
      ▼
  finalize → END (success)

Each "stop" branch means a node already populated error_status_code /
error_content in state — the graph doesn't need a dedicated error node,
routing straight to END is enough since run_orchestrator_pipeline() reads
those two fields directly off the final state.

resolve_dataset_identity (Stage 0A — Dataset Registry, see the architecture
doc's "Dataset Registry & Lifecycle Management" section) is deterministic
infrastructure, not an agent: on a duplicate upload with an already-cached
pipeline result it fabricates agent1_body/agent2_full_result/primary_domain
directly and routes straight to call_agent3, skipping Agent 1 and Agent 2
entirely; a genuinely new dataset, a new version, or a
force_revalidate/force_reclassify request falls through the existing chain
unchanged.
"""

from typing import Any

from fastapi.responses import JSONResponse
from langgraph.graph import StateGraph, END

from app.agents.orchestration_agent.config import get_entry_point
from app.agents.orchestration_agent.state import PipelineState
from app.agents.orchestration_agent.nodes.pipeline import OrchestratorGraphNodes

_nodes = OrchestratorGraphNodes()


def build_orchestrator_graph():
    """Compile the orchestrator StateGraph. Stateless across requests (no
    per-request service construction needed, unlike Agent 2's graph) so it's
    built once at import time and reused for every request."""
    g = StateGraph(PipelineState)

    g.add_node("resolve_dataset_identity", _nodes.resolve_dataset_identity)
    g.add_node("call_agent1", _nodes.call_agent1)
    g.add_node("extract_domain_and_metadata", _nodes.extract_domain_and_metadata)
    g.add_node("call_agent2", _nodes.call_agent2)
    g.add_node("fetch_agent2_result", _nodes.fetch_agent2_result)
    g.add_node("persist_master_dataset", _nodes.persist_master_dataset)
    g.add_node("call_agent3", _nodes.call_agent3)
    g.add_node("finalize", _nodes.finalize)

    g.set_entry_point(get_entry_point())

    g.add_conditional_edges(
        "resolve_dataset_identity", _nodes.route_after_identity,
        {"cache_hit": "call_agent3", "cache_miss": "call_agent1"},
    )
    g.add_conditional_edges(
        "call_agent1", _nodes.route_after_agent1,
        {"stop": END, "continue": "extract_domain_and_metadata"},
    )
    g.add_conditional_edges(
        "extract_domain_and_metadata", _nodes.route_after_domain,
        {"stop": END, "continue": "call_agent2"},
    )
    g.add_conditional_edges(
        "call_agent2", _nodes.route_after_agent2,
        {"stop": END, "continue": "fetch_agent2_result"},
    )
    g.add_conditional_edges(
        "fetch_agent2_result", _nodes.route_after_fetch,
        {"stop": END, "continue": "persist_master_dataset"},
    )
    g.add_edge("persist_master_dataset", "call_agent3")
    g.add_edge("call_agent3", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


_graph = build_orchestrator_graph()


async def run_orchestrator_pipeline(
    filename: str,
    content_type: str,
    content: bytes,
    sheet_name: str | None = None,
    force_reclassify: bool = False,
    force_revalidate: bool = False,
    business_question: str | None = None,
    target_column: str | None = None,
    request_rules: str | None = None,
) -> dict[str, Any] | JSONResponse:
    """Execute the orchestrator graph. Returns the success dict
    ({"agent1", "agent2", "agent3", "primary_domain_used"}) on completion —
    "agent3" is null/"skipped" outside its Insurance+business_question+CSV
    scope — or a JSONResponse with the same stage/status-code shape the old
    linear run_pipeline() returned on any Agent 1/2 failure.

    request_rules: optional JSON string of additional Agent 2 business
    rules (see Data-Profiling-Agent's rule_loader.py for the schema) —
    forwarded to Agent 2 verbatim, on top of the mandatory/expected_unique
    column inference extract_domain_and_metadata already does. Distinct
    from that inference: a cross-field consistency rule (e.g. "column A
    should be <= column B") asserts a real relationship between two
    specific columns, which nothing about a column's *name* can safely
    infer — a wrong guess here would score a false relationship as
    quality, actively eroding trust rather than helping. Left as an
    explicit, caller-supplied opt-in rather than another auto-inferred
    default.
    """
    initial_state: PipelineState = {
        "filename": filename,
        "content_type": content_type,
        "content": content,
        "sheet_name": sheet_name,
        "force_reclassify": force_reclassify,
        "force_revalidate": force_revalidate,
        "business_question": business_question,
        "target_column": target_column,
        "request_rules": request_rules,
    }

    final_state = await _graph.ainvoke(initial_state, config={"recursion_limit": 25})

    if final_state.get("error_content") is not None:
        return JSONResponse(
            status_code=final_state["error_status_code"],
            content=final_state["error_content"],
        )

    return final_state["result"]
