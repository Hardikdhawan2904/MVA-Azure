"""Schema Intelligence Agent pipeline as a LangGraph StateGraph.

Topology:

  validate_and_load → check_existing_dataset → run_quality_gate ─(quality_gate_router)─┬─"fail"──────→ END (422)
                                                                                          └─"pass"──→ extract_metadata_node
  extract_metadata_node ─(classification_needed_router)─┬─"classify"──→ classify_dataset_node ─┐
                                                          └─"reuse"─────→ reuse_existing_classification_node ─┤
                                                                                                                ▼
                                                                                                  persist_and_finalize → END

check_existing_dataset resolves a same-filename re-upload as a replacement,
not an append — every downstream node (quality gate, metadata, response)
always sees exactly this upload, never a blend with an earlier one under
the same name (see nodes/pipeline.py's Node 2/3 comments for why).

classify_dataset_node has its own internal ReAct sub-graph (see
classification_agent/, a self-contained sub-agent package with its own
agent.yaml/config.py/state.py/tools.py/graph.py) — from this graph's point of
view it's a single node, only ever reached when classification is actually
needed (new dataset, or force_reclassify=true).
"""

from __future__ import annotations

from typing import Any

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from langgraph.graph import StateGraph, END

from app.agents.schema_intelligence_agent.config import get_entry_point
from app.agents.schema_intelligence_agent.state import SchemaIntelligenceState
from app.agents.schema_intelligence_agent.nodes.pipeline import (
    validate_and_load,
    run_quality_gate,
    quality_gate_router,
    check_existing_dataset,
    extract_metadata_node,
    classification_needed_router,
    classify_dataset_node,
    reuse_existing_classification_node,
    persist_and_finalize,
)


def build_schema_intelligence_graph():
    """Compile the Schema Intelligence StateGraph. Stateless across requests
    (no per-request service construction needed) so it's built once at import
    time and reused for every request."""
    g = StateGraph(SchemaIntelligenceState)

    g.add_node("validate_and_load", validate_and_load)
    g.add_node("run_quality_gate", run_quality_gate)
    g.add_node("check_existing_dataset", check_existing_dataset)
    g.add_node("extract_metadata_node", extract_metadata_node)
    g.add_node("classify_dataset_node", classify_dataset_node)
    g.add_node("reuse_existing_classification_node", reuse_existing_classification_node)
    g.add_node("persist_and_finalize", persist_and_finalize)

    g.set_entry_point(get_entry_point())

    g.add_edge("validate_and_load", "check_existing_dataset")
    g.add_edge("check_existing_dataset", "run_quality_gate")
    g.add_conditional_edges(
        "run_quality_gate", quality_gate_router,
        {"fail": END, "pass": "extract_metadata_node"},
    )
    g.add_conditional_edges(
        "extract_metadata_node", classification_needed_router,
        {"classify": "classify_dataset_node", "reuse": "reuse_existing_classification_node"},
    )
    g.add_edge("classify_dataset_node", "persist_and_finalize")
    g.add_edge("reuse_existing_classification_node", "persist_and_finalize")
    g.add_edge("persist_and_finalize", END)

    return g.compile()


_graph = build_schema_intelligence_graph()


def run_schema_intelligence_graph(
    filename: str,
    content: bytes,
    force_reclassify: bool = False,
) -> dict[str, Any] | JSONResponse:
    """Execute the schema-intelligence graph. Returns the UploadResponse-shaped
    dict on completion, or a JSONResponse (422, quality report body) if the
    dataset fails the quality gate — the same contract upload_dataset() had
    before the graph existed. File-validation/load failures still raise
    HTTPException, which FastAPI handles natively (unchanged from before)."""
    initial_state: SchemaIntelligenceState = {
        "filename": filename,
        "content": content,
        "force_reclassify": force_reclassify,
    }

    final_state = _graph.invoke(initial_state, config={"recursion_limit": 25})

    if final_state.get("error_content") is not None:
        return JSONResponse(
            status_code=final_state["error_status_code"],
            content=jsonable_encoder(final_state["error_content"]),
        )

    return final_state["response"]
