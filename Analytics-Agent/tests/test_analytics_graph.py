"""tests/test_analytics_graph.py — Routing tests for the Analytics Agent's
LangGraph (app/agents/analytics_agent/graph.py + nodes/pipeline.py).

Agent 3 redesign, Phase 4 (plan "zany-giggling-crayon") — rewritten for
the new Stage 1-9 linear topology. There's no more per-intent dispatch
into 7 different handler nodes (route_by_intent/route_after_handler are
gone); AnalyticsPlanner/AnalyticsScheduler narrow a request down instead
of graph routing. The two routers that remain are
route_after_interpret (early-exit when a named KPI couldn't be resolved)
and route_after_execute (early-exit when nothing could be planned, or the
single scheduled analysis produced no evidence).

Focuses on graph wiring, not each analyzer's evidence-building logic
(covered by test_analyzers.py, test_kpi_analyzers.py, and
test_harness.py's live response-content assertions).
"""

import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.agents.analytics_agent.graph import build_analytics_graph
from app.agents.analytics_agent.nodes.pipeline import AnalyticsGraphNodes

_DATASET = Path(__file__).parent / "fixtures" / "insurance_variance_data_native.csv"

pytestmark = pytest.mark.skipif(
    not _DATASET.exists(), reason=f"Insurance test dataset not found at {_DATASET}"
)


@pytest.fixture(scope="module")
def nodes():
    # Real callers get detected_domain from the Orchestrator (Agent 1's
    # domain classification); a direct call like this fixture must state
    # its domain assumption explicitly — see GenericDomainPlugin's
    # docstring (Phase 4) for why silently defaulting to Insurance for any
    # unlabeled dataset was a real bug.
    return AnalyticsGraphNodes(dataset_path=str(_DATASET), conversation_id=str(uuid.uuid4()), detected_domain="Insurance")


# ── route_after_interpret ────────────────────────────────────────────────────

def test_route_after_interpret_response_skips_planning(nodes):
    assert nodes.route_after_interpret({"response": "KPI not found"}) == "record_memory"


def test_route_after_interpret_no_response_proceeds_to_planning(nodes):
    assert nodes.route_after_interpret({"question_intent": None}) == "plan_analytics"


# ── route_after_execute ──────────────────────────────────────────────────────

def test_route_after_execute_response_skips_narrate(nodes):
    assert nodes.route_after_execute({"response": "already answered"}) == "record_memory"


def test_route_after_execute_evidence_builder_goes_to_narrate(nodes):
    from app.services.evidence.evidence_builder import EvidenceBuilder
    assert nodes.route_after_execute({"evidence_builder": EvidenceBuilder()}) == "narrate"


# ── Graph compilation ─────────────────────────────────────────────────────────

def test_graph_builds_and_compiles():
    graph = build_analytics_graph(dataset_path=str(_DATASET), conversation_id=str(uuid.uuid4()))
    # A compiled LangGraph exposes .invoke — presence confirms compile() succeeded.
    assert hasattr(graph, "invoke")


# ── End-to-end: business question resolves to the expected single analysis ──

@pytest.mark.parametrize("query,expected_analysis_type", [
    ("Show Gross Written Premium for FY2025", "kpi_summary"),
    ("Show loss ratio variance vs budget for EMEA", "kpi_variance"),
    ("Why did underwriting result decline in FY2025?", "root_cause"),
    ("Show the trend of loss ratio over time", "trend"),
    ("Forecast underwriting result for next 6 months", "forecast"),
    ("Detect anomalies in loss ratios", "anomaly_detection"),
    ("Segment portfolio by risk profile", "segmentation"),
])
def test_end_to_end_single_analysis_routing(query, expected_analysis_type):
    """Each of these queries matches the old system's single-intent
    behavior exactly — one scheduled analysis, not a multi-analysis
    report — confirmed via the backward-compat adapter's state["intent"]
    (see nodes/pipeline.py::_adapt_evidence_for_narration)."""
    conversation_id = str(uuid.uuid4())
    graph = build_analytics_graph(dataset_path=str(_DATASET), conversation_id=conversation_id, detected_domain="Insurance")
    final_state = graph.invoke(
        {
            "business_question": query, "dataset_path": str(_DATASET), "conversation_id": conversation_id,
            # run_analytics_graph() (the real entry point) always populates
            # these — the old topology never read them from state (MLTool's
            # readiness score came from the AnalyticsGraphNodes constructor
            # arg instead), but Stage 1's resolve_capabilities does.
            "ml_readiness_score": 99.75, "llm_readiness_score": 99.75,
        },
        config={"recursion_limit": 25},
    )
    assert final_state["intent"] == expected_analysis_type
    assert final_state.get("response")
