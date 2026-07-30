"""app/agents/analytics_agent/config.py — LangGraph-specific accessors.

Deployment/infra config (secrets, host/port) lives in app/config.py's
Settings-equivalent constants — see that module's docstring for why this
agent keeps YAML loading there too rather than re-parsing agent.yaml a
second time here. This module only owns the one thing genuinely specific
to the graph's own topology: its entry point.

Agent 3 redesign, Phase 4 (plan "zany-giggling-crayon") — get_execution_plans()
was dropped here: agent.yaml's execution_plans dict (Intent -> static tool
chain) is no longer read by code. tools_used is now derived from what
genuinely executed for a given request (nodes/pipeline.py's
_build_tools_used) rather than a hand-maintained, Insurance-only mapping.
"""


def get_entry_point() -> str:
    """The LangGraph entry node name — every query starts here. Builds the
    dataset context (Stage 0 of the Agent 3 redesign) before Stage 1's
    capability resolution, since every later stage needs it."""
    return "build_dataset_context"
