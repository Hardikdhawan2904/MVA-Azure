"""Typed state for the orchestrator LangGraph — threads the pipeline handoff
between Agent 1 (Schema Intelligence Layer) and Agent 2 (Data Profiling Engine)
between nodes, replacing the old sequential-with-early-return function body.
"""

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    filename: str
    content_type: str
    content: bytes
    sheet_name: str | None
    force_reclassify: bool
    force_revalidate: bool
    business_question: str | None
    target_column: str | None
    request_rules: str | None

    # ── Dataset Registry (Stage 0A) ────────────────────────────────────────
    file_extension: str
    fingerprint: str | None
    copy_id: str | None
    was_cached: bool

    # ── Agent 1 ─────────────────────────────────────────────────────────────
    agent1_status_code: int
    agent1_body: dict[str, Any]

    # ── Derived from Agent 1's classification ──────────────────────────────
    primary_domain: str | None
    schema_metadata: dict[str, Any]

    # ── Agent 2 ─────────────────────────────────────────────────────────────
    agent2_status_code: int
    agent2_body: dict[str, Any]
    run_id: str
    agent2_full_result: dict[str, Any]

    # ── Agent 3 (Analytics Agent — CLI subprocess, best-effort) ────────────
    agent3_body: dict[str, Any] | None

    # ── Output ──────────────────────────────────────────────────────────────
    result: dict[str, Any]
    error_status_code: int
    error_content: dict[str, Any]
