"""Stage 0A — Dataset Registry domain models.

Field names mirror the approved architecture doc's "Dataset Registry &
Lifecycle Management" section verbatim (MasterDataset / DatasetCopy).
`schema`/`validation`/`column_profiles`/`hierarchy`/`charts`/
`ml_readiness`/`llm_readiness`/`feature_recommendation` are the same
metadata the existing pipeline already produces (Agent 1's classification +
quality report, Agent 2's full profiling result) — the Registry caches it,
it does not compute anything new.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MasterDataset:
    fingerprint: str
    dataset_id: str
    original_filename: str
    storage_location: str
    file_extension: str
    byte_size: int
    row_count: int | None
    column_count: int | None
    latest_version: int
    previous_fingerprint: str | None
    reference_count: int
    first_uploaded_at: datetime
    last_referenced_at: datetime

    # Cached pipeline outputs — populated once Agent 1 + Agent 2 have run
    # for this fingerprint; None on a freshly-created row before that.
    schema: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    column_profiles: dict[str, Any] | None = None
    hierarchy: dict[str, Any] | None = None
    charts: dict[str, Any] | None = None
    ml_readiness: dict[str, Any] | None = None
    llm_readiness: dict[str, Any] | None = None
    feature_recommendation: dict[str, Any] | None = None

    @property
    def has_cached_results(self) -> bool:
        return self.schema is not None


@dataclass
class DatasetCopy:
    copy_id: str
    fingerprint: str
    uploaded_filename: str
    upload_timestamp: datetime
    version: int
    user_id: str | None = None
    deleted_flag: bool = False


@dataclass
class CachedPipelineResult:
    """The exact three fields Agent-Orchestrator's finalize() node reads
    off state today (agent1_body, agent2_full_result, primary_domain) —
    what a cache hit reconstructs, verbatim, without calling Agent 1/2."""
    primary_domain: str
    agent1_body: dict[str, Any]
    agent2_full_result: dict[str, Any]
    response_schema_version: int = 1


@dataclass
class RegisterResult:
    master: MasterDataset
    copy: DatasetCopy
    was_duplicate: bool
    is_new_version: bool
    cached_result: CachedPipelineResult | None = None
