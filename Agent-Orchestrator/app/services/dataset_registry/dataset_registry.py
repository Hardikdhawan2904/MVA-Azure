"""Stage 0A — DatasetRegistry: the facade the Orchestrator's graph node
calls. Not an AI agent — deterministic infrastructure only (see the
approved architecture doc's "Dataset Registry & Lifecycle Management"
section). Method names mirror that doc's pseudocode interface
(register/computeFingerprint/lookupMasterDataset/detectDuplicate/
detectVersion/createCopy/updateReferenceCount), expressed in this
codebase's snake_case convention.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.services.dataset_registry import duplicate_detector, metadata_cache, reference_counter, storage, version_manager
from app.services.dataset_registry.fingerprint import compute_fingerprint
from app.services.dataset_registry.models import CachedPipelineResult, DatasetCopy, MasterDataset, RegisterResult

logger = logging.getLogger(__name__)


class DatasetRegistry:
    def __init__(self, storage_dir: str):
        self._storage_dir = storage_dir

    # ── Public interface (mirrors the architecture doc) ────────────────────

    def compute_fingerprint(self, content: bytes) -> str:
        return compute_fingerprint(content)

    def lookup_master_dataset(self, fingerprint: str) -> MasterDataset | None:
        return metadata_cache.lookup_master_dataset(fingerprint)

    def detect_duplicate(self, fingerprint: str) -> MasterDataset | None:
        return duplicate_detector.find_duplicate(fingerprint)

    def detect_version(self, original_filename: str) -> MasterDataset | None:
        return version_manager.find_prior_version(original_filename)

    def get_cached_result(self, fingerprint: str) -> CachedPipelineResult | None:
        return metadata_cache.get_cached_result(fingerprint)

    def create_copy(self, *, fingerprint: str, uploaded_filename: str, version: int,
                     user_id: str | None = None) -> DatasetCopy:
        copy = DatasetCopy(
            copy_id=str(uuid4()), fingerprint=fingerprint, uploaded_filename=uploaded_filename,
            upload_timestamp=datetime.now(timezone.utc), version=version, user_id=user_id, deleted_flag=False,
        )
        metadata_cache.insert_copy(copy)
        return copy

    def update_reference_count(self, fingerprint: str, delta: int) -> None:
        if delta > 0:
            reference_counter.increment(fingerprint)
        elif delta < 0:
            reference_counter.decrement(fingerprint)

    def _create_copy_and_increment(self, *, fingerprint: str, uploaded_filename: str, version: int,
                                    user_id: str | None = None) -> DatasetCopy:
        """Used by register() instead of create_copy() + update_reference_count()
        separately — one connection/one transaction for the insert+increment
        pair, so a crash between the two steps can never under-count
        reference_count (see metadata_cache.insert_copy_and_increment_reference's
        own docstring for why this matters)."""
        copy = DatasetCopy(
            copy_id=str(uuid4()), fingerprint=fingerprint, uploaded_filename=uploaded_filename,
            upload_timestamp=datetime.now(timezone.utc), version=version, user_id=user_id, deleted_flag=False,
        )
        metadata_cache.insert_copy_and_increment_reference(copy)
        return copy

    def register(self, *, filename: str, file_extension: str, content: bytes,
                 user_id: str | None = None) -> RegisterResult:
        """Stage 0A's entry point — fingerprint -> lookup -> duplicate /
        new-version / genuinely-new branching -> copy creation. Never calls
        Agent 1/2 itself; on a cache miss the caller (the Orchestrator's
        graph node) still needs to run the existing Agent 1 -> Agent 2
        chain and then call persist_result() below."""
        fingerprint = self.compute_fingerprint(content)

        existing = self.detect_duplicate(fingerprint)
        if existing is not None:
            copy = self._create_copy_and_increment(
                fingerprint=fingerprint, uploaded_filename=filename, version=existing.latest_version,
                user_id=user_id,
            )
            metadata_cache.touch_last_referenced(fingerprint)
            cached_result = self.get_cached_result(fingerprint)
            return RegisterResult(
                master=existing, copy=copy, was_duplicate=True, is_new_version=False, cached_result=cached_result,
            )

        prior_version = self.detect_version(filename)
        if prior_version is not None:
            version_number = version_manager.next_version_number(prior_version)
            dataset_id = prior_version.dataset_id
            previous_fingerprint = prior_version.fingerprint
        else:
            version_number = 1
            dataset_id = str(uuid4())
            previous_fingerprint = None

        storage_location = storage.write_master_file(self._storage_dir, fingerprint, file_extension, content)
        now = datetime.now(timezone.utc)
        master = MasterDataset(
            fingerprint=fingerprint, dataset_id=dataset_id, original_filename=filename,
            storage_location=storage_location, file_extension=file_extension, byte_size=len(content),
            row_count=None, column_count=None, latest_version=version_number,
            previous_fingerprint=previous_fingerprint, reference_count=0,
            first_uploaded_at=now, last_referenced_at=now,
        )
        metadata_cache.insert_master_dataset(master)

        copy = self._create_copy_and_increment(
            fingerprint=fingerprint, uploaded_filename=filename, version=version_number, user_id=user_id,
        )

        return RegisterResult(
            master=master, copy=copy, was_duplicate=False, is_new_version=prior_version is not None,
            cached_result=None,
        )

    def persist_result(self, *, fingerprint: str, primary_domain: str, agent1_body: dict,
                        agent2_full_result: dict, row_count: int | None, column_count: int | None) -> None:
        """Called after a cache-miss run completes — stores Agent 1 +
        Agent 2's outputs against this fingerprint so the next
        byte-identical upload becomes a cache hit."""
        metadata_cache.upsert_result(fingerprint, primary_domain, agent1_body, agent2_full_result)
        metadata_cache.update_master_counts(fingerprint, row_count, column_count)
