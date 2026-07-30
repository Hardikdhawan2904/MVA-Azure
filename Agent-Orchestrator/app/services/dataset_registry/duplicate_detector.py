"""Stage 0A — duplicate detection.

Thin, explicit wrapper over the Master Dataset Repository lookup — kept as
its own module (matching the approved architecture doc's folder structure)
so "is this a duplicate" is a named decision, not buried inline in
DatasetRegistry.register().
"""

from __future__ import annotations

from app.services.dataset_registry import metadata_cache
from app.services.dataset_registry.models import MasterDataset


def find_duplicate(fingerprint: str) -> MasterDataset | None:
    """Exact-match only (see fingerprint.py) — returns the existing Master
    Dataset if this fingerprint has been seen before, else None. Raises on
    DB error — DatasetRegistry.register() wraps this for the non-fatal
    degrade-to-miss behavior."""
    return metadata_cache.lookup_master_dataset(fingerprint)
