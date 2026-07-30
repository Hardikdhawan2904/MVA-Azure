"""Stage 0A — dataset versioning.

A re-upload isn't always either "the exact same content" (duplicate) or
"a wholly unrelated dataset" (new) — it can be a *new version* of a
dataset the Registry has already seen: same logical dataset, edited rows,
different fingerprint. Since exact-match-only was the explicit scoping
decision for content identity (no fuzzy/statistical similarity — see
fingerprint.py), version lineage is decided by a separate, explicit
signal instead: the uploaded filename. This mirrors the precedent already
set by Agent 1's own check_existing_dataset()/classification_needed_router
(Schema-Intelligence-Layer/app/agents/schema_intelligence_agent/nodes/pipeline.py)
which already treats "same filename" as "the same logical dataset" for its
own replace-vs-new decision — reused here, at the Registry layer, to chain
versions instead of overwriting them.
"""

from __future__ import annotations

from app.services.dataset_registry import metadata_cache
from app.services.dataset_registry.models import MasterDataset


def find_prior_version(original_filename: str) -> MasterDataset | None:
    """The most recent Master Dataset previously uploaded under this exact
    filename, if any — None means this is a genuinely new, unrelated
    dataset (or the very first upload of this filename). Raises on DB
    error — DatasetRegistry.register() wraps this for the non-fatal
    degrade-to-"treat as new" behavior."""
    return metadata_cache.lookup_latest_version_by_filename(original_filename)


def next_version_number(prior: MasterDataset | None) -> int:
    return 1 if prior is None else prior.latest_version + 1
