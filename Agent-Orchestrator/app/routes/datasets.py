"""Dataset Registry admin routes — list Master Datasets, list/delete
copies, delete a Master Dataset (guarded — see reference_counter.py).
Read-only listing + lifecycle management only; uploads still go through
POST /pipeline/run, which is where Stage 0A's resolve_dataset_identity
node actually registers new content.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.dataset_registry import metadata_cache, reference_counter

router = APIRouter(prefix="/datasets", tags=["Dataset Registry"])


class MasterDatasetOut(BaseModel):
    fingerprint: str
    dataset_id: str
    original_filename: str
    file_extension: str
    byte_size: int
    row_count: int | None
    column_count: int | None
    latest_version: int
    previous_fingerprint: str | None
    reference_count: int
    first_uploaded_at: str
    last_referenced_at: str
    has_cached_results: bool


class DatasetCopyOut(BaseModel):
    copy_id: str
    fingerprint: str
    uploaded_filename: str
    upload_timestamp: str
    version: int
    user_id: str | None
    deleted_flag: bool


@router.get("/masters", response_model=list[MasterDatasetOut])
def list_masters(limit: int = Query(default=100, le=500)):
    masters = metadata_cache.list_master_datasets(limit=limit)
    return [
        MasterDatasetOut(
            fingerprint=m.fingerprint, dataset_id=m.dataset_id, original_filename=m.original_filename,
            file_extension=m.file_extension, byte_size=m.byte_size, row_count=m.row_count,
            column_count=m.column_count, latest_version=m.latest_version,
            previous_fingerprint=m.previous_fingerprint, reference_count=m.reference_count,
            first_uploaded_at=m.first_uploaded_at.isoformat(), last_referenced_at=m.last_referenced_at.isoformat(),
            has_cached_results=m.has_cached_results,
        )
        for m in masters
    ]


@router.get("/masters/{fingerprint}/copies", response_model=list[DatasetCopyOut])
def list_copies(fingerprint: str, include_deleted: bool = Query(default=False)):
    copies = metadata_cache.list_copies_for_fingerprint(fingerprint, include_deleted=include_deleted)
    return [
        DatasetCopyOut(
            copy_id=c.copy_id, fingerprint=c.fingerprint, uploaded_filename=c.uploaded_filename,
            upload_timestamp=c.upload_timestamp.isoformat(), version=c.version, user_id=c.user_id,
            deleted_flag=c.deleted_flag,
        )
        for c in copies
    ]


@router.delete("/copies/{copy_id}")
def delete_copy(copy_id: str):
    """Soft-deletes this copy and decrements the Master Dataset's
    reference_count. Never touches the Master Dataset or its physical
    file."""
    fingerprint = reference_counter.soft_delete_copy(copy_id)
    if fingerprint is None:
        raise HTTPException(status_code=404, detail=f"Copy '{copy_id}' not found or already deleted.")
    return {"copy_id": copy_id, "fingerprint": fingerprint, "deleted": True}


@router.delete("/masters/{fingerprint}")
def delete_master(fingerprint: str, force: bool = Query(default=False)):
    """Refuses unless forced when active copies still reference this
    Master Dataset (see reference_counter.can_delete_master)."""
    ok, reason = reference_counter.delete_master(fingerprint, force=force)
    if not ok:
        status_code = 404 if reason and "not found" in reason else 409
        raise HTTPException(status_code=status_code, detail=reason)
    return {"fingerprint": fingerprint, "deleted": True}
