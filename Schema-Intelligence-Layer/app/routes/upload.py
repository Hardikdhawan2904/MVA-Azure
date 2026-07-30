"""
API routes for dataset upload and retrieval.
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import UploadResponse, DatasetMetadata, DatasetListItem
from app.agents.schema_intelligence_agent.graph import run_schema_intelligence_graph
from app.services.database import get_metadata, list_all_metadata
from app.datastore.registry import get_dataframe

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/upload-dataset",
    response_model=UploadResponse,
    summary="Upload a CSV or Excel dataset",
    description="Accepts a CSV or Excel file, validates it, persists it in its dedicated PostgreSQL database, and registers/classifies its metadata. Re-uploading an already-seen filename replaces the prior version rather than combining with it.",
    responses={
        400: {"description": "File unreadable or corrupted"},
        413: {"description": "File exceeds MAX_UPLOAD_SIZE_MB"},
        415: {"description": "Unsupported file format"},
        422: {"description": "Empty dataset or invalid structure"},
        500: {"description": "Internal processing error"},
    },
)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV or Excel file to upload"),
    force_reclassify: bool = Query(
        False,
        description="When replacing an existing dataset (same filename re-uploaded), re-run LLM "
                    "column descriptions and domain classification instead of reusing the original "
                    "result. Use this to recover a dataset stuck with a failed/fallback "
                    "classification (e.g. from an earlier LLM rate-limit or connection error).",
    ),
):
    """
    Main upload endpoint — delegates the full pipeline (validate, load, quality
    gate, replace-vs-new detection, classification, persistence) to the
    Schema Intelligence Agent's LangGraph (app/agents/schema_intelligence_agent/graph.py). See that
    module's docstring for the graph topology.
    """
    if file is None or file.filename is None or file.filename.strip() == "":
        raise HTTPException(status_code=400, detail="No file provided or filename is missing.")

    filename = file.filename
    content = await file.read()
    logger.info(f"Received upload: {filename} ({len(content)} bytes)")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum upload size of {settings.MAX_UPLOAD_SIZE_MB}MB "
                   f"(got {len(content) / (1024 * 1024):.1f}MB).",
        )

    result = run_schema_intelligence_graph(
        filename=filename,
        content=content,
        force_reclassify=force_reclassify,
    )

    if isinstance(result, JSONResponse):
        return result

    return UploadResponse(**result)


@router.get(
    "/datasets",
    response_model=list[DatasetListItem],
    summary="List all uploaded datasets",
    description="Returns a summary list of all datasets that have been uploaded and processed.",
)
async def list_datasets():
    """List all uploaded dataset metadata records."""
    all_metadata = list_all_metadata()
    return [
        DatasetListItem(
            dataset_id=m.dataset_id,
            dataset_name=m.dataset_name,
            business_domain=m.business_domain,
            sub_domain=m.sub_domain,
            row_count=m.row_count,
            column_count=m.column_count,
            upload_timestamp=m.upload_timestamp,
            processing_status=m.processing_status,
        )
        for m in all_metadata
    ]


@router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetMetadata,
    summary="Get dataset metadata",
    description="Returns the full metadata record for a specific dataset.",
    responses={404: {"description": "Dataset not found"}},
)
async def get_dataset_metadata(dataset_id: str):
    """Get full metadata for a specific dataset."""
    metadata = get_metadata(dataset_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return metadata


@router.get(
    "/datasets/{dataset_id}/dataframe",
    summary="Get dataset DataFrame as JSON",
    description="Returns the in-memory DataFrame for a dataset as JSON records. "
                "Intended for downstream agent consumption.",
    responses={404: {"description": "Dataset not found or DataFrame not in memory"}},
)
async def get_dataset_dataframe(dataset_id: str, limit: int = 100):
    """
    Get the stored DataFrame as JSON records.
    
    Args:
        dataset_id: The dataset identifier.
        limit: Maximum number of rows to return (default 100, use -1 for all).
    """
    df = get_dataframe(dataset_id)
    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"DataFrame for '{dataset_id}' not found in memory. "
                   "It may have been evicted or the server restarted since it was uploaded — "
                   "re-upload the file to restore it."
        )

    if limit > 0:
        result_df = df.head(limit)
    else:
        result_df = df

    records = result_df.to_dict(orient="records")
    return {
        "dataset_id": dataset_id,
        "total_rows": len(df),
        "returned_rows": len(records),
        "data": records,
    }
