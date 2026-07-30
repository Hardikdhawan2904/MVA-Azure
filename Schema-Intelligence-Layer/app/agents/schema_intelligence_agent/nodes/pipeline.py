"""Schema Intelligence Agent graph nodes.

Mirrors the exact validation/loading/quality-gate/classification logic the old
linear upload_dataset() route handler used (app/routes/upload.py, pre-graph
version) — this file does not change what gets validated or how classification
degrades on failure, it only re-expresses each stage as a graph node with an
explicit conditional edge, instead of a chain of if/else branches inside one
function body.

Two real decision points: quality_gate_router (stop the whole upload before any
LLM call if the dataset fails the deterministic quality gate) and
classification_needed_router (skip the classification agent — and every LLM
call in this service — entirely when replacing an already-classified
dataset without force_reclassify).
"""

from __future__ import annotations

import io
import logging
from typing import Any, Literal

import pandas as pd
from fastapi import HTTPException

from app.config import settings
from app.datastore.registry import store_dataframe
from app.agents.classification_agent import run_classification_agent
from app.agents.schema_intelligence_agent.state import SchemaIntelligenceState
from app.services.database import (
    get_metadata_by_name,
    get_next_dataset_id,
    insert_metadata,
    update_metadata_after_reupload,
    update_metadata_after_classification,
)
from app.services.metadata_extractor import extract_metadata
from app.services.quality_validator import DataQualityValidator
from app.services.validator import SUPPORTED_EXTENSIONS, get_file_extension

logger = logging.getLogger(__name__)


# ── Node 1: validate + load ─────────────────────────────────────────────────

def validate_and_load(state: SchemaIntelligenceState) -> dict[str, Any]:
    filename = state["filename"]
    content = state["content"]

    file_extension = get_file_extension(filename)
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file_extension}'. "
                   f"Accepted formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    buffer = io.BytesIO(content)
    try:
        if file_extension == ".csv":
            df = pd.read_csv(buffer)
        else:
            df = pd.read_excel(buffer, engine="openpyxl")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read the uploaded file: {str(e)}")

    if df is None or df.empty:
        raise HTTPException(status_code=422, detail=f"Uploaded dataset '{filename}' is empty (0 rows).")
    if len(df.columns) == 0:
        raise HTTPException(
            status_code=422, detail=f"File '{filename}' does not contain valid tabular data (no columns detected)."
        )
    # MAX_UPLOAD_SIZE_MB alone doesn't bound row/column count — a highly
    # compressible file well under the byte cap can still expand into
    # millions of rows once parsed. Same limit names/values as Agent 2's
    # MAX_DATASET_ROWS/MAX_DATASET_COLUMNS.
    if len(df) > settings.MAX_DATASET_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset '{filename}' has {len(df)} rows, exceeding the maximum of "
                   f"{settings.MAX_DATASET_ROWS}.",
        )
    if len(df.columns) > settings.MAX_DATASET_COLUMNS:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset '{filename}' has {len(df.columns)} columns, exceeding the maximum of "
                   f"{settings.MAX_DATASET_COLUMNS}.",
        )

    return {"file_extension": file_extension, "df": df}


# ── Node 2: existing-dataset lookup (replace vs new) ────────────────────────
# A filename this service has already seen replaces that prior entry rather
# than accumulating onto it — every response (row counts, quality report,
# duplicate-row checks, everything) always reflects exactly the file that
# was just uploaded, never a silent blend with an earlier upload under the
# same name. run_quality_gate still surfaces an informational note when a
# replacement happens (see its own comment) so the "you just uploaded this
# again" signal isn't lost, it just no longer corrupts the numbers.

def check_existing_dataset(state: SchemaIntelligenceState) -> dict[str, Any]:
    existing = get_metadata_by_name(state["filename"])
    if existing is None:
        return {"existing_metadata": None, "is_new": True, "profiling_df": state["df"]}

    logger.info(
        f"Dataset '{state['filename']}' already exists (ID: {existing.dataset_id}, "
        f"{existing.row_count} rows) — replacing with this upload ({len(state['df'])} rows)."
    )
    return {"existing_metadata": existing, "is_new": False, "profiling_df": state["df"]}


# ── Node 3: deterministic quality gate ──────────────────────────────────────
# Validates profiling_df (this upload alone — see check_existing_dataset for
# why it's never combined with a prior upload under the same filename).

def run_quality_gate(state: SchemaIntelligenceState) -> dict[str, Any]:
    report = DataQualityValidator().run_validation(state["profiling_df"])

    existing = state.get("existing_metadata")
    if existing is not None:
        report.setdefault("warnings", []).append(
            f"This filename was previously uploaded on {existing.upload_timestamp} "
            f"({existing.row_count} rows) — this upload ({len(state['profiling_df'])} rows) "
            "replaces that version rather than being combined with it."
        )

    if report["decision"] == "FAIL":
        logger.warning(
            f"Dataset '{state['filename']}' failed data quality validation "
            f"(Score: {report['dataset_score']}/{report['passing_score']}). Aborting."
        )
        return {"quality_report": report, "error_status_code": 422, "error_content": report}
    return {"quality_report": report}


def quality_gate_router(state: SchemaIntelligenceState) -> Literal["fail", "pass"]:
    return "fail" if state.get("error_content") is not None else "pass"


# ── Node 4: metadata extraction (+ initial persistence) ─────────────────────

def extract_metadata_node(state: SchemaIntelligenceState) -> dict[str, Any]:
    filename = state["filename"]
    file_extension = state["file_extension"]
    report = state["quality_report"]
    profiling_df = state["profiling_df"]

    if state["is_new"]:
        logger.info(f"Dataset '{filename}' is new. Creating metadata catalog entry.")
        dataset_id = get_next_dataset_id()
        metadata = extract_metadata(profiling_df, filename, dataset_id, file_extension)
        metadata.quality_score = report["dataset_score"]
        metadata.quality_report = report
        insert_metadata(metadata)
    else:
        existing = state["existing_metadata"]
        metadata = extract_metadata(profiling_df, filename, existing.dataset_id, file_extension)
        metadata.quality_score = report["dataset_score"]
        metadata.quality_report = report
        update_metadata_after_reupload(
            dataset_id=existing.dataset_id,
            row_count=metadata.row_count,
            column_count=metadata.column_count,
            column_names=metadata.column_names,
            column_data_types=metadata.column_data_types,
            upload_timestamp=metadata.upload_timestamp,
            sample_data=metadata.sample_data,
            processing_status="Completed",
        )

    return {"metadata": metadata}


def classification_needed_router(state: SchemaIntelligenceState) -> Literal["classify", "reuse"]:
    return "classify" if state["is_new"] or state.get("force_reclassify") else "reuse"


# ── Node 5a: classification agent (full path) ───────────────────────────────

def classify_dataset_node(state: SchemaIntelligenceState) -> dict[str, Any]:
    metadata = state["metadata"]
    is_new = state["is_new"]

    try:
        descriptions, classification = run_classification_agent(metadata)
        logger.info(f"Generated column descriptions for {metadata.dataset_id}")
        logger.info(
            f"Classified {metadata.dataset_id} as '{classification.business_domain}' "
            f"/ '{classification.sub_domain}'"
        )
        return {
            "column_descriptions": descriptions,
            "business_domain": classification.business_domain,
            "sub_domain": classification.sub_domain,
            "dataset_summary": classification.dataset_summary,
            "processing_status": "Completed",
        }
    except Exception as e:
        logger.error(f"Dataset classification failed for {metadata.dataset_id}: {e}")
        return {
            "column_descriptions": {col: f"Column '{col}'" for col in metadata.column_names},
            "business_domain": "Other",
            "sub_domain": "General",
            "dataset_summary": "Classification failed — domain could not be determined.",
            "processing_status": "Partial" if is_new else "Completed",
        }


# ── Node 5b: reuse existing classification (skip path) ──────────────────────

def reuse_existing_classification_node(state: SchemaIntelligenceState) -> dict[str, Any]:
    existing = state["existing_metadata"]
    return {
        "column_descriptions": existing.column_descriptions,
        "business_domain": existing.business_domain,
        "sub_domain": existing.sub_domain,
        "dataset_summary": existing.dataset_summary,
        "processing_status": "Completed",
    }


# ── Node 6: persist + build response ────────────────────────────────────────

def persist_and_finalize(state: SchemaIntelligenceState) -> dict[str, Any]:
    metadata = state["metadata"]
    report = state["quality_report"]
    profiling_df = state["profiling_df"]

    update_metadata_after_classification(
        dataset_id=metadata.dataset_id,
        business_domain=state["business_domain"],
        sub_domain=state["sub_domain"],
        dataset_summary=state["dataset_summary"],
        column_descriptions=state["column_descriptions"],
        quality_score=report["dataset_score"],
        quality_report=report,
        processing_status=state["processing_status"],
    )

    store_dataframe(metadata.dataset_id, profiling_df)

    response = {
        "dataset_id": metadata.dataset_id,
        "dataset_name": metadata.dataset_name,
        "business_domain": state["business_domain"],
        "sub_domain": state["sub_domain"],
        "dataset_summary": state["dataset_summary"],
        "row_count": metadata.row_count,
        "column_count": metadata.column_count,
        "column_descriptions": state["column_descriptions"],
        "status": state["processing_status"],
        "dataframe_records": profiling_df.to_dict(orient="records"),
        "quality_report": report,
    }
    return {"response": response}
