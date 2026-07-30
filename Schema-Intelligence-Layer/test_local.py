"""
Local test script — runs the full Dataset Identification Agent pipeline
directly from the command line, no server required.

Usage:
    python test_local.py                          # Tests all files in ../test_data/
    python test_local.py ../test_data/banking_variance_data.csv   # Test a single file
"""

import sys
import os
import json
import logging
import pandas as pd
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from app.config import settings
from app.services.database import (
    init_db,
    get_next_dataset_id,
    insert_metadata,
    update_metadata_after_classification,
    get_metadata,
    get_metadata_by_name,
    update_metadata_after_append,
)
from app.services.metadata_extractor import extract_metadata
from app.services.llm_service import (
    generate_column_descriptions,
    classify_dataset,
)
from app.datastore.registry import store_dataframe, get_dataframe
from app.services.quality_validator import DataQualityValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_local")


def load_file(filepath: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    ext = Path(filepath).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def process_file(filepath: str) -> dict:
    """
    Run the full agent pipeline on a single file:
      1. Load into DataFrame
      2. Extract metadata
      3. Generate column descriptions (LLM)
      4. Classify business domain (LLM)
      5. Store in PostgreSQL + in-memory registry
    """
    filepath = str(Path(filepath).resolve())
    filename = Path(filepath).name
    ext = Path(filepath).suffix.lower()

    print(f"\n{'='*70}")
    print(f"  Processing: {filename}")
    print(f"{'='*70}")

    # Step 1: Load
    print("\n[1/5] Loading dataset...")
    df = load_file(filepath)
    print(f"      ✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Step 1.5: Run Data Quality Validator
    quality_validator = DataQualityValidator()
    report = quality_validator.run_validation(df)

    print("\n[1.5/5] Running Data Quality Validator...")
    print(f"      ✓ Score achieved: {report['dataset_score']} (Passing: {report['passing_score']})")
    print(f"      ✓ Decision: {report['decision']}")
    if report["warnings"]:
        print("      ⚠ Warnings:")
        for w in report["warnings"]:
            print(f"        - {w}")

    if report["decision"] == "FAIL":
        print(f"\n      ✗ Data quality validation failed. Aborting pipeline.")
        result = {
            "dataset_id": "N/A",
            "dataset_name": filename,
            "business_domain": "N/A",
            "sub_domain": "N/A",
            "dataset_summary": "Validation failed.",
            "row_count": len(df),
            "column_count": len(df.columns),
            "status": "FAIL",
            "dataframe_records": [],
        }
        
        # Print final result (metadata only, not full records)
        printable_result = {k: v for k, v in result.items() if k not in ("dataframe_records", "column_descriptions")}
        print(f"\n{'─'*70}")
        print("  FINAL RESULT  (Validation Failed)")
        print(f"{'─'*70}")
        print(json.dumps(printable_result, indent=2))
        print(f"{'─'*70}\n")
        return result

    # Check if dataset already exists
    existing = get_metadata_by_name(filename)

    if existing:
        print(f"\n      ℹ Dataset '{filename}' already exists (ID: {existing.dataset_id}). Appending rows.")
        # Step 2: Combine with the cached in-memory DataFrame, if still present
        print("\n[2/5] Combining with in-memory dataset...")
        current_df = get_dataframe(existing.dataset_id)
        if current_df is not None:
            combined_df = pd.concat([current_df, df], ignore_index=True)
        else:
            print("      ⚠ No cached DataFrame found (server/script restarted) — treating as replacement.")
            combined_df = df
        new_metadata = extract_metadata(combined_df, filename, existing.dataset_id, ext)

        # Step 5: Store results
        print("\n[5/5] Updating metadata catalog...")
        update_metadata_after_append(
            dataset_id=existing.dataset_id,
            row_count=new_metadata.row_count,
            column_count=new_metadata.column_count,
            column_names=new_metadata.column_names,
            column_data_types=new_metadata.column_data_types,
            upload_timestamp=new_metadata.upload_timestamp,
            sample_data=new_metadata.sample_data,
            processing_status="Completed",
        )
        new_metadata.quality_score = report["dataset_score"]
        new_metadata.quality_report = report

        update_metadata_after_classification(
            dataset_id=existing.dataset_id,
            business_domain=existing.business_domain,
            sub_domain=existing.sub_domain,
            dataset_summary=existing.dataset_summary,
            column_descriptions=existing.column_descriptions,
            quality_score=report["dataset_score"],
            quality_report=report,
            processing_status="Completed",
        )
        store_dataframe(existing.dataset_id, combined_df)
        print(f"      ✓ Catalog updated. Total row count is now {new_metadata.row_count}.")

        dataframe_records = combined_df.to_dict(orient="records")

        result = {
            "dataset_id": existing.dataset_id,
            "dataset_name": existing.dataset_name,
            "business_domain": existing.business_domain,
            "sub_domain": existing.sub_domain,
            "dataset_summary": existing.dataset_summary,
            "row_count": new_metadata.row_count,
            "column_count": new_metadata.column_count,
            "status": "Completed",
            "column_descriptions": existing.column_descriptions,
            "dataframe_records": dataframe_records,
        }

    else:
        # Step 2: Extract metadata
        print("\n[2/5] Extracting metadata...")
        dataset_id = get_next_dataset_id()

        metadata = extract_metadata(df, filename, dataset_id, ext)
        metadata.quality_score = report["dataset_score"]
        metadata.quality_report = report
        
        insert_metadata(metadata)
        print(f"      ✓ Dataset ID: {dataset_id}")
        print(f"      ✓ File type: {metadata.file_type}")

        # Step 3: Generate column descriptions via LLM
        print("\n[3/5] Generating column descriptions (Azure OpenAI)...")
        try:
            column_descriptions = generate_column_descriptions(metadata)
            metadata.column_descriptions = column_descriptions
            print(f"      ✓ Generated descriptions for {len(column_descriptions)} columns")
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            column_descriptions = {col: f"Column '{col}'" for col in metadata.column_names}
            metadata.column_descriptions = column_descriptions

        # Step 4: Classify dataset via LLM
        print("\n[4/5] Classifying business domain (Azure OpenAI)...")
        try:
            classification = classify_dataset(metadata)
            metadata.business_domain = classification.business_domain
            metadata.sub_domain = classification.sub_domain
            metadata.dataset_summary = classification.dataset_summary
            processing_status = "Completed"
            print(f"      ✓ Domain: {classification.business_domain}")
            print(f"      ✓ Sub-domain: {classification.sub_domain}")
            print(f"      ✓ Summary: {classification.dataset_summary}")
            if classification.confidence:
                print(f"      ✓ Confidence: {classification.confidence:.0%}")
            if classification.reason:
                print(f"      ✓ Reason: {classification.reason}")
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            metadata.business_domain = "Other"
            metadata.sub_domain = "General"
            metadata.dataset_summary = "Classification failed."
            processing_status = "Partial"

        # Step 5: Store results
        print("\n[5/5] Storing results...")
        metadata.processing_status = processing_status
        update_metadata_after_classification(
            dataset_id=dataset_id,
            business_domain=metadata.business_domain,
            sub_domain=metadata.sub_domain,
            dataset_summary=metadata.dataset_summary,
            column_descriptions=metadata.column_descriptions,
            quality_score=report["dataset_score"],
            quality_report=report,
            processing_status=processing_status,
        )
        store_dataframe(dataset_id, df)
        print(f"      ✓ Metadata saved to PostgreSQL ({settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB})")
        print(f"      ✓ DataFrame stored in memory registry")

        # Serialize DataFrame records (full, for downstream handoff)
        dataframe_records = df.to_dict(orient="records")

        # Build result summary (mirrors UploadResponse)
        result = {
            "dataset_id": metadata.dataset_id,
            "dataset_name": metadata.dataset_name,
            "business_domain": metadata.business_domain,
            "sub_domain": metadata.sub_domain,
            "dataset_summary": metadata.dataset_summary,
            "row_count": metadata.row_count,
            "column_count": metadata.column_count,
            "status": metadata.processing_status,
            "column_descriptions": metadata.column_descriptions,
            "dataframe_records": dataframe_records,  # full DataFrame for downstream
        }

    # Print final result (metadata only, not full records)
    printable_result = {k: v for k, v in result.items() if k not in ("dataframe_records", "column_descriptions")}
    print(f"\n{'─'*70}")
    print("  FINAL RESULT  (metadata + schema)")
    print(f"{'─'*70}")
    print(json.dumps(printable_result, indent=2))

    # Print DataFrame preview (first 5 rows)
    print(f"\n{'─'*70}")
    print(f"  DATAFRAME PREVIEW  (first 5 of {len(df)} rows)")
    print(f"{'─'*70}")
    print(df.head(5).to_string(index=False))
    print(f"{'─'*70}\n")

    return result


def main():
    # Force UTF-8 output encoding for Windows terminals
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    # Initialize database
    init_db()

    # Determine which files to process
    if len(sys.argv) > 1:
        # Process specific file(s) passed as arguments
        files = sys.argv[1:]
    else:
        # Process all files in the repo-root test_data/ (shared across all
        # services, not owned by this one specifically — see repo root README).
        test_data_dir = Path(__file__).parent.parent / "test_data"
        if not test_data_dir.exists():
            print("Error: ../test_data/ folder not found.")
            print("Usage: python test_local.py <file_path>")
            sys.exit(1)

        files = [
            str(f) for f in test_data_dir.iterdir()
            if f.suffix.lower() in (".csv", ".xlsx", ".xls")
        ]

        if not files:
            print("No CSV or Excel files found in ../test_data/")
            sys.exit(1)

    print(f"\n Dataset Identification Agent — Local Test")
    print(f"   Files to process: {len(files)}")
    print(f"   Azure OpenAI Deployment: {settings.AZURE_OPENAI_DEPLOYMENT}")
    print(f"   Database: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")

    results = []
    for filepath in files:
        try:
            result = process_file(filepath)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")
            print(f"\n  ✗ ERROR processing {filepath}: {e}\n")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {len(results)}/{len(files)} files processed successfully")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['dataset_id']} | {r['dataset_name']:<45} | {r['business_domain']:<20} | {r['status']}")
    print()


if __name__ == "__main__":
    main()
