"""
Metadata extraction service.
Extracts structured metadata from a loaded Pandas DataFrame.
"""

from datetime import datetime, timezone
import pandas as pd

from app.models.schemas import DatasetMetadata
from app.config import settings


def extract_metadata(
    df: pd.DataFrame,
    filename: str,
    dataset_id: str,
    file_extension: str,
) -> DatasetMetadata:
    """
    Extract metadata from a loaded DataFrame.
    """
    # Determine file type label
    file_type_map = {
        ".csv": "CSV",
        ".xlsx": "Excel (XLSX)",
        ".xls": "Excel (XLS)",
    }
    file_type = file_type_map.get(file_extension, "Unknown")

    column_names = df.columns.tolist()
    column_data_types = [str(dtype) for dtype in df.dtypes.tolist()]

    sample_count = min(settings.SAMPLE_ROWS, len(df))
    sample_df = df.head(sample_count)

    sample_data = []
    for _, row in sample_df.iterrows():
        row_dict = {}
        for col in column_names:
            val = row[col]
            if pd.isna(val):
                row_dict[col] = None
            elif hasattr(val, "item"):
                row_dict[col] = val.item()
            else:
                row_dict[col] = val
        sample_data.append(row_dict)

    return DatasetMetadata(
        dataset_id=dataset_id,
        dataset_name=filename,
        file_type=file_type,
        upload_timestamp=datetime.now(timezone.utc).isoformat(),
        row_count=len(df),
        column_count=len(column_names),
        column_names=column_names,
        column_data_types=column_data_types,
        sample_data=sample_data,
        processing_status="Processing",
    )
