"""CSV file loader — handles encoding detection, delimiter detection, and validation."""

from pathlib import Path

import chardet
import pandas as pd

from app.core.config import Settings
from app.core.exceptions import FileValidationError, DatasetLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)


class CSVLoader:
    """Load and validate CSV files into a pandas DataFrame."""

    def __init__(self, settings: Settings):
        self._max_rows = settings.max_dataset_rows
        self._max_columns = settings.max_dataset_columns

    def detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        raw = file_path.read_bytes()
        # Use a bounded sample for large files
        sample = raw[:65536]
        result = chardet.detect(sample)
        encoding = result.get("encoding") or "utf-8"
        confidence = result.get("confidence", 0)

        logger.debug(
            "encoding_detected",
            encoding=encoding,
            confidence=confidence,
            path=str(file_path),
        )
        return encoding

    def detect_delimiter(self, file_path: Path, encoding: str) -> str:
        """Detect CSV delimiter by analyzing the first few lines."""
        import csv

        try:
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                sample = f.read(8192)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except csv.Error:
            # Default to comma if detection fails
            return ","

    def load(self, file_path: Path) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.

        Validates:
        - File is readable
        - Has a header row (at least one column)
        - Does not exceed row/column limits

        Returns the loaded DataFrame.
        """
        if not file_path.exists():
            raise FileValidationError(
                code="FILE_NOT_FOUND",
                message="The file could not be found for processing.",
                details={"path": str(file_path)},
            )

        encoding = self.detect_encoding(file_path)
        delimiter = self.detect_delimiter(file_path, encoding)

        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=delimiter,
                dtype=str,  # Load all as string initially for type refinement
                keep_default_na=True,
                na_values=["", "NA", "N/A", "null", "NULL", "None", "none", "nan", "NaN"],
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            raise FileValidationError(
                code="CSV_PARSE_ERROR",
                message=f"Failed to parse CSV file: {str(e)}",
                details={"error": str(e)},
            )
        except UnicodeDecodeError as e:
            raise FileValidationError(
                code="ENCODING_ERROR",
                message=f"Failed to decode file with detected encoding '{encoding}'.",
                details={"encoding": encoding, "error": str(e)},
            )

        self._validate_dataframe(df)
        return df

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate the loaded dataframe against configured limits."""
        if len(df.columns) == 0:
            raise FileValidationError(
                code="EMPTY_DATASET",
                message="The file contains no data or no columns.",
                details={"rows": len(df), "columns": len(df.columns)},
            )

        if len(df) == 0:
            raise FileValidationError(
                code="NO_DATA_ROWS",
                message="The file contains a header but no data rows.",
                details={"columns": len(df.columns)},
            )

        if len(df) > self._max_rows:
            raise DatasetLimitError(
                code="DATASET_ROW_LIMIT_EXCEEDED",
                message=(
                    f"The dataset contains {len(df):,} rows. "
                    f"The configured maximum is {self._max_rows:,} rows."
                ),
                details={"actual_rows": len(df), "maximum_rows": self._max_rows},
            )

        if len(df.columns) > self._max_columns:
            raise DatasetLimitError(
                code="DATASET_COLUMN_LIMIT_EXCEEDED",
                message=(
                    f"The dataset contains {len(df.columns)} columns. "
                    f"The configured maximum is {self._max_columns} columns."
                ),
                details={"actual_columns": len(df.columns), "maximum_columns": self._max_columns},
            )
