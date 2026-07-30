"""Tests for CSV file loading."""

import pytest
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import FileValidationError, DatasetLimitError
from app.services.ingestion.csv_loader import CSVLoader


@pytest.fixture
def loader() -> CSVLoader:
    settings = Settings(
        DATABASE_URL="postgresql://x:x@localhost/test",
        MAX_DATASET_ROWS=1000,
        MAX_DATASET_COLUMNS=50,
    )
    return CSVLoader(settings)


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    """Create a simple valid CSV file."""
    content = "id,name,amount\n1,Alice,100.50\n2,Bob,200.75\n3,Charlie,50.00\n"
    path = tmp_path / "test.csv"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def semicolon_csv(tmp_path: Path) -> Path:
    """Create a semicolon-delimited CSV."""
    content = "id;name;amount\n1;Alice;100.50\n2;Bob;200.75\n"
    path = tmp_path / "semicolon.csv"
    path.write_text(content, encoding="utf-8")
    return path


class TestCSVLoader:
    """Test CSV loading and validation."""

    def test_load_valid_csv(self, loader: CSVLoader, csv_file: Path):
        df = loader.load(csv_file)
        assert len(df) == 3
        assert list(df.columns) == ["id", "name", "amount"]

    def test_load_semicolon_delimiter(self, loader: CSVLoader, semicolon_csv: Path):
        df = loader.load(semicolon_csv)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "amount"]

    def test_load_preserves_column_names(self, loader: CSVLoader, tmp_path: Path):
        content = "Transaction ID,Customer Name,Total Amount\n1,Alice,100\n"
        path = tmp_path / "spaces.csv"
        path.write_text(content)
        df = loader.load(path)
        assert "Transaction ID" in df.columns
        assert "Customer Name" in df.columns

    def test_empty_file_rejected(self, loader: CSVLoader, tmp_path: Path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(FileValidationError) as exc_info:
            loader.load(path)
        assert exc_info.value.code in ("CSV_PARSE_ERROR", "EMPTY_DATASET")

    def test_header_only_rejected(self, loader: CSVLoader, tmp_path: Path):
        path = tmp_path / "header_only.csv"
        path.write_text("col1,col2,col3\n")
        with pytest.raises(FileValidationError) as exc_info:
            loader.load(path)
        assert exc_info.value.code == "NO_DATA_ROWS"

    def test_row_limit_exceeded(self, tmp_path: Path):
        settings = Settings(
            DATABASE_URL="postgresql://x:x@localhost/test",
            MAX_DATASET_ROWS=5,
            MAX_DATASET_COLUMNS=50,
        )
        loader = CSVLoader(settings)
        lines = ["id,val"] + [f"{i},{i*10}" for i in range(10)]
        path = tmp_path / "big.csv"
        path.write_text("\n".join(lines))
        with pytest.raises(DatasetLimitError) as exc_info:
            loader.load(path)
        assert exc_info.value.code == "DATASET_ROW_LIMIT_EXCEEDED"

    def test_column_limit_exceeded(self, tmp_path: Path):
        settings = Settings(
            DATABASE_URL="postgresql://x:x@localhost/test",
            MAX_DATASET_ROWS=1000,
            MAX_DATASET_COLUMNS=3,
        )
        loader = CSVLoader(settings)
        header = ",".join([f"col{i}" for i in range(5)])
        row = ",".join(["val"] * 5)
        path = tmp_path / "wide.csv"
        path.write_text(f"{header}\n{row}\n")
        with pytest.raises(DatasetLimitError) as exc_info:
            loader.load(path)
        assert exc_info.value.code == "DATASET_COLUMN_LIMIT_EXCEEDED"

    def test_file_not_found(self, loader: CSVLoader, tmp_path: Path):
        path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileValidationError) as exc_info:
            loader.load(path)
        assert exc_info.value.code == "FILE_NOT_FOUND"

    def test_encoding_detection(self, loader: CSVLoader, tmp_path: Path):
        # Write a file with latin-1 encoding
        content = "name,city\nJosé,São Paulo\nMüller,München\n"
        path = tmp_path / "latin.csv"
        path.write_bytes(content.encode("latin-1"))
        df = loader.load(path)
        assert len(df) == 2

    def test_all_loaded_as_string_dtype(self, loader: CSVLoader, csv_file: Path):
        df = loader.load(csv_file)
        # All columns should be object/string since we load with dtype=str
        for col in df.columns:
            assert df[col].dtype == object
