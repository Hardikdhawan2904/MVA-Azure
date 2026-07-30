"""Tests for file validation."""

import pytest

from app.core.config import Settings
from app.core.enums import FileType
from app.core.exceptions import FileValidationError
from app.services.ingestion.file_validator import FileValidator


@pytest.fixture
def validator() -> FileValidator:
    settings = Settings(DATABASE_URL="postgresql://x:x@localhost/test")
    return FileValidator(settings)


class TestFileValidator:
    """Test file validation logic."""

    def test_valid_csv_extension(self, validator: FileValidator):
        result = validator.validate_extension("data.csv")
        assert result == FileType.CSV

    def test_valid_xlsx_extension(self, validator: FileValidator):
        result = validator.validate_extension("report.xlsx")
        assert result == FileType.XLSX

    def test_case_insensitive_extension(self, validator: FileValidator):
        result = validator.validate_extension("DATA.CSV")
        assert result == FileType.CSV

    def test_unsupported_extension(self, validator: FileValidator):
        with pytest.raises(FileValidationError) as exc_info:
            validator.validate_extension("data.json")
        assert exc_info.value.code == "UNSUPPORTED_FILE_TYPE"

    def test_no_extension(self, validator: FileValidator):
        with pytest.raises(FileValidationError) as exc_info:
            validator.validate_extension("noextension")
        assert exc_info.value.code == "UNSUPPORTED_FILE_TYPE"

    def test_empty_filename(self, validator: FileValidator):
        with pytest.raises(FileValidationError) as exc_info:
            validator.validate_extension("")
        assert exc_info.value.code == "MISSING_FILENAME"

    def test_valid_size(self, validator: FileValidator):
        # Should not raise
        validator.validate_size(1024, "test.csv")

    def test_empty_file_rejected(self, validator: FileValidator):
        with pytest.raises(FileValidationError) as exc_info:
            validator.validate_size(0, "empty.csv")
        assert exc_info.value.code == "EMPTY_FILE"

    def test_oversized_file_rejected(self, validator: FileValidator):
        # Default max is 25 MB
        size = 26 * 1024 * 1024
        with pytest.raises(FileValidationError) as exc_info:
            validator.validate_size(size, "large.csv")
        assert exc_info.value.code == "FILE_SIZE_EXCEEDED"

    def test_validate_combined(self, validator: FileValidator):
        result = validator.validate("report.xlsx", 5000)
        assert result == FileType.XLSX

    def test_validate_rejects_pdf(self, validator: FileValidator):
        with pytest.raises(FileValidationError):
            validator.validate("report.pdf", 5000)
