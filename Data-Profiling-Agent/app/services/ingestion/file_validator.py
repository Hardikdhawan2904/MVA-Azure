"""File validation service — checks extension, size, and basic readability."""

from pathlib import Path

from app.core.config import Settings
from app.core.constants import SUPPORTED_EXTENSIONS
from app.core.enums import FileType
from app.core.exceptions import FileValidationError


class FileValidator:
    """Validates uploaded files against configured constraints."""

    def __init__(self, settings: Settings):
        self._max_size_bytes = settings.max_upload_size_bytes

    def validate_extension(self, filename: str) -> FileType:
        """Validate file extension and return the detected FileType."""
        if not filename:
            raise FileValidationError(
                code="MISSING_FILENAME",
                message="No filename was provided.",
                details={},
            )

        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise FileValidationError(
                code="UNSUPPORTED_FILE_TYPE",
                message=f"File type '{suffix}' is not supported. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
                details={"extension": suffix, "allowed": sorted(SUPPORTED_EXTENSIONS)},
            )

        if suffix == ".csv":
            return FileType.CSV
        return FileType.XLSX

    def validate_size(self, size_bytes: int, filename: str) -> None:
        """Validate file size against configured maximum."""
        if size_bytes <= 0:
            raise FileValidationError(
                code="EMPTY_FILE",
                message="The uploaded file is empty.",
                details={"filename": filename, "size_bytes": size_bytes},
            )

        if size_bytes > self._max_size_bytes:
            raise FileValidationError(
                code="FILE_SIZE_EXCEEDED",
                message=(
                    f"The uploaded file ({size_bytes / (1024*1024):.1f} MB) exceeds "
                    f"the maximum allowed size ({self._max_size_bytes / (1024*1024):.0f} MB)."
                ),
                details={
                    "filename": filename,
                    "actual_bytes": size_bytes,
                    "maximum_bytes": self._max_size_bytes,
                },
            )

    def validate(self, filename: str, size_bytes: int) -> FileType:
        """Run all validations and return the file type."""
        file_type = self.validate_extension(filename)
        self.validate_size(size_bytes, filename)
        return file_type
