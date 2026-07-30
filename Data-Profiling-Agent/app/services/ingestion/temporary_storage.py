"""Temporary file storage — run-scoped directory with cleanup guarantees."""

import uuid
import shutil
from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class TemporaryStorage:
    """Manages run-scoped temporary directories for uploaded files."""

    def __init__(self, settings: Settings):
        self._base_dir = Path(settings.temp_storage_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def create_run_directory(self, run_id: uuid.UUID) -> Path:
        """Create an isolated temporary directory for a processing run."""
        run_dir = self._base_dir / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_upload(self, run_id: uuid.UUID, content: bytes, safe_filename: str) -> Path:
        """
        Save uploaded file content to the run directory.

        Uses a sanitized filename to prevent path traversal.
        The original filename is NOT used as-is on the filesystem.
        """
        run_dir = self.create_run_directory(run_id)

        # Use only the base name, strip path components
        safe_name = Path(safe_filename).name
        # Additional safety: replace any remaining suspicious chars
        safe_name = safe_name.replace("..", "").replace("/", "").replace("\\", "")
        if not safe_name:
            safe_name = f"upload_{run_id.hex[:8]}"

        file_path = run_dir / safe_name
        file_path.write_bytes(content)

        logger.info(
            "file_saved",
            run_id=str(run_id),
            path=str(file_path),
            size_bytes=len(content),
        )
        return file_path

    def cleanup_run(self, run_id: uuid.UUID) -> bool:
        """
        Delete the entire run directory and its contents.

        Returns True if cleanup succeeded, False if directory didn't exist.
        """
        run_dir = self._base_dir / str(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
            logger.info("run_cleanup_complete", run_id=str(run_id))
            return True
        return False

    def get_run_directory(self, run_id: uuid.UUID) -> Path | None:
        """Get the run directory path if it exists."""
        run_dir = self._base_dir / str(run_id)
        return run_dir if run_dir.exists() else None

    def cleanup_all(self) -> int:
        """Remove all temporary directories. Returns count of removed dirs."""
        if not self._base_dir.exists():
            return 0
        count = 0
        for child in self._base_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                count += 1
        return count
