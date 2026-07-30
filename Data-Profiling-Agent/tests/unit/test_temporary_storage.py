"""Tests for temporary file storage."""

import uuid
import pytest
from pathlib import Path

from app.core.config import Settings
from app.services.ingestion.temporary_storage import TemporaryStorage


@pytest.fixture
def storage(tmp_path: Path) -> TemporaryStorage:
    settings = Settings(
        DATABASE_URL="postgresql://x:x@localhost/test",
        TEMP_STORAGE_DIR=str(tmp_path / "uploads"),
    )
    return TemporaryStorage(settings)


class TestTemporaryStorage:
    """Test temporary file storage management."""

    def test_create_run_directory(self, storage: TemporaryStorage):
        run_id = uuid.uuid4()
        run_dir = storage.create_run_directory(run_id)
        assert run_dir.exists()
        assert run_dir.is_dir()
        assert str(run_id) in str(run_dir)

    def test_save_upload(self, storage: TemporaryStorage):
        run_id = uuid.uuid4()
        content = b"col1,col2\nval1,val2\n"
        path = storage.save_upload(run_id, content, "data.csv")
        assert path.exists()
        assert path.read_bytes() == content

    def test_save_upload_prevents_path_traversal(self, storage: TemporaryStorage):
        run_id = uuid.uuid4()
        content = b"test"
        path = storage.save_upload(run_id, content, "../../etc/passwd")
        # Should NOT create file outside the run directory
        assert str(run_id) in str(path.parent)
        assert path.exists()

    def test_cleanup_run(self, storage: TemporaryStorage):
        run_id = uuid.uuid4()
        content = b"data"
        path = storage.save_upload(run_id, content, "test.csv")
        assert path.exists()

        result = storage.cleanup_run(run_id)
        assert result is True
        assert not path.exists()
        assert not path.parent.exists()

    def test_cleanup_nonexistent_run(self, storage: TemporaryStorage):
        run_id = uuid.uuid4()
        result = storage.cleanup_run(run_id)
        assert result is False

    def test_get_run_directory_exists(self, storage: TemporaryStorage):
        run_id = uuid.uuid4()
        storage.create_run_directory(run_id)
        result = storage.get_run_directory(run_id)
        assert result is not None
        assert result.exists()

    def test_get_run_directory_not_exists(self, storage: TemporaryStorage):
        run_id = uuid.uuid4()
        result = storage.get_run_directory(run_id)
        assert result is None

    def test_cleanup_all(self, storage: TemporaryStorage):
        for _ in range(3):
            run_id = uuid.uuid4()
            storage.save_upload(run_id, b"x", "f.csv")

        count = storage.cleanup_all()
        assert count == 3
