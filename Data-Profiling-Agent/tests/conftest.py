"""Shared test fixtures."""

import pytest
from pathlib import Path

from app.core.config import Settings
from app.repositories.configuration_repository import ConfigurationRepository


@pytest.fixture
def settings() -> Settings:
    """Provide test settings."""
    return Settings(
        DATABASE_URL="postgresql://test:test@localhost:5432/mva_test",
        TEMP_STORAGE_DIR="./tmp/test_uploads",
        MAX_UPLOAD_SIZE_MB=25,
        MAX_DATASET_ROWS=200000,
        MAX_DATASET_COLUMNS=200,
        LOG_LEVEL="DEBUG",
        LOG_FORMAT="console",
    )


@pytest.fixture
def config_repo() -> ConfigurationRepository:
    """Provide a configuration repository pointing to the real config dir."""
    config_dir = Path(__file__).parent.parent / "config"
    return ConfigurationRepository(config_dir=config_dir)


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test file uploads."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir
