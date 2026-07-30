"""FastAPI dependency injection configuration."""

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import create_session_factory


@lru_cache(maxsize=1)
def get_cached_settings() -> Settings:
    """Return cached application settings."""
    return get_settings()


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return cached session factory."""
    settings = get_cached_settings()
    return create_session_factory(settings)


def get_db() -> Generator[Session, None, None]:
    """Provide a transactional database session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()
