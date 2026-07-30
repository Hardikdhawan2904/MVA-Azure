"""Database session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings


def create_db_engine(settings: Settings):
    """Create SQLAlchemy engine from settings."""
    return create_engine(
        settings.database_url,
        echo=settings.database_echo,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )


def create_session_factory(settings: Settings) -> sessionmaker[Session]:
    """Create a session factory bound to the configured engine."""
    engine = create_db_engine(settings)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)
