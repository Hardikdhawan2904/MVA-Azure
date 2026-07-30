"""SQLAlchemy declarative base and metadata."""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models. Tables live in the `agent2` schema
    (one shared Postgres database, namespaced per agent)."""
    metadata = MetaData(schema="agent2")
