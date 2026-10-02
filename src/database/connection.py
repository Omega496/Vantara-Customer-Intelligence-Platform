"""Database connection manager with PostgreSQL support and automatic SQLite fallback."""

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base
from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def get_engine() -> Engine:
    """Initializes and returns SQLAlchemy engine, falling back to SQLite if PostgreSQL is unreachable."""
    global _engine, _SessionFactory
    if _engine is not None:
        return _engine

    db_url = os.getenv("DATABASE_URL") or ConfigManager.get("database.url", "")
    fallback_db_path = Path("data/processed/retail.db").resolve()
    fallback_db_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_url = f"sqlite:///{fallback_db_path}"

    if db_url and db_url.startswith(("postgresql://", "postgres://", "postgresql+psycopg2://")):
        # Ensure psycopg2 driver dialect is specified if raw postgresql:// is provided
        driver_url = db_url
        if driver_url.startswith("postgresql://"):
            driver_url = driver_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        elif driver_url.startswith("postgres://"):
            driver_url = driver_url.replace("postgres://", "postgresql+psycopg2://", 1)

        try:
            logger.info(f"Attempting PostgreSQL connection: {driver_url.split('@')[-1]}")
            engine = create_engine(
                driver_url,
                pool_size=ConfigManager.get("database.pool_size", 10),
                max_overflow=ConfigManager.get("database.max_overflow", 20),
                connect_args={"connect_timeout": 2},
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Successfully connected to PostgreSQL database.")
            _engine = engine
        except Exception as e:
            logger.warning(
                f"PostgreSQL connection failed ({e}). Falling back to local SQLite at {fallback_url}"
            )
            _engine = create_engine(fallback_url, connect_args={"check_same_thread": False})
    else:
        logger.info(f"Using local SQLite database: {fallback_url}")
        _engine = create_engine(fallback_url, connect_args={"check_same_thread": False})

    _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def init_db(engine: Engine | None = None) -> None:
    """Creates all database tables defined in ORM models."""
    if engine is None:
        engine = get_engine()
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized successfully.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding transactional database sessions."""
    global _SessionFactory
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None

    db = _SessionFactory()
    try:
        yield db
    finally:
        db.close()
