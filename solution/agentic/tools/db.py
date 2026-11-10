"""Database engines and session helpers for the core (udahub) and external (cultpass) DBs.

By default both are the committed SQLite files. Setting UDAHUB_DB_URL or CULTPASS_DB_URL (any SQLAlchemy
URL, e.g. Postgres) points them elsewhere; the microservices in service/ do that.
"""
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from agentic import config


@lru_cache(maxsize=None)
def _engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        return create_engine(url, echo=False, connect_args={"check_same_thread": False})
    # Server databases: drop connections the server closed instead of failing the next query
    return create_engine(url, echo=False, pool_pre_ping=True)


def udahub_engine() -> Engine:
    return _engine(config.UDAHUB_DB_URL or f"sqlite:///{config.UDAHUB_DB_PATH}")


def cultpass_engine() -> Engine:
    return _engine(config.CULTPASS_DB_URL or f"sqlite:///{config.CULTPASS_DB_PATH}")


@contextmanager
def session_scope(engine: Engine):
    session: Session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
