"""Run the services in-process against copies of the committed SQLite databases.

The agent's calls to the other services go through FastAPI TestClients, so every test exercises the real
HTTP APIs, validation and internal-token checks, without Docker or Postgres.
"""
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "solution"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from agentic import config  # noqa: E402
from agentic.remote import http  # noqa: E402
import agentic.tools.knowledge_tools as kt  # noqa: E402
from data.models import cultpass  # noqa: E402

TOKEN = "test-internal-token"


def _move_dates_to_today(db_path: Path) -> None:
    """Same as solution/tests/conftest.py: experiences start tomorrow (same order and spacing), and the
    seeded reservations count toward this month's quota, whatever day the tests run."""
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        experiences = session.query(cultpass.Experience).all()
        now = datetime.now()
        shift = now + timedelta(days=1) - min(e.when for e in experiences)
        for e in experiences:
            e.when += shift
        for r in session.query(cultpass.Reservation).all():
            r.created_at = now
        session.commit()
    engine.dispose()


@pytest.fixture(autouse=True)
def databases(tmp_path, monkeypatch):
    """Fresh copies of the demo databases, with the dates moved to today so booking works
    whatever today's date is."""
    for name, path in [("UDAHUB_DB_PATH", config.UDAHUB_DB_PATH), ("CULTPASS_DB_PATH", config.CULTPASS_DB_PATH)]:
        copy = tmp_path / path.name
        shutil.copy(path, copy)
        monkeypatch.setattr(config, name, copy)
    _move_dates_to_today(config.CULTPASS_DB_PATH)
    monkeypatch.setattr(config, "EVENT_LOG_PATH", tmp_path / "events.jsonl")
    monkeypatch.setenv("INTERNAL_API_TOKEN", TOKEN)
    monkeypatch.setattr(config, "INTERNAL_API_TOKEN", TOKEN)
    # Offline: keyword retrieval instead of embeddings
    monkeypatch.setattr(kt, "get_embeddings", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    kt._retrievers.clear()
    yield


@pytest.fixture
def cultpass_client():
    from service.cultpass_service.main import app
    with TestClient(app, headers={"X-Internal-Token": TOKEN}) as client:
        yield client


@pytest.fixture
def ticket_client():
    from service.ticket_service.main import app
    with TestClient(app, headers={"X-Internal-Token": TOKEN}) as client:
        yield client


@pytest.fixture
def knowledge_client():
    from service.knowledge_service.main import app
    with TestClient(app, headers={"X-Internal-Token": TOKEN}) as client:
        yield client


@pytest.fixture
def remote_services(cultpass_client, ticket_client, knowledge_client):
    """Route the agentic data functions to the services, as in the agent container.

    The clients send no token of their own here: agentic.remote.http adds it, as it does in production.
    """
    calls = []
    for name, client in (("cultpass", cultpass_client), ("tickets", ticket_client), ("knowledge", knowledge_client)):
        client.headers.pop("X-Internal-Token", None)
        client.event_hooks["request"] = [lambda request, name=name: calls.append((name, request.method, request.url.path))]
        http.use_client(name, client)
    yield calls
    for name in ("cultpass", "tickets", "knowledge"):
        http.use_client(name, None)
