"""Run the services in-process against copies of the committed SQLite databases.

The agent's calls to the other services go through FastAPI TestClients, so every test exercises the real
HTTP APIs, validation and internal-token checks, without Docker or Postgres.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "solution"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastapi.testclient import TestClient  # noqa: E402

from agentic import config  # noqa: E402
from agentic.demo_data import copy_demo_databases  # noqa: E402
from agentic.remote import http  # noqa: E402
import agentic.tools.knowledge_tools as kt  # noqa: E402

TOKEN = "test-internal-token"


@pytest.fixture(autouse=True)
def databases(tmp_path, monkeypatch):
    """Fresh copies of the demo databases, with the dates moved to today so booking works
    whatever today's date is."""
    udahub_db, cultpass_db = copy_demo_databases(tmp_path)
    monkeypatch.setattr(config, "UDAHUB_DB_PATH", udahub_db)
    monkeypatch.setattr(config, "CULTPASS_DB_PATH", cultpass_db)
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


def pytest_configure(config):
    # Using a LangGraph API deprecated in 1.x (to be removed in 2.0) fails the tests instead of only warning
    config.addinivalue_line("filterwarnings", "error::langgraph.warnings.LangGraphDeprecatedSinceV10")
