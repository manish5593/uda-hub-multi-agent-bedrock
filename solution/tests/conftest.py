import sys
from pathlib import Path

import pytest

STARTER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STARTER))

from agentic import config  # noqa: E402
from agentic.demo_data import copy_demo_databases  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_dbs(tmp_path, monkeypatch):
    """Run every test against fresh copies of the databases built by notebooks 01 and 02, with the dates
    moved to today (see agentic/demo_data.py)."""
    try:
        udahub_db, cultpass_db = copy_demo_databases(tmp_path)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    monkeypatch.setattr(config, "UDAHUB_DB_PATH", udahub_db)
    monkeypatch.setattr(config, "CULTPASS_DB_PATH", cultpass_db)
    monkeypatch.setattr(config, "EVENT_LOG_PATH", tmp_path / "events.jsonl")
    yield


def pytest_configure(config):
    # Using a LangGraph API deprecated in 1.x (to be removed in 2.0) fails the tests instead of only warning
    config.addinivalue_line("filterwarnings", "error::langgraph.warnings.LangGraphDeprecatedSinceV10")
