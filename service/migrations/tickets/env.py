"""Alembic environment for the tickets database. The shared logic is in service/migrations/common.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from service.migrations.common import run_env  # noqa: E402

run_env("tickets")
