"""Throwaway copies of the committed demo databases, for tests and evaluations.

Notebook 01 dates the sample data relative to the day it ran, so once those dates pass, searches return
nothing and bookings fail. The copies have their dates moved to today: experiences start tomorrow (same
order and spacing), and the seeded reservations count toward this month's quota.
"""
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic import config
from data.models import cultpass


def move_dates_to_today(cultpass_db: Path) -> None:
    engine = create_engine(f"sqlite:///{cultpass_db}")
    try:
        with Session(engine) as session:
            experiences = session.query(cultpass.Experience).all()
            now = datetime.now()
            shift = now + timedelta(days=1) - min(e.when for e in experiences)
            for e in experiences:
                e.when += shift
            for r in session.query(cultpass.Reservation).all():
                r.created_at = now
            session.commit()
    finally:
        engine.dispose()


def copy_demo_databases(target_dir: Path, udahub_db: Path = None, cultpass_db: Path = None) -> tuple[Path, Path]:
    """Copy udahub.db and cultpass.db into target_dir, move the dates to today, and return the copies'
    paths (udahub, cultpass). Sources default to the paths in agentic.config."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    copies = []
    for source in (udahub_db or config.UDAHUB_DB_PATH, cultpass_db or config.CULTPASS_DB_PATH):
        if not Path(source).exists():
            raise FileNotFoundError(f"{source} missing: run 01_external_db_setup and 02_core_db_setup first")
        copies.append(Path(shutil.copy(source, target_dir / Path(source).name)))
    move_dates_to_today(copies[1])
    return copies[0], copies[1]
