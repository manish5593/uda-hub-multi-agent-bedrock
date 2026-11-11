"""Copy the committed demo data from the SQLite files into each service's database.

    python -m service.scripts.migrate     # first: creates the tables
    python -m service.scripts.seed

Each service has its own database (database-per-service), so the data is split:

    CULTPASS_DB_URL          <- cultpass.db: users, subscriptions, experiences, reservations
    TICKETS_DB_URL           <- udahub.db: accounts, users, tickets, messages, customer memory
    KNOWLEDGE_DB_URL         <- udahub.db: accounts, knowledge articles

The tables come from the Alembic migrations (service/migrations), not from this script. A table that
already has rows is left alone, so running it again never overwrites live data. Use --reset to empty the
tables first and reload the demo data.
"""
import argparse
import logging
import os
import sys

from sqlalchemy import create_engine, delete, func, insert, inspect, select

from agentic import config
from service.migrations.common import DATABASES

logger = logging.getLogger("seed")

SOURCES = {"cultpass": config.CULTPASS_DB_PATH, "tickets": config.UDAHUB_DB_PATH, "knowledge": config.UDAHUB_DB_PATH}


def seed(name: str, target_url: str, reset: bool = False) -> dict:
    db = DATABASES[name]
    tables = [t for t in db.metadata.sorted_tables if db.includes(t.name)]
    source = create_engine(f"sqlite:///{SOURCES[name]}")
    target = create_engine(target_url)
    copied = {}
    try:
        missing = [t.name for t in tables if t.name not in set(inspect(target).get_table_names())]
        if missing:
            raise RuntimeError(f"{name}: tables {missing} don't exist. Run `python -m service.scripts.migrate` first.")
        with source.connect() as src, target.begin() as dst:
            if reset:
                for table in reversed(tables):  # children before parents
                    dst.execute(delete(table))
            for table in tables:  # parents before children
                if dst.execute(select(func.count()).select_from(table)).scalar():
                    copied[table.name] = "kept existing rows"
                    continue
                rows = [dict(r._mapping) for r in src.execute(select(table))]
                if rows:
                    dst.execute(insert(table), rows)
                copied[table.name] = len(rows)
    finally:
        source.dispose()
        target.dispose()
    return copied


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="empty the tables and reload the demo data")
    args = parser.parse_args(argv)
    seeded = {name: db for name, db in DATABASES.items() if db.seeded}
    missing = [db.url_env for db in seeded.values() if not os.getenv(db.url_env)]
    if missing:
        logger.error("Set %s", ", ".join(missing))
        return 1
    for name, db in seeded.items():
        try:
            result = seed(name, os.environ[db.url_env], reset=args.reset)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("%s: %s", db.url_env, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
