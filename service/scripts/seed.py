"""Create each service's tables and copy in the committed demo data from the SQLite files.

    python -m service.scripts.seed

Each service has its own database (database-per-service), so the data is split:

    CULTPASS_DB_URL          <- cultpass.db: users, subscriptions, experiences, reservations
    TICKETS_DB_URL           <- udahub.db: accounts, users, tickets, messages, customer memory
    KNOWLEDGE_DB_URL         <- udahub.db: accounts, knowledge articles

A table that already has rows is left alone, so running it again never overwrites live data.
Use --reset to empty the tables first and reload the demo data.
"""
import argparse
import logging
import os
import sys

from sqlalchemy import create_engine, delete, func, insert, select

from agentic import config
from data.models import cultpass, udahub

logger = logging.getLogger("seed")

TARGETS = {
    "CULTPASS_DB_URL": (config.CULTPASS_DB_PATH, cultpass.Base.metadata, None),
    "TICKETS_DB_URL": (config.UDAHUB_DB_PATH, udahub.Base.metadata,
                       ["accounts", "users", "tickets", "ticket_metadata", "ticket_messages", "customer_memory"]),
    "KNOWLEDGE_DB_URL": (config.UDAHUB_DB_PATH, udahub.Base.metadata, ["accounts", "knowledge"]),
}


def seed(source_path, metadata, table_names, target_url: str, reset: bool = False) -> dict:
    tables = [t for t in metadata.sorted_tables if table_names is None or t.name in table_names]
    source = create_engine(f"sqlite:///{source_path}")
    target = create_engine(target_url)
    metadata.create_all(target, tables=tables)
    copied = {}
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
    source.dispose()
    target.dispose()
    return copied


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="empty the tables and reload the demo data")
    args = parser.parse_args(argv)
    missing = [name for name in TARGETS if not os.getenv(name)]
    if missing:
        logger.error("Set %s", ", ".join(missing))
        return 1
    for env, (path, metadata, names) in TARGETS.items():
        result = seed(path, metadata, names, os.environ[env], reset=args.reset)
        logger.info("%s: %s", env, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
