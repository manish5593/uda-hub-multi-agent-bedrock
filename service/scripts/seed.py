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

Customers of your own (for a demo on yourself) go in service/local/customers.json, which git ignores
(see service/local/README.md). They are added after the demo data whenever they're missing, so they
survive --reset, and with a password hash their login survives a new auth database too.
"""
import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, delete, func, insert, inspect, select

from agentic import config
from service.migrations.common import DATABASES

logger = logging.getLogger("seed")

SOURCES = {"cultpass": config.CULTPASS_DB_PATH, "tickets": config.UDAHUB_DB_PATH, "knowledge": config.UDAHUB_DB_PATH}
LOCAL_CUSTOMERS = Path(os.getenv("UDAHUB_LOCAL_CUSTOMERS", Path(__file__).resolve().parents[1] / "local" / "customers.json"))


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


def load_local_customers(path: Path = None) -> list[dict]:
    path = Path(path or LOCAL_CUSTOMERS)
    if not path.is_file():
        return []
    customers = json.loads(path.read_text())
    if not isinstance(customers, list):
        raise RuntimeError(f"{path}: expected a JSON list of customers")
    return customers


def add_local_customers(cultpass_url: str, auth_url: str = None, customers: list[dict] = None) -> list[str]:
    """Add each local customer that isn't there yet: the CultPass user, subscription and reservations, and the
    login when the entry has a password hash and an auth database is given. Returns the emails added."""
    customers = load_local_customers() if customers is None else customers
    if not customers:
        return []
    meta = DATABASES["cultpass"].metadata.tables
    users, subs, reservations, experiences = (meta[n] for n in ("users", "subscriptions", "reservations", "experiences"))
    now = datetime.now(UTC).replace(tzinfo=None)
    added = []
    engine = create_engine(cultpass_url)
    try:
        with engine.begin() as conn:
            for c in customers:
                email = c["email"].strip().lower()
                exists = conn.execute(select(func.count()).select_from(users).where(
                    (users.c.user_id == c["user_id"]) | (func.lower(users.c.email) == email))).scalar()
                if exists:
                    continue
                conn.execute(insert(users).values(user_id=c["user_id"], full_name=c["full_name"], email=email,
                                                  is_blocked=c.get("is_blocked", False), created_at=now, updated_at=now))
                if sub := c.get("subscription"):
                    conn.execute(insert(subs).values(
                        subscription_id=sub["subscription_id"], user_id=c["user_id"], status=sub.get("status", "active"),
                        tier=sub.get("tier", "basic"), monthly_quota=sub.get("monthly_quota", 4),
                        started_at=now - timedelta(days=1), created_at=now, updated_at=now))
                for r in c.get("reservations", []):
                    conn.execute(insert(reservations).values(
                        reservation_id=r["reservation_id"], user_id=c["user_id"], experience_id=r["experience_id"],
                        status=r.get("status", "reserved"), created_at=now, updated_at=now))
                    if r.get("status", "reserved") == "reserved":
                        conn.execute(experiences.update().where(experiences.c.experience_id == r["experience_id"])
                                     .values(slots_available=experiences.c.slots_available - 1))
                added.append(email)
    finally:
        engine.dispose()

    if auth_url:
        from service.auth_service.models import Credential
        credentials = Credential.__table__
        engine = create_engine(auth_url)
        try:
            with engine.begin() as conn:
                for c in customers:
                    email = c["email"].strip().lower()
                    if not c.get("password_hash") or conn.execute(
                            select(func.count()).select_from(credentials).where(credentials.c.email == email)).scalar():
                        continue
                    conn.execute(insert(credentials).values(
                        email=email, external_user_id=c["user_id"], password_hash=c["password_hash"],
                        failed_logins=0, created_at=now))
        finally:
            engine.dispose()
    return added


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
    try:
        added = add_local_customers(os.environ["CULTPASS_DB_URL"], os.getenv("AUTH_DB_URL"))
    except (RuntimeError, ValueError, KeyError) as exc:
        logger.error("Local customers (%s): %s", LOCAL_CUSTOMERS, exc)
        return 1
    if added:
        logger.info("Local customers added: %d", len(added))
    return 0


if __name__ == "__main__":
    sys.exit(main())
