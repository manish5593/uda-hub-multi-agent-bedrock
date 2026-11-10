"""Schema migrations for the service databases, one Alembic history per database.

Each service owns its database (database-per-service), so each has its own migration history in
migrations/<name>/versions and its own section in service/alembic.ini:

    cultpass   CULTPASS_DB_URL   the CultPass tables
    tickets    TICKETS_DB_URL    accounts, users, tickets, messages, customer memory
    knowledge  KNOWLEDGE_DB_URL  accounts, knowledge articles

The tickets and knowledge databases are both built from the udahub models, so each history only sees its
own tables. The agent's checkpoint database is not here: LangGraph's PostgresSaver.setup() manages it.
"""
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "solution"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alembic import command, context  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import MetaData, create_engine, inspect, pool  # noqa: E402

from data.models import cultpass, udahub  # noqa: E402

ALEMBIC_INI = ROOT / "service" / "alembic.ini"
# The first revision of every history: the schema as seed.py used to create it with create_all()
BASELINE = "0001"
VERSION_TABLE = "alembic_version"

logger = logging.getLogger("migrate")


@dataclass(frozen=True)
class Database:
    name: str
    url_env: str
    metadata: MetaData
    tables: tuple

    def includes(self, table_name: str) -> bool:
        return table_name in self.tables


DATABASES = {
    "cultpass": Database("cultpass", "CULTPASS_DB_URL", cultpass.Base.metadata,
                         tuple(cultpass.Base.metadata.tables)),
    "tickets": Database("tickets", "TICKETS_DB_URL", udahub.Base.metadata,
                        ("accounts", "users", "tickets", "ticket_metadata", "ticket_messages", "customer_memory")),
    "knowledge": Database("knowledge", "KNOWLEDGE_DB_URL", udahub.Base.metadata, ("accounts", "knowledge")),
}


def alembic_config(name: str, url: str = None, connection=None) -> Config:
    cfg = Config(str(ALEMBIC_INI), ini_section=name)
    # Passed as attributes, not as sqlalchemy.url, so a password with % in it needs no escaping
    cfg.attributes["url"] = url
    cfg.attributes["connection"] = connection
    return cfg


def run_env(name: str) -> None:
    """The body of migrations/<name>/env.py."""
    import os

    db = DATABASES[name]
    cfg = context.config
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("alembic").setLevel(logging.INFO)

    def include_name(obj_name, type_, parent_names):
        # Tables in the database: ignore any that belong to another history (or to nobody)
        return db.includes(obj_name) if type_ == "table" else True

    def include_object(obj, obj_name, type_, reflected, compare_to):
        # Tables in the models: the udahub models also describe the other database's tables
        return db.includes(obj_name) if type_ == "table" else True

    options = dict(target_metadata=db.metadata, include_name=include_name, include_object=include_object,
                   version_table=VERSION_TABLE, compare_type=True)

    if context.is_offline_mode():  # alembic upgrade --sql: print the SQL instead of running it
        url = cfg.attributes.get("url") or os.getenv(db.url_env)
        if not url:
            raise RuntimeError(f"Set {db.url_env}")
        context.configure(url=url, literal_binds=True, dialect_opts={"paramstyle": "named"}, **options)
        with context.begin_transaction():
            context.run_migrations()
        return

    def run(connection):
        # SQLite can't ALTER most things; batch mode rebuilds the table instead
        context.configure(connection=connection, render_as_batch=connection.dialect.name == "sqlite", **options)
        with context.begin_transaction():
            context.run_migrations()

    connection = cfg.attributes.get("connection")
    if connection is not None:
        run(connection)
        return
    url = cfg.attributes.get("url") or os.getenv(db.url_env)
    if not url:
        raise RuntimeError(f"Set {db.url_env}")
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        run(connection)
    engine.dispose()


def upgrade(name: str, url: str, revision: str = "head") -> str:
    """Bring one database to `revision` and return the revision it ends at.

    A database whose tables were created before migrations existed (by seed.py's old create_all) has the
    tables but no version table. It is stamped at the baseline first, so the baseline doesn't try to
    create tables that are already there.
    """
    db = DATABASES[name]
    engine = create_engine(url, poolclass=pool.NullPool)
    try:
        with engine.begin() as connection:
            if connection.dialect.name == "postgresql":
                # Two migrate jobs at once (e.g. two deploys) wait for each other instead of racing
                connection.exec_driver_sql("SELECT pg_advisory_xact_lock(hashtext('udahub-migrate-' || %s))",
                                           (name,))
            existing = set(inspect(connection).get_table_names())
            ours = existing & set(db.tables)
            cfg = alembic_config(name, url, connection)
            if VERSION_TABLE not in existing and ours:
                if ours != set(db.tables):
                    raise RuntimeError(f"{name}: the database has some of its tables ({sorted(ours)}) but no "
                                       f"{VERSION_TABLE} table, so its schema version is unknown. Fix it by hand.")
                logger.info("%s: tables exist without migration history; stamping baseline %s", name, BASELINE)
                command.stamp(cfg, BASELINE)
            command.upgrade(cfg, revision)
            from alembic.migration import MigrationContext
            return MigrationContext.configure(connection, opts={"version_table": VERSION_TABLE}).get_current_revision()
    finally:
        engine.dispose()
