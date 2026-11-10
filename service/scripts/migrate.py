"""Bring the service databases up to the latest schema (or to a given revision).

    python -m service.scripts.migrate                  # every database whose URL is set
    python -m service.scripts.migrate tickets          # just one
    python -m service.scripts.migrate tickets 0001     # to a specific revision (upgrade only)

Databases: cultpass (CULTPASS_DB_URL), tickets (TICKETS_DB_URL), knowledge (KNOWLEDGE_DB_URL). The compose
`migrate` job runs this before `seed`. To create a revision or to downgrade, use the alembic CLI with
service/alembic.ini (see that file).
"""
import argparse
import logging
import os
import sys

from service.migrations.common import DATABASES, upgrade

logger = logging.getLogger("migrate")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database", nargs="?", choices=sorted(DATABASES), help="only this database")
    parser.add_argument("revision", nargs="?", default="head", help="target revision (default: head)")
    args = parser.parse_args(argv)

    names = [args.database] if args.database else list(DATABASES)
    missing = [DATABASES[n].url_env for n in names if not os.getenv(DATABASES[n].url_env)]
    if missing:
        logger.error("Set %s", ", ".join(missing))
        return 1
    for name in names:
        revision = upgrade(name, os.environ[DATABASES[name].url_env], args.revision)
        logger.info("%s: at revision %s", name, revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
