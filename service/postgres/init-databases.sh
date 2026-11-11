#!/bin/bash
# One database and one login role per service. Each role can connect only to its own database, so no
# service can read another service's data.
#
# Idempotent: it creates only what is missing. It runs when the Postgres volume is first created
# (docker-entrypoint-initdb.d), and on every `docker compose up` (the db-setup job), so a database added
# later (e.g. auth) also appears in an existing volume.
set -euo pipefail

for db in cultpass tickets knowledge agent_state auth; do
    password_var="$(echo "${db}" | tr '[:lower:]' '[:upper:]')_DB_PASSWORD"
    password="${!password_var:?set ${password_var}}"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
         -v db="$db" -v role="${db}_svc" -v pw="$password" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'role', :'pw')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'role') \gexec
SELECT format('CREATE DATABASE %I OWNER %I', :'db', :'role')
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'db') \gexec
REVOKE ALL ON DATABASE :"db" FROM PUBLIC;
SQL
done
