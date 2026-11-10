#!/bin/bash
# Runs once, when the Postgres volume is first created: one database and one login role per service.
# Each role can connect only to its own database, so no service can read another service's data.
set -euo pipefail

for db in cultpass tickets knowledge agent_state; do
    password_var="$(echo "${db}" | tr '[:lower:]' '[:upper:]')_DB_PASSWORD"
    password="${!password_var:?set ${password_var}}"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
         -v db="$db" -v role="${db}_svc" -v pw="$password" <<'SQL'
CREATE ROLE :"role" LOGIN PASSWORD :'pw';
CREATE DATABASE :"db" OWNER :"role";
REVOKE ALL ON DATABASE :"db" FROM PUBLIC;
SQL
done
