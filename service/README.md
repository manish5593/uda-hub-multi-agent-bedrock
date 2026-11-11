C# UDA-Hub as microservices

This folder runs UDA-Hub as four services behind an API gateway, with Postgres. The approach follows
[fastapi-langgraph-agent-production-ready-template](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template),
but splits the system along the lines that already exist in the code: the CultPass system, ticket storage,
the knowledge base, and the agents.

The agents, prompts, routing rules and tools are the ones in `solution/agentic`, unchanged. Each data function
in `agentic/tools` runs either locally against a database, as in the notebooks, or over HTTP against the
service that owns that data, depending on configuration (see [How the split works](#how-the-split-works)).

**Status:** the services, the Postgres paths, the migrations and the HTTP flow between the services are covered
by `service/tests`, which CI also runs against Postgres 16. The images build and the whole stack, monitoring
included, has been run with `docker compose`: migrations, seed, health checks, the gateway, Prometheus and
Grafana. That run used a placeholder model key, so messages got the fallback reply; a run with real model
credentials is still to do.

## Architecture

![UDA-Hub deployment as microservices](../solution/agentic/design/deployment.svg)

The gateway is the only public entry point. The **agent service** runs the whole LangGraph workflow: its nodes are
shown in the order they run, with the same names as in `solution/agentic/workflow.py` (the full workflow, with
each node's role and routing conditions, is [`architecture.svg`](../solution/agentic/design/architecture.svg)).
The tickets, cultpass and knowledge services only hold data, each in its own Postgres database.

Each node is coloured by its agent type, the module in `agentic/agents` it comes from: gray `memory_agent`
(`load_context`, `finalize`, `write_memory`), teal `classifier`, purple `supervisor` (`supervisor`,
`request_info`), blue `knowledge_agent`, brown `account_agent`, orange `quality_gate` (`quality_gate`, `approve`)
and red `escalation_agent`. A red **LLM** badge marks every node that calls the model; it is not a colour. Green is
the customer, yellow is tools and data, and dashed arrows are conditional routes. This diagram adds magenta for the
containers (gateway, services, monitoring) and dark slate for the external model APIs.

| Service | Owns | Does | Talks to |
|---|---|---|---|
| `gateway` | nothing | The only public endpoint (port 8080). Checks API keys, rate-limits each key, adds request ids, and returns only the reply and ticket status, never the agent's trace or internal notes | agent, tickets |
| `agent` | `agent_state` DB (LangGraph checkpoints) | Runs the LangGraph workflow for one customer message, with the agents from `solution/agentic/agents` (see [The agents in the agent service](#the-agents-in-the-agent-service)) | cultpass, tickets, knowledge, the chat model |
| `cultpass` | `cultpass` DB | The CultPass system of record: customers, subscriptions, experiences, reservations, and their rules (quota, premium tier, blocked accounts, free slots) | nothing |
| `tickets` | `tickets` DB | Tickets, messages, UDA-Hub users and long-term customer memory | nothing |
| `knowledge` | `knowledge` DB | Knowledge-base articles and their embedding index; RAG search with the keyword fallback | the embedding model |

Supporting containers:

| Container | Runs | Does |
|---|---|---|
| `db` | Always | Postgres 16: one database and one login role per service, created on first start |
| `migrate` | Once per `up`, before the services | Brings the cultpass, tickets and knowledge databases to the latest schema ([Schema migrations](#schema-migrations)) |
| `seed` | Once per `up`, after `migrate` | Loads the demo data into empty tables |
| `prometheus` | `--profile monitoring` | Scrapes `/metrics` on every service every 15 seconds |
| `grafana` | `--profile monitoring` | The UDA-Hub overview dashboard ([Monitoring](#monitoring)) |

Each service has its **own database and its own Postgres role**, which can't connect to any other database.
They share one Postgres server to keep the demo cheap; moving a database to its own server only changes a URL.

Only the agent and knowledge services get model credentials (`llm.env`), and only the agent can write to
CultPass. Internal services accept a request only with the shared `INTERNAL_API_TOKEN`.

### The agents in the agent service

The agent service runs the same graph as the notebooks (`solution/agentic/workflow.py`), with the same node
names. Only the data calls change: they go to the service that owns the data.

| Node | Module | Uses the LLM | Calls |
|---|---|---|---|
| `load_context` | `memory_agent.py` | No | tickets service (ticket, messages, memories), cultpass service (customer profile) |
| `classifier` | `classifier.py` | Yes | nothing |
| `supervisor` | `supervisor.py` | No | cultpass service (find a guest by email), tickets service (link the ticket) |
| `request_info` | `supervisor.py` | Only to translate | nothing |
| `knowledge_agent` | `knowledge_agent.py` | Yes | knowledge service (RAG search) |
| `account_agent` | `account_agent.py` | Yes (`create_agent` with tools) | cultpass service (account, bookings, subscription), tickets service (refund request note), knowledge service (search) |
| `quality_gate`, `approve` | `quality_gate.py` | `quality_gate` only | nothing |
| `escalation_agent` | `escalation_agent.py` | Yes | tickets service (handoff note) |
| `finalize` | `memory_agent.py` | No | tickets service (reply, ticket status) |
| `write_memory` | `memory_agent.py` | Yes | tickets service (customer memory) |

## How the split works

The data functions in `solution/agentic/tools` (for example `cultpass_tools.reserve_experience` or
`ticket_tools.add_message`) carry a `@remote("cultpass")` or `@remote("tickets")` decorator:

- **No service URL set** (notebooks, `solution/tests`): the function runs locally against SQLite, as before.
- **`CULTPASS_SERVICE_URL` set** (the agent container): the call goes to the function of the same name in
  `agentic/remote/cultpass.py`, which sends it to the CultPass service over HTTP. Same arguments, same result.
- **In a service that owns the data**: the service runs the same function locally against its own Postgres
  database, so each business rule is still written once.

`get_retriever()` works the same way for the knowledge base. Because the agents and tools don't change,
the safeguards hold in both modes: tools are still bound to one customer, booking still requires an id the
agent looked up, and policy decisions are still made in code.

What the services add:

| Concern | How |
|---|---|
| Conversation state | `PostgresSaver` checkpointer in `agent_state`: any agent replica can continue any ticket, and conversations survive restarts |
| Two messages on one ticket at once | Each turn holds a Postgres advisory lock on the ticket id, across replicas |
| Two customers booking the last slot | The experience row is locked (`SELECT ... FOR UPDATE`) while a booking or cancellation runs |
| A service is down | Tool calls return a structured error the agent explains; other failures hit the workflow's existing fallback reply and flag the ticket for a human. Failed connections are retried twice |
| Tracing one message | The gateway's `X-Request-ID` is forwarded on every internal call and appears in every service's JSON logs |
| Metrics | `/metrics` on every service (request count and latency by route), plus what the agent did with each message (outcome, route, escalation team and priority, tools called); scraped by the optional Prometheus and shown in Grafana |
| Schema changes | Alembic migrations, one history per database, applied by the `migrate` job before the services start |
| Health | `/health` on every service checks its database; the gateway's `/ready` checks the services behind it |

## Running it

Requires Docker with Compose v2. From `service/`:

```bash
cp .env.example .env          # API keys, internal token, database passwords
cp llm.env.example llm.env    # OpenAI key, or Bedrock / SageMaker settings
docker compose up --build
```

On first start, Postgres creates the four databases and roles (`postgres/init-databases.sh`). Then two
one-shot jobs run, on every `up`:

1. `migrate` brings the cultpass, tickets and knowledge databases to the latest schema
   (`scripts/migrate.py`, see [Schema migrations](#schema-migrations)).
2. `seed` copies the committed demo data from the SQLite files into them (`scripts/seed.py`). It skips any
   table that already has rows, so restarts never overwrite data.

The services start only after both succeed. To reload the demo data:

```bash
docker compose run --rm seed python -m service.scripts.seed --reset
```

Send a message (use the same ticket id for the whole conversation):

```bash
curl -s localhost:8080/v1/tickets/T-100/messages \
  -H 'X-API-Key: change-me-web-chat-key' -H 'Content-Type: application/json' \
  -d '{"text": "How do I reset my password?"}'
# {"ticket_id":"T-100","reply":"...","status":"resolved","escalated_to":null}

curl -s localhost:8080/v1/tickets/T-100/messages -H 'X-API-Key: change-me-web-chat-key' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Cancel my reservation", "metadata": {"email": "cathy.bloom@florals.org"}}'

curl -s localhost:8080/v1/tickets/T-100 -H 'X-API-Key: change-me-web-chat-key'
curl -s localhost:8080/v1/tickets/T-100/messages -H 'X-API-Key: change-me-web-chat-key'
```

`status` is `resolved`, `awaiting_customer`, `escalated`, or `error` (the fallback reply was sent and the
ticket was flagged for a human). The interactive API docs are at <http://localhost:8080/docs>.

### Monitoring

```bash
docker compose --profile monitoring up --build
```

This also starts Prometheus on <http://localhost:9090> and Grafana on <http://localhost:3000> (user `admin`,
password `GRAFANA_ADMIN_PASSWORD` from `.env`). Both listen on localhost only. Grafana opens on the
**UDA-Hub overview** dashboard:

- **Overview** for the selected time range: services up, customer messages, the share resolved and the share
  escalated, p95 reply time, and the share of server errors.
- **Support outcomes**: messages by outcome and by route over time, escalations by team and by priority,
  and tool calls.
- **Traffic and latency**: requests and 5xx errors by service, reply time (p50, p95, p99), p95 latency by
  service, requests the gateway rejected (401 bad key, 429 rate limited), the slowest routes, and service
  health.

The datasource and the dashboard are provisioned from `monitoring/grafana`, so they are recreated with the
container. To change the dashboard, edit `monitoring/grafana/dashboards/udahub.json` (or edit a copy in the UI
and export its JSON into that file). Grafana picks up changes to the file within about 10 seconds.

Counts over a time range use Prometheus's `increase()`, which extrapolates to the edges of the range, so a
small count can read slightly high (for example 5 for 4 messages).

### Schema migrations

Each service database has its own [Alembic](https://alembic.sqlalchemy.org) history in
`migrations/<database>/versions`, configured in `alembic.ini` (one section per database). The models stay in
`solution/data/models`; the tickets and knowledge histories each cover only their own tables.

| Database | URL variable | Tables |
|---|---|---|
| `cultpass` | `CULTPASS_DB_URL` | users, subscriptions, experiences, reservations |
| `tickets` | `TICKETS_DB_URL` | accounts, users, tickets, ticket_metadata, ticket_messages, customer_memory |
| `knowledge` | `KNOWLEDGE_DB_URL` | accounts, knowledge |

To change the schema, edit the model, then generate and review a revision. From the repository root, with the
database's URL set (for example a local Postgres, or a scratch SQLite file):

```bash
export TICKETS_DB_URL=postgresql+psycopg://...    # a database at the current head
alembic -c service/alembic.ini -n tickets revision --autogenerate --rev-id 0002 -m "add ticket priority"
# review service/migrations/tickets/versions/0002_add_ticket_priority.py, then:
python -m service.scripts.migrate tickets         # or: docker compose up, which runs the migrate job
```

Other commands: `alembic -c service/alembic.ini -n <database> current | history | downgrade -1`, and
`upgrade head --sql` to print the SQL without running it.

- `service/tests/test_migrations.py` fails if a model changes without a matching revision, if a history has two
  heads, or if a downgrade doesn't undo its upgrade.
- Databases created before migrations existed (the old `seed` made the tables itself) are recognised by having
  every table but no `alembic_version`. The `migrate` job stamps them at the first revision and keeps their
  data. A database with only some of the tables is refused.
- The `agent_state` database isn't in Alembic: LangGraph's `PostgresSaver.setup()` creates and upgrades the
  checkpoint tables when the agent starts.

### Public API (gateway)

| Method and path | Body | Returns |
|---|---|---|
| `POST /v1/tickets/{ticket_id}/messages` | `{"text": "...", "metadata": {...}}`; metadata is optional: `email` or `external_user_id`, `channel`, `urgency`, `account_id` | `{ticket_id, reply, status, escalated_to}` |
| `GET /v1/tickets/{ticket_id}` | | `{ticket_id, status, channel, issue_type}` |
| `GET /v1/tickets/{ticket_id}/messages` | | The conversation as the customer saw it; system notes (escalation handoffs, refund requests) are left out |

API callers are trusted channel integrations, not end customers: as in the original design, metadata such as
the customer's email is taken as given.

## Tests

From the repository root, with the dependencies of `requirements.txt` and `service/requirements.txt` installed:

```bash
python -m pytest service/tests
```

The tests run every service in the same process against copies of the SQLite databases, with the agent's
calls going through real HTTP requests (FastAPI `TestClient`) and only the LLM scripted. They cover the
service APIs, internal-token and API-key checks, rate limiting, the full gateway → agent → services flow,
multi-turn conversations, escalation, memory, the agent metrics, and a service outage mid-turn.

The migration tests run on SQLite. To also run them on Postgres, as CI does, give them a role that can create
databases (each test uses its own new database and drops it afterwards):

```bash
UDAHUB_TEST_POSTGRES_URL=postgresql+psycopg://postgres:<password>@localhost:5432/postgres python -m pytest service/tests
```

## Configuration

| Variable | Used by | Meaning |
|---|---|---|
| `GATEWAY_API_KEYS` | gateway | Comma-separated client keys |
| `GATEWAY_RATE_LIMIT_PER_MINUTE` | gateway | Requests per key per minute (default 30) |
| `INTERNAL_API_TOKEN` | all | Shared secret for service-to-service calls; services refuse requests without it |
| `CULTPASS_DB_URL`, `UDAHUB_DB_URL` | cultpass; tickets and knowledge | SQLAlchemy URL of the service's own database |
| `CULTPASS_DB_URL`, `TICKETS_DB_URL`, `KNOWLEDGE_DB_URL` | migrate and seed jobs | The three service databases |
| `CULTPASS_SERVICE_URL`, `TICKET_SERVICE_URL`, `KNOWLEDGE_SERVICE_URL` | agent | Where the other services are; the agent refuses to start without all three |
| `AGENT_STATE_DB_URL` | agent | Postgres URL of the checkpointer; without it, conversations live in memory |
| `UDAHUB_SERVICE_TIMEOUT` | agent | Seconds per internal call (default 15) |
| Model settings (`llm.env`) | agent, knowledge | As in `solution/.env.example` |
| `GRAFANA_ADMIN_PASSWORD` | grafana | Password of Grafana's `admin` user (default `admin`) |

## Limitations

- The demo experiences in the committed `cultpass.db` are dated 1 to 7 days after the day it was built
  (10 to 16 November 2026). Once those dates pass, searches return nothing and bookings fail, in the services
  as in the notebooks (the tests are unaffected: they move the dates to today). Rebuild the database with
  `01_external_db_setup.ipynb` before seeding.
- The rate limit is kept in each gateway replica's memory. With several gateway replicas, move it to Redis
  (or Valkey, as the template does).
- A message is answered in one HTTP request that lasts as long as the LLM calls (often 5 to 20 seconds).
  Streaming (SSE) or a queue with callbacks would suit chat front ends better.
- The CultPass service shares the rule functions in `agentic/tools` with the notebooks instead of having its
  own copy. That keeps one source of truth for this project; a real CultPass would be a separate system.
- The gateway's `/metrics` is reachable through the published port. In production, publish the gateway behind
  a proxy that serves only `/v1`.
- Service-to-service calls use one shared token over the private Docker network, not mTLS or per-service
  identities.
