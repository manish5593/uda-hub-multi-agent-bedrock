"""Agent service: runs the UDA-Hub LangGraph workflow, one customer message per request.

It owns no business data. Every read or write of CultPass accounts, tickets, memory or the knowledge base
goes to the other services over HTTP (CULTPASS_SERVICE_URL, TICKET_SERVICE_URL, KNOWLEDGE_SERVICE_URL), and
it refuses to start without them. Its only database is the LangGraph checkpointer (AGENT_STATE_DB_URL): the
conversation of each ticket, so any replica can continue any ticket after a restart.

Two messages on the same ticket must not run at the same time (both would read the same checkpoint and
the second would overwrite the first), so each turn holds a Postgres advisory lock on the ticket id, which
works across replicas. Without Postgres (local runs and tests) an in-process lock does the same job.
"""
import logging
import os
import threading
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, get_args

from fastapi import Path
from prometheus_client import Counter
from pydantic import BaseModel, Field

from agentic import config
from agentic.agents.escalation_agent import EscalationHandoff
from agentic.remote.http import is_remote
from agentic.state import Route
from agentic.workflow import build_graph, process_ticket
from service.common.app import create_app

logger = logging.getLogger("agent")
STATE_DB_URL = os.getenv("AGENT_STATE_DB_URL") or None
TICKET_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"

# What the agent did with each message (the HTTP metrics in service.common only see requests and timings).
# Every label comes from a fixed set (Route, ticket statuses, EscalationHandoff, the tool names).
TURNS = Counter("udahub_agent_turns_total", "Customer messages handled", ["route", "status"])
ESCALATIONS = Counter("udahub_agent_escalations_total", "Tickets handed to a human team", ["team", "priority"])
TOOL_CALLS = Counter("udahub_agent_tool_calls_total", "Tool calls made while handling messages", ["tool"])
TURN_STATUSES = ("resolved", "awaiting_customer", "escalated", "error")


def _start_counters_at_zero() -> None:
    """Create every known series at 0. Prometheus's increase() can't see a series' first increment if the
    series doesn't exist yet, so without this the first escalation to each team would never be counted."""
    for route in (*get_args(Route), "none"):
        for status in TURN_STATUSES:
            TURNS.labels(route, status)
    for team in get_args(EscalationHandoff.model_fields["team"].annotation):
        for priority in get_args(EscalationHandoff.model_fields["priority"].annotation):
            ESCALATIONS.labels(team, priority)


_start_counters_at_zero()


class Runtime:
    graph = None
    pool = None  # psycopg ConnectionPool for the checkpointer


runtime = Runtime()


def check_services() -> None:
    missing = [name for name in ("cultpass", "tickets", "knowledge") if not is_remote(name)]
    if missing:
        raise RuntimeError(f"Agent service needs the URLs of these services: {missing}. Without them it would "
                           "fall back to the SQLite files bundled with the code.")


def build_runtime() -> None:
    if STATE_DB_URL:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        runtime.pool = ConnectionPool(STATE_DB_URL, min_size=1, max_size=int(os.getenv("AGENT_DB_POOL_SIZE", 10)),
                                      kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                                      open=True)
        checkpointer = PostgresSaver(runtime.pool)
        checkpointer.setup()  # creates or migrates the checkpoint tables
        logger.info("Conversation state in Postgres")
    else:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        logger.warning("AGENT_STATE_DB_URL not set: conversations are kept in memory and lost on restart")
    runtime.graph = build_graph(checkpointer)


@asynccontextmanager
async def lifespan(app):
    check_services()
    build_runtime()
    yield
    if runtime.pool is not None:
        runtime.pool.close()


def health() -> None:
    if runtime.graph is None:
        raise RuntimeError("graph not ready")
    if runtime.pool is not None:
        with runtime.pool.connection() as conn:
            conn.execute("SELECT 1")


app = create_app("agent", "UDA-Hub agent service", health_check=health, lifespan=lifespan)

_local_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_local_locks_guard = threading.Lock()


@contextmanager
def ticket_lock(ticket_id: str):
    """Serialise turns on one ticket. Postgres advisory lock across replicas, or a local lock."""
    if STATE_DB_URL:
        import psycopg
        # A dedicated connection: the lock lives as long as the session, and is released even if this
        # process dies mid-turn. Not taken from the checkpointer's pool, so turns can't starve it.
        with psycopg.connect(STATE_DB_URL, autocommit=True) as conn:
            conn.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (ticket_id,))
            try:
                yield
            finally:
                conn.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (ticket_id,))
    else:
        with _local_locks_guard:
            lock = _local_locks[ticket_id]
        with lock:
            yield


class MessageIn(BaseModel):
    text: str = Field(max_length=4000)
    # Ticket metadata from the channel, e.g. {"email": ..., "channel": "web", "urgency": "high"}
    metadata: Optional[dict] = None


class EscalationOut(BaseModel):
    team: str
    priority: str
    reason: str


class TurnOut(BaseModel):
    ticket_id: str
    reply: str
    status: str  # resolved | awaiting_customer | escalated | error
    route: Optional[str] = None
    route_reason: Optional[str] = None
    escalation: Optional[EscalationOut] = None
    tools_used: list[str] = []
    trace: list[str] = []


@app.post("/tickets/{ticket_id}/messages", response_model=TurnOut)
def post_message(body: MessageIn, ticket_id: str = Path(pattern=TICKET_ID_PATTERN)):
    with ticket_lock(ticket_id):
        # process_ticket never raises: a failure returns a fallback reply and flags the ticket for a human
        state = process_ticket(body.text, ticket_id, body.metadata, agent=runtime.graph)
    escalation = state.get("escalation")
    status = state.get("ticket_status") or "error"
    tools_used = [a["tool"] for a in state.get("actions_taken") or []]
    TURNS.labels(state.get("route") or "none", status).inc()
    if escalation:
        ESCALATIONS.labels(escalation["team"], escalation["priority"]).inc()
    for tool in tools_used:
        TOOL_CALLS.labels(tool).inc()
    return TurnOut(
        ticket_id=ticket_id,
        reply=state.get("final_response", ""),
        status=status,
        route=state.get("route"),
        route_reason=state.get("route_reason"),
        escalation=EscalationOut(**{k: escalation[k] for k in ("team", "priority", "reason")}) if escalation else None,
        tools_used=tools_used,
        trace=state.get("trace") or [],
    )
