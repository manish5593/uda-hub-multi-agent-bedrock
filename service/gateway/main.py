"""API gateway: the only UDA-Hub service reachable from outside.

* Authenticates clients with API keys (header X-API-Key; GATEWAY_API_KEYS is a comma-separated list).
* Rate-limits each key (GATEWAY_RATE_LIMIT_PER_MINUTE).
* Forwards requests to the internal services with the internal token and a request id.
* Returns only what a channel integration needs: the reply and the ticket status, never the agent's
  internal trace, routing reasons or the system notes on the ticket.

Callers are trusted channel integrations (web chat, email bridge), not end customers: metadata such as
the customer's email is taken as given, as in the original design.
"""
import hmac
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import Depends, Header, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from service.common.app import REQUEST_ID_HEADER, TOKEN_HEADER, create_app, internal_token, request_id_var

TICKET_ID = Path(pattern=r"^[A-Za-z0-9._:-]{1,64}$")
UPSTREAMS = {
    "agent": os.getenv("AGENT_SERVICE_URL", "http://agent:8000"),
    "tickets": os.getenv("TICKET_SERVICE_URL", "http://tickets:8000"),
}
# A turn makes several LLM calls; give it time before giving up
AGENT_TIMEOUT = float(os.getenv("GATEWAY_AGENT_TIMEOUT", 120))
CUSTOMER_VISIBLE_ROLES = {"user", "ai", "agent"}


def api_keys() -> list[str]:
    return [k.strip() for k in os.getenv("GATEWAY_API_KEYS", "").split(",") if k.strip()]


class RateLimiter:
    """Sliding window per API key. Kept in memory, so the limit applies per gateway replica."""

    def __init__(self):
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        with self.lock:
            hits = self.hits[key]
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True


limiter = RateLimiter()
clients: dict[str, httpx.AsyncClient] = {}


def make_client(name: str) -> httpx.AsyncClient:
    """Overridden in tests to route to in-process apps."""
    return httpx.AsyncClient(base_url=UPSTREAMS[name], timeout=httpx.Timeout(10, read=AGENT_TIMEOUT),
                             transport=httpx.AsyncHTTPTransport(retries=2))


@asynccontextmanager
async def lifespan(app):
    if not api_keys():
        raise RuntimeError("GATEWAY_API_KEYS is empty: no client could call the API")
    for name in UPSTREAMS:
        clients[name] = make_client(name)
    yield
    for client in clients.values():
        await client.aclose()
    clients.clear()


app = create_app("gateway", "UDA-Hub API", lifespan=lifespan)
# The gateway checks API keys, not the internal token
app.state.require_token = False


def authenticate(x_api_key: str = Header(default="")) -> str:
    key = next((k for k in api_keys() if hmac.compare_digest(x_api_key, k)), None)
    if key is None:
        raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})
    limit = int(os.getenv("GATEWAY_RATE_LIMIT_PER_MINUTE", 30))
    if not limiter.allow(key, limit):
        raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})
    return key


async def forward(name: str, method: str, path: str, json=None) -> httpx.Response:
    headers = {REQUEST_ID_HEADER: request_id_var.get() or uuid.uuid4().hex}
    if internal_token():
        headers[TOKEN_HEADER] = internal_token()
    try:
        return await clients[name].request(method, path, json=json, headers=headers)
    except httpx.TimeoutException:
        raise HTTPException(504, f"The {name} service did not answer in time")
    except httpx.TransportError:
        raise HTTPException(502, f"The {name} service is unavailable")


def relay_error(response: httpx.Response) -> None:
    """Pass client errors through, hide the details of server errors."""
    if response.status_code == 404:
        raise HTTPException(404, "Ticket not found")
    if response.status_code == 422:
        raise HTTPException(422, response.json().get("detail", "Invalid request"))
    if response.status_code >= 400:
        raise HTTPException(502, "Upstream service error")


class MessageIn(BaseModel):
    text: str = Field(max_length=4000)
    metadata: Optional[dict] = Field(None, description="e.g. {\"email\": ..., \"channel\": \"web\", \"urgency\": \"high\"}")


class ReplyOut(BaseModel):
    ticket_id: str
    reply: str
    status: str
    escalated_to: Optional[str] = None


@app.post("/v1/tickets/{ticket_id}/messages", response_model=ReplyOut)
async def send_message(body: MessageIn, ticket_id: str = TICKET_ID, _key: str = Depends(authenticate)):
    """Send one customer message and get the reply. Use the same ticket id for the whole conversation."""
    response = await forward("agent", "POST", f"/tickets/{ticket_id}/messages", json=body.model_dump())
    relay_error(response)
    turn = response.json()
    return ReplyOut(ticket_id=turn["ticket_id"], reply=turn["reply"], status=turn["status"],
                    escalated_to=(turn.get("escalation") or {}).get("team"))


class TicketOut(BaseModel):
    ticket_id: str
    status: str
    channel: Optional[str] = None
    issue_type: Optional[str] = None


@app.get("/v1/tickets/{ticket_id}", response_model=TicketOut)
async def get_ticket(ticket_id: str = TICKET_ID, _key: str = Depends(authenticate)):
    response = await forward("tickets", "GET", f"/tickets/{ticket_id}")
    relay_error(response)
    t = response.json()
    return TicketOut(ticket_id=t["ticket_id"], status=t["status"], channel=t["channel"], issue_type=t["main_issue_type"])


@app.get("/v1/tickets/{ticket_id}/messages")
async def get_messages(ticket_id: str = TICKET_ID, _key: str = Depends(authenticate)) -> list[dict]:
    """The conversation as the customer saw it. Internal system notes (escalation handoffs, refund
    requests) stay inside."""
    response = await forward("tickets", "GET", f"/tickets/{ticket_id}/messages")
    relay_error(response)
    return [m for m in response.json() if m["role"] in CUSTOMER_VISIBLE_ROLES]


@app.get("/ready", include_in_schema=False)
async def ready(request: Request):
    """Readiness of the whole system: every upstream's /health."""
    status = {}
    for name in UPSTREAMS:
        try:
            status[name] = (await clients[name].get("/health")).status_code == 200
        except httpx.HTTPError:
            status[name] = False
    return JSONResponse({"ready": all(status.values()), "services": status},
                        status_code=200 if all(status.values()) else 503)
