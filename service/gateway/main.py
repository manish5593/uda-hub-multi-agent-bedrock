"""API gateway: the only UDA-Hub service reachable from outside.

Two kinds of caller:

* Channel integrations (web chat, email bridge) send an API key in X-API-Key (GATEWAY_API_KEYS is a
  comma-separated list). They are trusted: metadata such as the customer's email is taken as given, as in
  the original design.
* Customers who logged in (POST /v1/auth/login, then the emailed code to /v1/auth/login/verify) send their access token in `Authorization: Bearer`. The
  token is verified here (service/common/tokens.py). Their identity comes from the token, never from the
  message metadata, and they can only see and continue their own tickets.

It also:

* Rate-limits each API key and each customer (GATEWAY_RATE_LIMIT_PER_MINUTE), and the sign-up and login
  endpoints per client IP (GATEWAY_AUTH_RATE_LIMIT_PER_MINUTE).
* Forwards requests to the internal services with the internal token and a request id.
* Returns only what a client needs: the reply and the ticket status, never the agent's internal trace,
  routing reasons or the system notes on the ticket.
"""
import hmac
import os
import pathlib
import threading
import time
import uuid
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal, Optional

import httpx
import jwt
from fastapi import Depends, Header, HTTPException, Path, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from service.common.app import REQUEST_ID_HEADER, TOKEN_HEADER, create_app, internal_token, request_id_var
from service.common.tokens import jwt_configured, verify_access_token

logger = logging.getLogger("gateway")

TICKET_ID = Path(pattern=r"^[A-Za-z0-9._:-]{1,64}$")
UPSTREAMS = {
    "agent": os.getenv("AGENT_SERVICE_URL", "http://agent:8000"),
    "tickets": os.getenv("TICKET_SERVICE_URL", "http://tickets:8000"),
    "auth": os.getenv("AUTH_SERVICE_URL", "http://auth:8000"),
}
# A turn makes several LLM calls; give it time before giving up
AGENT_TIMEOUT = float(os.getenv("GATEWAY_AGENT_TIMEOUT", 120))
CUSTOMER_VISIBLE_ROLES = {"user", "ai", "agent"}
# Who the customer is: set from the token for logged-in customers, never taken from their metadata
IDENTITY_FIELDS = {"email", "external_user_id", "user_name", "account_id"}


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
    if not jwt_configured():
        logger.warning("JWT_SECRET_KEY is not set (or too short): customer logins are disabled, API keys still work")
    for name in UPSTREAMS:
        clients[name] = make_client(name)
    yield
    for client in clients.values():
        await client.aclose()
    clients.clear()


app = create_app("gateway", "UDA-Hub API", lifespan=lifespan)
# The gateway checks API keys, not the internal token
app.state.require_token = False


@dataclass(frozen=True)
class Caller:
    kind: Literal["channel", "customer"]
    # The API key for a channel, the CultPass user id for a customer
    id: str
    name: Optional[str] = None


bearer = HTTPBearer(auto_error=False)


def authenticate(x_api_key: str = Header(default=""),
                 credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> Caller:
    """A customer's Bearer token or a channel's API key. A Bearer token wins when both are sent."""
    if credentials is not None:
        try:
            claims = verify_access_token(credentials.credentials)
        except (jwt.InvalidTokenError, RuntimeError):
            raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
        caller = Caller("customer", claims["sub"], claims.get("name"))
    else:
        key = next((k for k in api_keys() if hmac.compare_digest(x_api_key, k)), None)
        if key is None:
            raise HTTPException(401, "Missing or invalid API key or token",
                                headers={"WWW-Authenticate": "Bearer, ApiKey"})
        caller = Caller("channel", key)
    limit = int(os.getenv("GATEWAY_RATE_LIMIT_PER_MINUTE", 30))
    if not limiter.allow(f"{caller.kind}:{caller.id}", limit):
        raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})
    return caller


def limit_client_ip(request: Request) -> None:
    """Sign-up and login have no caller identity yet: limit them per client IP, more tightly."""
    limit = int(os.getenv("GATEWAY_AUTH_RATE_LIMIT_PER_MINUTE", 10))
    if not limiter.allow(f"ip:{request.client.host if request.client else 'unknown'}", limit):
        raise HTTPException(429, "Too many attempts", headers={"Retry-After": "60"})


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


async def load_ticket(caller: Caller, ticket_id: str, missing_ok: bool = False) -> Optional[dict]:
    """The ticket, if this caller may see it. Another customer's ticket is a 404, like a missing one, so
    ticket ids can't be probed."""
    response = await forward("tickets", "GET", f"/tickets/{ticket_id}")
    if response.status_code == 404 and missing_ok:
        return None
    relay_error(response)
    ticket = response.json()
    if caller.kind == "customer" and ticket["external_user_id"] != caller.id:
        raise HTTPException(404, "Ticket not found")
    return ticket


@app.post("/v1/tickets/{ticket_id}/messages", response_model=ReplyOut)
async def send_message(body: MessageIn, ticket_id: str = TICKET_ID, caller: Caller = Depends(authenticate)):
    """Send one customer message and get the reply. Use the same ticket id for the whole conversation.

    A logged-in customer can start a new ticket or continue one of their own. Their identity comes from the
    token: identity fields in `metadata` (email, external_user_id, user_name, account_id) are ignored."""
    payload = body.model_dump()
    if caller.kind == "customer":
        await load_ticket(caller, ticket_id, missing_ok=True)
        metadata = {k: v for k, v in (body.metadata or {}).items() if k not in IDENTITY_FIELDS}
        metadata["external_user_id"] = caller.id
        if caller.name:
            metadata["user_name"] = caller.name
        payload["metadata"] = metadata
    response = await forward("agent", "POST", f"/tickets/{ticket_id}/messages", json=payload)
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
async def get_ticket(ticket_id: str = TICKET_ID, caller: Caller = Depends(authenticate)):
    t = await load_ticket(caller, ticket_id)
    return TicketOut(ticket_id=t["ticket_id"], status=t["status"], channel=t["channel"], issue_type=t["main_issue_type"])


@app.get("/v1/tickets/{ticket_id}/messages")
async def get_messages(ticket_id: str = TICKET_ID, caller: Caller = Depends(authenticate)) -> list[dict]:
    """The conversation as the customer saw it. Internal system notes (escalation handoffs, refund
    requests) stay inside."""
    if caller.kind == "customer":
        await load_ticket(caller, ticket_id)
    response = await forward("tickets", "GET", f"/tickets/{ticket_id}/messages")
    relay_error(response)
    return [m for m in response.json() if m["role"] in CUSTOMER_VISIBLE_ROLES]


# --- Customer sign-up and login (no caller identity yet: limited per client IP) ---------------------------

class EmailIn(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class RegisterIn(EmailIn):
    code: str = Field(pattern=r"^\d{6}$", description="The 6-digit code sent to the email")
    password: str = Field(min_length=10, max_length=200,
                          description="At least 10 characters, with a letter and a digit")


class LoginIn(EmailIn):
    password: str = Field(min_length=1, max_length=200)


class LoginCodeIn(EmailIn):
    code: str = Field(pattern=r"^\d{6}$", description="The 6-digit login code sent to the email")


class LoginCodeSent(BaseModel):
    detail: str
    code_required: bool


class TokenOut(BaseModel):
    access_token: str
    token_type: str
    expires_at: str


async def forward_auth(path: str, body: BaseModel) -> JSONResponse:
    """Relay the auth service's answer: its client errors (400, 401, 409, 422) as they are, a 5xx as a 502."""
    if not jwt_configured():
        raise HTTPException(503, "Customer login is not configured")
    response = await forward("auth", "POST", path, json=body.model_dump())
    if response.status_code >= 500:
        raise HTTPException(502, "Upstream service error")
    return JSONResponse(response.json(), status_code=response.status_code)


@app.post("/v1/auth/register/start", status_code=202, dependencies=[Depends(limit_client_ip)])
async def register_start(body: EmailIn):
    """Step 1 of sign-up: send a verification code to a CultPass customer's email. The answer is the same
    whether or not the email belongs to a customer."""
    return await forward_auth("/register/start", body)


@app.post("/v1/auth/register/complete", status_code=201, response_model=TokenOut,
          dependencies=[Depends(limit_client_ip)])
async def register_complete(body: RegisterIn):
    """Step 2 of sign-up: the code from the email and a password. Returns an access token."""
    return await forward_auth("/register/complete", body)


@app.post("/v1/auth/login", response_model=TokenOut, dependencies=[Depends(limit_client_ip)],
          responses={202: {"model": LoginCodeSent, "description": "Password accepted; a login code was emailed"}})
async def login(body: LoginIn):
    """Step 1 of login: email and password. A login code is then sent to the customer's email (`202`), to send
    to `/v1/auth/login/verify`. If login codes are turned off, this returns the access token directly."""
    return await forward_auth("/login", body)


@app.post("/v1/auth/login/verify", response_model=TokenOut, dependencies=[Depends(limit_client_ip)])
async def login_verify(body: LoginCodeIn):
    """Step 2 of login: the code from the email. Returns an access token; send it as
    `Authorization: Bearer <token>`."""
    return await forward_auth("/login/verify", body)


DEMO_PAGE = pathlib.Path(__file__).with_name("demo.html")


@app.get("/demo", include_in_schema=False)
async def demo_page():
    """A web chat for demos. It holds no secret: the person using it enters a channel API key, which stays
    in their browser, and every message goes through the same authenticated endpoint as any client."""
    return FileResponse(DEMO_PAGE, media_type="text/html")


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
