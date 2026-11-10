"""What every UDA-Hub service shares: JSON logs, request ids, Prometheus metrics, health checks and
service-to-service authentication.

Internal services (CultPass, Ticket, Knowledge, Agent) require the shared secret INTERNAL_API_TOKEN in the
X-Internal-Token header on every request except /health and /metrics. Only the gateway is reachable from
outside the Docker network; it authenticates clients with API keys and adds the token itself.
"""
import hmac
import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_ID_HEADER = "X-Request-ID"
TOKEN_HEADER = "X-Internal-Token"
OPEN_PATHS = {"/health", "/metrics"}

request_id_var: ContextVar[Optional[str]] = ContextVar("service_request_id", default=None)

REQUESTS = Counter("udahub_http_requests_total", "HTTP requests handled", ["service", "method", "route", "status"])
LATENCY = Histogram("udahub_http_request_seconds", "HTTP request latency", ["service", "method", "route"],
                    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120))


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


def configure_logging(service: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    # uvicorn's access log duplicates the request log below
    logging.getLogger("uvicorn.access").disabled = True


def internal_token() -> Optional[str]:
    return os.getenv("INTERNAL_API_TOKEN") or None


def create_app(service: str, title: str, health_check: Callable[[], None] = None, owns_data: bool = False,
               **kwargs) -> FastAPI:
    """A FastAPI app with the shared middleware. `health_check` raises if a dependency (e.g. the
    database) is down, which turns /health into a 503. A service that `owns_data` runs every agentic data
    function locally against its own database, never over HTTP."""
    configure_logging(service)
    logger = logging.getLogger(service)
    app = FastAPI(title=title, **kwargs)
    app.state.require_token = True

    @app.middleware("http")
    async def _middleware(request: Request, call_next):
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token_ref = request_id_var.set(rid)
        # Also forwarded by the agent's outbound calls (agentic.remote.http)
        try:
            from agentic.remote import http as remote_http
            remote_ref = remote_http.request_id.set(rid)
            local_ref = remote_http._serving_locally.set(True) if owns_data else None
        except ImportError:  # the gateway image doesn't ship the agentic package
            remote_http = remote_ref = local_ref = None
        start = time.perf_counter()
        status = 500
        try:
            expected = internal_token()
            if app.state.require_token and request.url.path not in OPEN_PATHS:
                if expected is None:
                    logger.error("INTERNAL_API_TOKEN is not set; refusing requests")
                    response = JSONResponse({"detail": "Service misconfigured"}, status_code=503)
                elif not hmac.compare_digest(request.headers.get(TOKEN_HEADER, ""), expected):
                    response = JSONResponse({"detail": "Missing or invalid internal token"}, status_code=401)
                else:
                    response = await call_next(request)
            else:
                response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = rid
            return response
        finally:
            route = request.scope.get("route")
            route_path = getattr(route, "path", "unmatched")
            elapsed = time.perf_counter() - start
            if request.url.path not in OPEN_PATHS:
                REQUESTS.labels(service, request.method, route_path, str(status)).inc()
                LATENCY.labels(service, request.method, route_path).observe(elapsed)
                logger.info("%s %s -> %s", request.method, request.url.path, status, extra={"fields": {
                    "method": request.method, "path": request.url.path, "status": status,
                    "duration_ms": round(elapsed * 1000, 1)}})
            request_id_var.reset(token_ref)
            if remote_http is not None:
                remote_http.request_id.reset(remote_ref)
                if local_ref is not None:
                    remote_http._serving_locally.reset(local_ref)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": "Internal server error"}, status_code=500)

    @app.get("/health", include_in_schema=False)
    def health():
        if health_check is not None:
            try:
                health_check()
            except Exception as exc:
                logger.warning("Health check failed: %s", exc)
                return JSONResponse({"status": "unhealthy", "service": service}, status_code=503)
        return {"status": "ok", "service": service}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def ping_database(engine) -> None:
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def refuse_own_url(service: str, url: Optional[str]) -> None:
    """A service that owns a database must use it directly. If its own service URL were set, its data
    functions would call themselves over HTTP in a loop."""
    if url:
        raise RuntimeError(f"The {service} service must not have its own service URL set ({url}).")
