"""Shared HTTP plumbing for calls between the services: base URLs, auth, request ids and errors."""
import contextlib
import contextvars
import functools
import importlib
from typing import Optional

import httpx

from agentic import config

# Request id of the inbound request being handled, forwarded on every outbound call so one customer
# message can be followed across the services' logs
request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)

_clients: dict[str, httpx.Client] = {}


def _base_url(service: str) -> Optional[str]:
    return {
        "cultpass": config.CULTPASS_SERVICE_URL,
        "tickets": config.TICKET_SERVICE_URL,
        "knowledge": config.KNOWLEDGE_SERVICE_URL,
    }[service]


# Set while a service handles a request against its own database: everything it runs is local, even if
# a service URL happens to be configured (that would make the service call itself)
_serving_locally: contextvars.ContextVar[bool] = contextvars.ContextVar("serving_locally", default=False)


@contextlib.contextmanager
def serving_locally():
    token = _serving_locally.set(True)
    try:
        yield
    finally:
        _serving_locally.reset(token)


def is_remote(service: str) -> bool:
    if _serving_locally.get():
        return False
    return service in _clients or bool(_base_url(service))


def use_client(service: str, client: Optional[httpx.Client]) -> None:
    """Send `service` calls through `client` (tests pass a FastAPI TestClient); None goes back to the URL."""
    if client is None:
        _clients.pop(service, None)
    else:
        _clients[service] = client


def _client(service: str) -> httpx.Client:
    if service not in _clients:
        _clients[service] = httpx.Client(
            base_url=_base_url(service),
            timeout=config.SERVICE_TIMEOUT_SECONDS,
            # Retries only failed connections, where the request never reached the service, so it is
            # safe for writes too
            transport=httpx.HTTPTransport(retries=2),
        )
    return _clients[service]


def call(service: str, method: str, path: str, *, params: dict = None, json=None, missing_ok: bool = False):
    """Call a service and return its JSON body.

    404 returns None when `missing_ok`. 400 and 422 raise ValueError with the service's message, matching
    what the local function raises for bad input. Other errors raise httpx.HTTPStatusError.
    """
    headers = {}
    if config.INTERNAL_API_TOKEN:
        headers["X-Internal-Token"] = config.INTERNAL_API_TOKEN
    if request_id.get():
        headers["X-Request-ID"] = request_id.get()
    if params:
        params = {k: v for k, v in params.items() if v is not None}
    response = _client(service).request(method, path, params=params, json=json, headers=headers)
    if response.status_code == 404 and missing_ok:
        return None
    if response.status_code in (400, 422):
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ValueError(f"{service} service rejected {method} {path}: {detail}")
    response.raise_for_status()
    return response.json()


def remote(service: str):
    """Run the decorated function in `service` when that service is configured, locally otherwise.

    The remote version is the function with the same name in agentic.remote.<service>.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if is_remote(service):
                module = importlib.import_module(f"agentic.remote.{service}")
                return getattr(module, func.__name__)(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator
