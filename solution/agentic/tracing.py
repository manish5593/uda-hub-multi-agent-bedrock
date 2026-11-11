"""Optional Langfuse tracing for graph runs.

Tracing is on only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (LANGFUSE_BASE_URL picks the
server; the SDK's default is Langfuse Cloud). Without them, or without the `langfuse` package, every run
works exactly as before. With them, each customer message becomes one Langfuse trace: every node, model
call (prompt, response, tokens) and tool call, grouped into a session per ticket.
"""
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger("udahub")

TRACE_NAME = "udahub-ticket"


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@lru_cache(maxsize=1)
def _callback_handler_class():
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.warning("LANGFUSE_* keys are set but the langfuse package is not installed: tracing is off")
        return None
    return CallbackHandler


def run_config(ticket_id: str, metadata: Optional[dict] = None) -> dict:
    """LangGraph config for one turn: the checkpointer thread, plus the Langfuse callback when enabled."""
    config: dict = {"configurable": {"thread_id": ticket_id}}
    handler_class = _callback_handler_class() if langfuse_enabled() else None
    if handler_class is None:
        return config

    metadata = metadata or {}
    trace_metadata = {"langfuse_trace_name": TRACE_NAME, "langfuse_session_id": ticket_id}
    # The CultPass user id, never the email: Langfuse shows it on every trace
    if metadata.get("external_user_id"):
        trace_metadata["langfuse_user_id"] = str(metadata["external_user_id"])
    tags = [str(metadata[key]) for key in ("account_id", "channel") if metadata.get(key)]
    if tags:
        trace_metadata["langfuse_tags"] = tags

    # One handler per run: it holds that run's span tree. The Langfuse client behind it is shared.
    config.update(callbacks=[handler_class()], run_name=TRACE_NAME, metadata=trace_metadata)
    return config
