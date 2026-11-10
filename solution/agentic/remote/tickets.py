"""Remote versions of the ticket and memory functions in agentic/tools/ticket_tools.py (Ticket service)."""
from typing import Optional
from urllib.parse import quote

from agentic import config
from agentic.remote.http import call


def _id(value: str) -> str:
    return quote(str(value), safe="")


def get_or_create_ticket(ticket_id: str, metadata: Optional[dict] = None) -> dict:
    return call("tickets", "PUT", f"/tickets/{_id(ticket_id)}", json={"metadata": metadata or {}})


def get_ticket(ticket_id: str) -> Optional[dict]:
    return call("tickets", "GET", f"/tickets/{_id(ticket_id)}", missing_ok=True)


def ticket_exists(ticket_id: str) -> bool:
    return get_ticket(ticket_id) is not None


def link_ticket_to_customer(ticket_id: str, external_user_id: str, user_name: str) -> str:
    result = call("tickets", "PUT", f"/tickets/{_id(ticket_id)}/customer",
                  json={"external_user_id": external_user_id, "user_name": user_name})
    return result["udahub_user_id"]


def add_message(ticket_id: str, role: str, content: str) -> None:
    call("tickets", "POST", f"/tickets/{_id(ticket_id)}/messages", json={"role": role, "content": content})


def update_ticket_metadata(ticket_id: str, status: Optional[str] = None, issue_type: Optional[str] = None,
                           tags: Optional[str] = None) -> None:
    call("tickets", "PATCH", f"/tickets/{_id(ticket_id)}/metadata",
         json={"status": status, "issue_type": issue_type, "tags": tags})


def get_ticket_messages(ticket_id: str) -> list[dict]:
    return call("tickets", "GET", f"/tickets/{_id(ticket_id)}/messages")


def get_prior_tickets(udahub_user_id: str, exclude_ticket_id: str, limit: int = config.MAX_PRIOR_TICKETS) -> list[dict]:
    return call("tickets", "GET", f"/users/{_id(udahub_user_id)}/tickets",
                params={"exclude": exclude_ticket_id, "limit": limit})


def get_memories(udahub_user_id: str, limit: int = config.MAX_MEMORIES) -> list[dict]:
    return call("tickets", "GET", f"/users/{_id(udahub_user_id)}/memories", params={"limit": limit})


def save_memory(account_id: str, udahub_user_id: str, category: str, content: str, source_ticket_id: str) -> bool:
    result = call("tickets", "POST", f"/users/{_id(udahub_user_id)}/memories",
                  json={"account_id": account_id, "category": category, "content": content,
                        "source_ticket_id": source_ticket_id})
    return result["stored"]
