"""Ticket service: owns UDA-Hub's ticket database (accounts, users, tickets, messages, long-term memory).

The persistence logic is agentic/tools/ticket_tools.py, run here against this service's database.
"""
from typing import Literal, Optional

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from agentic import config
from agentic.tools import ticket_tools as tt
from agentic.tools.db import udahub_engine
from service.common.app import create_app, ping_database, refuse_own_url

refuse_own_url("tickets", config.TICKET_SERVICE_URL)

app = create_app("tickets", "UDA-Hub ticket service",
                 health_check=lambda: ping_database(udahub_engine()), owns_data=True)


class TicketMetadataIn(BaseModel):
    account_id: Optional[str] = Field(None, max_length=64)
    external_user_id: Optional[str] = Field(None, max_length=64)
    user_name: Optional[str] = Field(None, max_length=200)
    channel: Optional[str] = Field(None, max_length=32)
    tags: Optional[str] = Field(None, max_length=500)


class TicketIn(BaseModel):
    metadata: TicketMetadataIn = TicketMetadataIn()


def _ticket_or_404(ticket_id: str) -> dict:
    ticket = tt.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
    return ticket


@app.put("/tickets/{ticket_id}")
def get_or_create_ticket(ticket_id: str, body: TicketIn):
    """Load the ticket, creating it (and its user) on first contact."""
    try:
        return tt.get_or_create_ticket(ticket_id, body.metadata.model_dump(exclude_none=True))
    except ValueError as exc:  # unknown account
        raise HTTPException(422, str(exc))


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    return _ticket_or_404(ticket_id)


class CustomerIn(BaseModel):
    external_user_id: str = Field(min_length=1, max_length=64)
    user_name: str = Field(min_length=1, max_length=200)


@app.put("/tickets/{ticket_id}/customer")
def link_customer(ticket_id: str, body: CustomerIn):
    """Assign a guest ticket to the customer identified during the conversation."""
    _ticket_or_404(ticket_id)
    return {"udahub_user_id": tt.link_ticket_to_customer(ticket_id, body.external_user_id, body.user_name)}


class MessageIn(BaseModel):
    role: Literal["user", "agent", "ai", "system"]
    content: str = Field(max_length=20_000)


@app.post("/tickets/{ticket_id}/messages", status_code=201)
def add_message(ticket_id: str, body: MessageIn):
    _ticket_or_404(ticket_id)
    tt.add_message(ticket_id, body.role, body.content)
    return {"ok": True}


@app.get("/tickets/{ticket_id}/messages")
def get_messages(ticket_id: str):
    _ticket_or_404(ticket_id)
    return tt.get_ticket_messages(ticket_id)


class MetadataUpdate(BaseModel):
    status: Optional[Literal["open", "awaiting_customer", "resolved", "escalated"]] = None
    issue_type: Optional[str] = Field(None, max_length=64)
    tags: Optional[str] = Field(None, max_length=500)


@app.patch("/tickets/{ticket_id}/metadata")
def update_metadata(ticket_id: str, body: MetadataUpdate):
    _ticket_or_404(ticket_id)
    tt.update_ticket_metadata(ticket_id, status=body.status, issue_type=body.issue_type, tags=body.tags)
    return _ticket_or_404(ticket_id)


@app.get("/users/{udahub_user_id}/tickets")
def prior_tickets(udahub_user_id: str, exclude: str = "", limit: int = Query(config.MAX_PRIOR_TICKETS, ge=1, le=50)):
    return tt.get_prior_tickets(udahub_user_id, exclude_ticket_id=exclude, limit=limit)


@app.get("/users/{udahub_user_id}/memories")
def memories(udahub_user_id: str, limit: int = Query(config.MAX_MEMORIES, ge=1, le=100)):
    return tt.get_memories(udahub_user_id, limit=limit)


class MemoryIn(BaseModel):
    account_id: str = Field(min_length=1, max_length=64)
    category: Literal["preference", "personal_fact", "issue_history"]
    content: str = Field(min_length=1, max_length=1000)
    source_ticket_id: str = Field(min_length=1, max_length=64)


@app.post("/users/{udahub_user_id}/memories")
def save_memory(udahub_user_id: str, body: MemoryIn):
    stored = tt.save_memory(body.account_id, udahub_user_id, body.category, body.content, body.source_ticket_id)
    return {"stored": stored}
