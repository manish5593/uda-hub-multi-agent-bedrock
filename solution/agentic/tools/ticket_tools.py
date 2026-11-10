"""Ticket persistence and long-term customer memory in the core (udahub) database."""
import re
import uuid
from typing import Optional

from agentic import config
from agentic.remote.http import remote
from agentic.tools.db import session_scope, udahub_engine
from data.models import udahub


def _get_or_create_user(session, account_id: str, external_user_id: str, user_name: str) -> udahub.User:
    user = (
        session.query(udahub.User)
        .filter_by(account_id=account_id, external_user_id=external_user_id)
        .first()
    )
    if user is None:
        user = udahub.User(
            user_id=str(uuid.uuid4()),
            account_id=account_id,
            external_user_id=external_user_id,
            user_name=user_name,
        )
        session.add(user)
        session.flush()
    return user


def _ticket_info(ticket: udahub.Ticket) -> dict:
    meta = ticket.ticket_metadata
    return {
        "ticket_id": ticket.ticket_id,
        "account_id": ticket.account_id,
        "udahub_user_id": ticket.user_id,
        "external_user_id": ticket.user.external_user_id,
        "user_name": ticket.user.user_name,
        "channel": ticket.channel,
        "status": meta.status if meta else "open",
        "main_issue_type": meta.main_issue_type if meta else None,
        "tags": meta.tags if meta else None,
    }


@remote("tickets")
def get_or_create_ticket(ticket_id: str, metadata: Optional[dict] = None) -> dict:
    """Load a ticket, creating it (and its user) if it doesn't exist yet.

    `metadata` may carry `account_id`, `external_user_id`, `user_name`, `channel`, `tags`.
    Unknown customers are attached to a per-account guest user until identified.
    """
    metadata = metadata or {}
    with session_scope(udahub_engine()) as session:
        ticket = session.get(udahub.Ticket, ticket_id)
        if ticket is None:
            account_id = metadata.get("account_id", config.DEFAULT_ACCOUNT_ID)
            if session.get(udahub.Account, account_id) is None:
                raise ValueError(f"Unknown account '{account_id}'. Run 02_core_db_setup first.")
            user = _get_or_create_user(
                session,
                account_id,
                metadata.get("external_user_id", config.GUEST_EXTERNAL_USER_ID),
                metadata.get("user_name", "Guest"),
            )
            ticket = udahub.Ticket(
                ticket_id=ticket_id,
                account_id=account_id,
                user_id=user.user_id,
                channel=metadata.get("channel", "chat"),
            )
            ticket.ticket_metadata = udahub.TicketMetadata(
                ticket_id=ticket_id, status="open", tags=metadata.get("tags")
            )
            session.add(ticket)
            session.flush()
        return _ticket_info(ticket)


@remote("tickets")
def get_ticket(ticket_id: str) -> Optional[dict]:
    """The ticket's details, or None if it doesn't exist."""
    with session_scope(udahub_engine()) as session:
        ticket = session.get(udahub.Ticket, ticket_id)
        return _ticket_info(ticket) if ticket else None


@remote("tickets")
def ticket_exists(ticket_id: str) -> bool:
    with session_scope(udahub_engine()) as session:
        return session.get(udahub.Ticket, ticket_id) is not None


@remote("tickets")
def link_ticket_to_customer(ticket_id: str, external_user_id: str, user_name: str) -> str:
    """Re-assign a guest ticket to an identified customer. Returns the udahub user id."""
    with session_scope(udahub_engine()) as session:
        ticket = session.get(udahub.Ticket, ticket_id)
        user = _get_or_create_user(session, ticket.account_id, external_user_id, user_name)
        ticket.user_id = user.user_id
        return user.user_id


@remote("tickets")
def add_message(ticket_id: str, role: str, content: str) -> None:
    with session_scope(udahub_engine()) as session:
        session.add(
            udahub.TicketMessage(
                message_id=str(uuid.uuid4()),
                ticket_id=ticket_id,
                role=udahub.RoleEnum(role),
                content=content,
            )
        )


@remote("tickets")
def update_ticket_metadata(
    ticket_id: str, status: Optional[str] = None, issue_type: Optional[str] = None, tags: Optional[str] = None
) -> None:
    with session_scope(udahub_engine()) as session:
        meta = session.get(udahub.TicketMetadata, ticket_id)
        if meta is None:
            meta = udahub.TicketMetadata(ticket_id=ticket_id, status=status or "open")
            session.add(meta)
        if status:
            meta.status = status
        if issue_type:
            meta.main_issue_type = issue_type
        if tags:
            existing = [t.strip() for t in (meta.tags or "").split(",") if t.strip()]
            new = [t.strip() for t in tags.split(",") if t.strip() and t.strip() not in existing]
            meta.tags = ", ".join(existing + new)


@remote("tickets")
def get_ticket_messages(ticket_id: str) -> list[dict]:
    with session_scope(udahub_engine()) as session:
        rows = (
            session.query(udahub.TicketMessage)
            .filter_by(ticket_id=ticket_id)
            .order_by(udahub.TicketMessage.created_at)
            .all()
        )
        return [{"role": m.role.value, "content": m.content} for m in rows]


@remote("tickets")
def get_prior_tickets(udahub_user_id: str, exclude_ticket_id: str, limit: int = config.MAX_PRIOR_TICKETS) -> list[dict]:
    """Summaries of the customer's previous tickets (long-term memory of past resolutions)."""
    with session_scope(udahub_engine()) as session:
        tickets = (
            session.query(udahub.Ticket)
            .filter(udahub.Ticket.user_id == udahub_user_id, udahub.Ticket.ticket_id != exclude_ticket_id)
            .order_by(udahub.Ticket.created_at.desc())
            .limit(limit)
            .all()
        )
        summaries = []
        for t in tickets:
            msgs = sorted(t.messages, key=lambda m: m.created_at)
            first_user = next((m.content for m in msgs if m.role == udahub.RoleEnum.user), None)
            last_reply = next(
                (m.content for m in reversed(msgs) if m.role in (udahub.RoleEnum.ai, udahub.RoleEnum.agent)),
                None,
            )
            summaries.append(
                {
                    "ticket_id": t.ticket_id,
                    "created_at": t.created_at.strftime("%Y-%m-%d") if t.created_at else None,
                    "status": t.ticket_metadata.status if t.ticket_metadata else None,
                    "issue_type": t.ticket_metadata.main_issue_type if t.ticket_metadata else None,
                    "customer_message": first_user,
                    "last_reply": last_reply,
                }
            )
        return summaries


# ---------------------------------------------------------------------------
# Long-term memory
# ---------------------------------------------------------------------------

@remote("tickets")
def get_memories(udahub_user_id: str, limit: int = config.MAX_MEMORIES) -> list[dict]:
    with session_scope(udahub_engine()) as session:
        rows = (
            session.query(udahub.CustomerMemory)
            .filter_by(user_id=udahub_user_id)
            .order_by(udahub.CustomerMemory.updated_at.desc())
            .limit(limit)
            .all()
        )
        return [{"category": m.category, "content": m.content} for m in rows]


def _normalize(text: str) -> str:
    """Lowercase and drop punctuation and extra spaces, so trivial variants compare equal."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


@remote("tickets")
def save_memory(account_id: str, udahub_user_id: str, category: str, content: str, source_ticket_id: str) -> bool:
    """Store a memory unless the customer already has the same one. Returns True if stored.

    Comparison ignores case, punctuation and spacing, so "Likes jazz." duplicates "likes jazz".
    """
    key = _normalize(content)
    if not key:
        return False
    with session_scope(udahub_engine()) as session:
        existing = session.query(udahub.CustomerMemory.content).filter_by(user_id=udahub_user_id).all()
        if any(_normalize(row.content) == key for row in existing):
            return False
        session.add(
            udahub.CustomerMemory(
                memory_id=str(uuid.uuid4()),
                account_id=account_id,
                user_id=udahub_user_id,
                category=category,
                content=content.strip(),
                source_ticket_id=source_ticket_id,
            )
        )
        return True
