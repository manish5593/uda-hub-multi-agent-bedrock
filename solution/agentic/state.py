"""Shared graph state for the UDA-Hub multi-agent workflow."""
from typing import Annotated, Literal, Optional

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


IssueType = Literal[
    "login_access",
    "reservation",
    "subscription",
    "billing_refund",
    "account_blocked",
    "technical",
    "account_privacy",
    "general_info",
    "complaint",
    "legal_or_safety",
    "other",
]

Urgency = Literal["low", "medium", "high", "critical"]

Route = Literal["knowledge_agent", "account_agent", "escalation_agent", "request_info"]


class TicketClassification(BaseModel):
    """Structured output of the Classifier agent."""
    issue_type: IssueType = Field(description="Main category of the customer's issue.")
    urgency: Urgency = Field(description="How time-sensitive / severe the issue is.")
    sentiment: Literal["positive", "neutral", "frustrated", "angry"]
    summary: str = Field(description="One-sentence summary of what the customer needs.")
    requires_account_action: bool = Field(
        description="True if resolving it needs the customer's account data or an action on "
        "their account (look up/cancel/make reservations, check or change subscription, refunds). "
        "False for questions about how something works, what the rules are, or what to do when something "
        "doesn't work, even when they are about the customer's own reservation, subscription or billing: "
        "the knowledge base answers those without the customer's data."
    )
    human_requested: bool = Field(description="True if the customer explicitly asks for a human.")
    customer_email: Optional[str] = Field(
        default=None, description="Email address the customer mentioned in the conversation, if any."
    )
    language: str = Field(description="Language the customer writes in, e.g. 'English'.")
    confidence: float = Field(ge=0, le=1, description="Confidence in this classification, 0-1.")


class ResolutionAssessment(BaseModel):
    """Structured output of the quality gate that decides resolve vs. escalate."""
    confidence: float = Field(
        ge=0, le=1, description="Confidence that the draft fully and correctly resolves the request."
    )
    grounded: bool = Field(
        description="True if every factual claim is supported by the knowledge/tool context."
    )
    answered_by_evidence: bool = Field(
        description="True only if the knowledge base articles or tool results actually answer the customer's "
        "question. False if they are merely related, or the draft has to send the customer elsewhere."
    )
    needs_human: bool = Field(
        description="True only if a human must take over NOW: policy requires one (refund approval, blocked "
        "account, legal, identity verification), the draft cannot resolve the issue, or documented steps "
        "already failed. False when the draft gives the steps to try and only offers escalation if they "
        "don't work."
    )
    awaiting_customer: bool = Field(
        description="True if the draft asks the customer a question they must answer before the request can "
        "be completed (which reservation, which email, which date). False when the draft answers the request "
        "and only offers more help, e.g. 'let me know if...' or 'if you'd like, I can...'."
    )
    reason: str = Field(description="Short justification for the decision.")


class CustomerContext(TypedDict, total=False):
    account_id: str
    udahub_user_id: str
    external_user_id: str
    name: str
    email: str
    is_blocked: bool
    subscription: Optional[dict]
    identified: bool


class UdaHubState(TypedDict, total=False):
    # Conversation (short-term memory, persisted per thread by the checkpointer)
    messages: Annotated[list, add_messages]

    # Ticket input metadata (optional on input; resolved from udahub.db otherwise)
    ticket_id: str
    metadata: dict  # e.g. {"account_id", "external_user_id", "email", "channel", "urgency"}
    channel: str
    ticket_status: str
    ticket_issue_type: Optional[str]  # main issue type recorded on the ticket by earlier turns

    # Context loaded from long-term memory
    customer: CustomerContext
    memories: list[str]
    prior_tickets: list[dict]
    ticket_history: list[dict]  # messages on the ticket before this session

    # Per-turn decision data
    classification: dict
    route: Route
    route_reason: str
    knowledge: list[dict]
    draft_response: Optional[str]
    resolver: Optional[str]
    actions_taken: list[dict]
    assessment: dict
    escalation: Optional[dict]
    final_response: str
    trace: list[str]
