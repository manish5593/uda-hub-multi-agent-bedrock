"""Supervisor: identifies the customer and routes the ticket to the right agent.

Routing is deterministic and auditable: the LLM classifies, the supervisor applies
business rules on top of that classification.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from agentic import config, verification
from agentic.agents.common import add_trace, last_user_message, text_of
from agentic.agents.memory_agent import build_customer, recall
from agentic.llm import get_llm
from agentic.tools import cultpass_tools, ticket_tools

ACCOUNT_ISSUES = {"reservation", "subscription", "billing_refund"}
BLOCKED_SENSITIVE_ISSUES = {"login_access", "reservation", "subscription", "billing_refund", "account_blocked", "complaint"}
# Problems that should reach a human when they come back: the documented fix didn't work.
# Routine requests (reservations, subscription changes, questions) are expected to repeat.
REPEAT_ESCALATION_ISSUES = {"login_access", "technical", "billing_refund", "account_privacy", "complaint"}


def repeat_issue_types(state: dict) -> set[str]:
    """Issue types this customer already reported: in previous tickets, or earlier on this ticket
    when it was marked resolved (the customer writing again means the answer didn't fix it)."""
    types = {t["issue_type"] for t in state.get("prior_tickets") or [] if t.get("issue_type")}
    if state.get("ticket_status") == "resolved" and state.get("ticket_issue_type"):
        types.add(state["ticket_issue_type"])
    return types


URGENCY_LEVELS = ["low", "medium", "high", "critical"]


def effective_urgency(classified, reported) -> str:
    """The more severe of the classifier's urgency and the urgency reported in the ticket metadata,
    so the LLM can raise a ticket's urgency but never lower what the source system reported."""
    levels = [u for u in (classified, reported) if u in URGENCY_LEVELS]
    return max(levels, key=URGENCY_LEVELS.index) if levels else "medium"


def route_ticket(classification: dict, customer: dict, ticket_status: str,
                 repeat_issues: frozenset = frozenset(), reported_urgency=None) -> tuple[str, str]:
    """Return (route, reason). Pure function so the decision logic is easy to test."""
    c = classification
    urgency = effective_urgency(c.get("urgency"), reported_urgency)
    if c.get("empty_message"):
        return "request_info", "empty message: asking the customer to describe their issue"
    if ticket_status == "escalated":
        return "escalation_agent", "ticket is already with the human team; forwarding the new information"
    if c.get("human_requested"):
        return "escalation_agent", "customer asked for a human"
    if c.get("issue_type") == "legal_or_safety":
        return "escalation_agent", "legal/safety issue requires a human"
    if urgency == "critical":
        source = "reported in the ticket metadata" if reported_urgency == "critical" else "classified"
        return "escalation_agent", f"critical urgency ({source}) requires a human"
    if c.get("confidence", 0) < config.CLASSIFIER_MIN_CONFIDENCE:
        return "escalation_agent", f"classification confidence {c.get('confidence', 0):.2f} is too low"
    if customer.get("identified") and customer.get("is_blocked") and (
        c.get("issue_type") in BLOCKED_SENSITIVE_ISSUES or c.get("requires_account_action")
    ):
        return "escalation_agent", "account is blocked: policy requires Trust & Safety review"
    if c.get("issue_type") == "account_blocked" and customer.get("identified"):
        return "escalation_agent", "customer reports a blocked account"
    if c.get("issue_type") in REPEAT_ESCALATION_ISSUES and c.get("issue_type") in repeat_issues:
        return "escalation_agent", f"repeat {c.get('issue_type')} issue: the customer reported it before"
    needs_account = c.get("requires_account_action") or c.get("issue_type") in ACCOUNT_ISSUES
    if needs_account and not customer.get("identified"):
        if not c.get("requires_account_action"):
            # A guest asking how reservations, the subscription or billing work: the knowledge base answers
            # that without their account, so don't ask who they are first. If no article answers it, the
            # quality gate escalates as for any other knowledge question.
            return "knowledge_agent", f"{c.get('issue_type')} question that needs no account data"
        if c.get("issue_type") not in ACCOUNT_ISSUES:
            # Knowing who the guest is only helps if an account tool can act on the request, and the tools
            # cover reservations, the subscription and billing. A forgotten password, a QR code that won't
            # scan or deleting an account are answered from the knowledge base, even when the classifier
            # marks them as account actions (Claude and Nova do; gpt-4o-mini doesn't).
            return "knowledge_agent", f"{c.get('issue_type')}: no account tool acts on it, so the knowledge base answers"
        return "request_info", "account action needed but customer is not identified"
    if needs_account:
        return "account_agent", f"{c.get('issue_type')} needs account data/actions"
    return "knowledge_agent", f"{c.get('issue_type')} can be answered from the knowledge base"


def _ask(state: dict, update: dict, info_request: str, reason: str) -> dict:
    """Route to request_info with what to ask for (see REQUEST_INFO_MESSAGES)."""
    return {**update, "route": "request_info", "route_reason": reason, "info_request": info_request,
            "trace": add_trace(state, f"[supervisor] route -> request_info: {reason}", event="route",
                               route="request_info", reason=reason)}


def supervise(state: dict) -> dict:
    update = {}
    customer = state.get("customer") or {}
    classification = state.get("classification") or {}
    email = classification.get("customer_email")
    pending = state.get("pending_verification")

    # A guest who typed an email proves it with the code sent there before anything on that account is
    # shown or changed (agentic/verification.py)
    if not customer.get("identified"):
        code = verification.find_code(last_user_message(state)) if verification.is_pending(pending) else None
        if code:
            kept, outcome = verification.check(pending, code)
            update["pending_verification"] = kept
            if outcome != "verified":
                state = {**state, "trace": add_trace(state, f"[supervisor] verification code {outcome}",
                                                     event="verify", outcome=outcome)}
                return _ask(state, update, f"code_{outcome}", f"verification code {outcome}")
            ticket_tools.link_ticket_to_customer(state["ticket_id"], pending["user_id"], pending["user_name"])
            customer = build_customer(ticket_tools.get_or_create_ticket(state["ticket_id"]))
            # Carry on with the request the customer made when they were asked who they are
            classification = pending.get("classification") or classification
            update.update(customer=customer, classification=classification, **recall(customer, state["ticket_id"]))
            state = {**state, "trace": add_trace(
                state, f"[supervisor] identified customer {pending['user_name']}: verified the code sent to "
                       f"{pending['email']}", event="identify", customer=pending["user_id"], method="email_code")}
        elif email:
            profile = cultpass_tools.find_customer_by_email(email)
            new, outcome = verification.start(email, profile, classification, pending)
            update["pending_verification"] = new
            found = "an account" if profile else "no account"
            state = {**state, "trace": add_trace(state, f"[supervisor] {email} ({found}): code {outcome}",
                                                 event="verify", outcome=f"code_{outcome}", account_found=bool(profile))}
            if outcome == "too_many":
                reason = "identity verification: too many codes requested on this ticket, a human must verify"
                return {**update, "route": "escalation_agent", "route_reason": reason, "trace": add_trace(
                    state, f"[supervisor] route -> escalation_agent: {reason}", event="route",
                    route="escalation_agent", reason=reason)}
            # Same reply whether or not the email has an account
            return _ask(state, {**update, "info_request_email": email.strip().lower()}, "code_sent",
                        "verification code sent to the email the customer gave")

    route, reason = route_ticket(classification, customer, state.get("ticket_status", "open"),
                                 frozenset(repeat_issue_types({**state, **update})),
                                 reported_urgency=(state.get("metadata") or {}).get("urgency"))
    if route == "request_info" and not customer.get("identified") and verification.is_pending(pending):
        # Asked who they are, the customer hasn't typed the code yet: remind them of it
        update.update(info_request="code_reminder", info_request_email=pending["email"])
    update.update(route=route, route_reason=reason,
                  trace=add_trace(state, f"[supervisor] route -> {route}: {reason}", event="route", route=route, reason=reason))
    return update


REQUEST_INFO_MESSAGE = (
    "I'd be happy to help with that! To look into your account, could you please share the email "
    "address registered with your CultPass account?"
)


EMPTY_MESSAGE_REPLY = "Hi! It looks like your message came through empty. What can I help you with today?"

TRANSLATE_PROMPT = """Translate this customer support message into {language}.
Keep the meaning, the friendly tone and any email address exactly as they are.
Reply with the translated message only."""


def _in_customer_language(message: str, language: str) -> str:
    """Return the fixed message in the customer's language. English needs no LLM call."""
    if not language or language.strip().lower() in {"english", "en"}:
        return message
    reply = get_llm().invoke([SystemMessage(content=TRANSLATE_PROMPT.format(language=language)),
                              HumanMessage(content=message)])
    return text_of(reply).strip() or message


# What request_info says for each step of verifying a typed email. The text never says whether the email
# has an account.
REQUEST_INFO_MESSAGES = {
    "code_sent": ("Thanks! To protect your account, I've sent a 6-digit code to {email} if it's registered "
                  "with CultPass. Please type the code here to continue. If nothing arrives within a few minutes, "
                  "check the address and send it again."),
    "code_reminder": ("To continue, please type the 6-digit code I sent to {email}. If it didn't arrive, "
                      "send your email address again."),
    "code_wrong": "That code doesn't match. Please check the latest email from us and type the 6-digit code again.",
    "code_burnt": ("That code didn't match too many times, so it no longer works. Please send your email "
                   "address again and I'll send a new code."),
    "code_expired": "That code has expired. Please send your email address again and I'll send a new code.",
}


# For the trace: "[request_info] asked customer <...>"
ASKED = {"code_sent": "for the code sent to their email", "code_reminder": "again for the code sent to their email",
         "code_wrong": "to retry the code (wrong code)", "code_burnt": "for their email again (code burnt)",
         "code_expired": "for their email again (code expired)"}


def request_info(state: dict) -> dict:
    """Ask the customer for the information needed to continue (keeps the ticket open)."""
    classification = state.get("classification") or {}
    kind = state.get("info_request")
    message = REQUEST_INFO_MESSAGE
    if classification.get("empty_message"):
        message = EMPTY_MESSAGE_REPLY
    elif kind in REQUEST_INFO_MESSAGES:
        message = REQUEST_INFO_MESSAGES[kind].format(email=state.get("info_request_email") or "your email")
    language = classification.get("language", "English")
    asked = ("what they need help with" if classification.get("empty_message")
             else ASKED.get(kind, "to identify themselves"))
    return {"final_response": _in_customer_language(message, language),
            "trace": add_trace(state, f"[request_info] asked customer {asked} ({language})",
                               event="request_info", asked=asked, language=language)}
