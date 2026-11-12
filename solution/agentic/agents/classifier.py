"""Classifier agent: understands the ticket and produces a structured classification."""
from langchain_core.messages import HumanMessage, SystemMessage

from agentic import verification
from agentic.agents.common import add_trace, conversation_text, customer_context_block, last_user_message
from agentic.llm import get_llm
from agentic.state import TicketClassification

CLASSIFIER_PROMPT = """You are the triage classifier of UDA-Hub, the customer support platform of CultPass,
a subscription that gives members access to cultural experiences (museums, concerts, tours...).

Classify the customer's CURRENT request (the latest customer message, using earlier messages as context).

Issue types:
- login_access: cannot log in, password reset, verification emails
- reservation: book, cancel, check or change reservations; find, recommend or check availability of
  experiences; attendance. Recommendations and availability need the real experience catalog,
  so set requires_account_action=true for them.
- subscription: plan details, tiers, quota, pause/cancel/resume subscription
- billing_refund: charges, payment methods, invoices, refunds
- account_blocked: the customer says the account is blocked/suspended/restricted
- technical: app crashes, errors, bugs, a QR code or ticket that won't scan
- account_privacy: change email/profile, delete account, personal data
- general_info: how the service works, what is included, opening hours
- complaint: dissatisfaction without a specific actionable request
- legal_or_safety: legal threats, fraud, data-protection complaints, harassment, safety incidents
- other: anything else

Urgency: low (informational), medium (needs help soon), high (blocked from using the service or an
event is imminent), critical (safety risk, legal threat, fraud).
Set confidence low (<0.5) when the request is ambiguous or unrelated to CultPass."""


EMPTY_MESSAGE = {"issue_type": "other", "urgency": "low", "sentiment": "neutral", "summary": "empty message",
                 "requires_account_action": False, "human_requested": False, "customer_email": None,
                 "language": "English", "confidence": 1.0, "empty_message": True}


def classify(state: dict) -> dict:
    if not last_user_message(state).strip():
        # Nothing to classify: no LLM call, the supervisor asks the customer what they need
        return {"classification": dict(EMPTY_MESSAGE), "trace": add_trace(
            state, "[classifier] empty message", event="classify", issue_type="other", empty_message=True)}
    pending = state.get("pending_verification")
    if (not (state.get("customer") or {}).get("identified") and verification.is_pending(pending)
            and pending.get("classification") and verification.is_code_only(last_user_message(state))):
        # Just the verification code: the request is the one made when the customer was asked who they are
        c = dict(pending["classification"])
        return {"classification": c, "trace": add_trace(
            state, f"[classifier] verification code: continuing the {c.get('issue_type')} request",
            event="classify", issue_type=c.get("issue_type"), verification_code=True)}
    llm = get_llm().with_structured_output(TicketClassification)
    result: TicketClassification = llm.invoke([
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(content=f"Context:\n{customer_context_block(state)}\n\n"
                             f"Conversation:\n{conversation_text(state)}"),
    ])
    c = result.model_dump()
    entry = (f"[classifier] {c['issue_type']} urgency={c['urgency']} sentiment={c['sentiment']} "
             f"account_action={c['requires_account_action']} confidence={c['confidence']:.2f}")
    return {"classification": c, "trace": add_trace(
        state, entry, event="classify",
        **{k: c[k] for k in ("issue_type", "urgency", "sentiment", "requires_account_action",
                             "human_requested", "language", "confidence")})}
