"""Escalation agent: hands the ticket over to the right human team with a complete summary."""
import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic.agents.common import add_trace, conversation_text, customer_context_block
from agentic.agents.quality_gate import should_escalate
from agentic.llm import get_llm
from agentic.tools import ticket_tools


class EscalationHandoff(BaseModel):
    team: Literal["trust_and_safety", "billing", "technical_support", "privacy_legal", "general_support"]
    priority: Literal["low", "medium", "high", "urgent"]
    summary: str = Field(description="What the customer needs, in 1-3 sentences, for the human agent.")
    steps_taken: list[str] = Field(description="What the AI already checked or did (tools, articles).")
    recommended_next_step: str
    customer_message: str = Field(
        description="Reply to the customer: empathetic, says a specialist will follow up (Mon-Fri 9:00-18:00, "
        "reply by email within 1 business day), includes any reference ids, makes no promises."
    )


ESCALATION_PROMPT = """You are the escalation specialist of CultPass support. The ticket must be handed to a human.
Write an internal handoff note for the human team and a reply for the customer in their language ({language}).
Never reveal internal reasons for an account block and never promise refunds or outcomes.
If actions were already completed on the customer's account (listed below), the reply must tell the customer
they are done, then say what the human team will follow up on.

Escalation reason: {reason}
Actions already completed: {completed}"""


def completed_actions(actions: list[dict]) -> list[dict]:
    """The account changes that succeeded this turn, each with the sentence that tells the customer and the
    words a reply must contain to have told them. An escalated reply once said "a specialist will help you book"
    after the booking had been made."""
    done = []
    for a in actions or []:
        r = a.get("result") if isinstance(a.get("result"), dict) else {}
        if not r.get("success"):
            continue
        if a["tool"] == "book_experience":
            done.append({"text": f"Your reservation for {r['experience']} on {r['when']} is booked "
                                 f"(reservation {r['reservation_id']}).", "must_mention": r["experience"]})
        elif a["tool"] == "change_reservation_date":
            done.append({"text": f"Your reservation is now for {r['experience']} on {r['when']} "
                                 f"(reservation {r['reservation_id']}); the previous one is cancelled.",
                         "must_mention": r["experience"]})
        elif a["tool"] == "cancel_my_reservation":
            done.append({"text": f"Your reservation for {r['experience']} is cancelled.",
                         "must_mention": r["experience"]})
        elif a["tool"] == "change_subscription":
            done.append({"text": f"Your subscription is now {r['new_status']}.", "must_mention": r["new_status"]})
    return done


def escalate(state: dict) -> dict:
    if state.get("route") == "escalation_agent":
        reason = state.get("route_reason", "")
    else:
        reason = should_escalate(state)[1]

    evidence = {
        "classification": state.get("classification"),
        "draft_reply_not_sent": state.get("draft_response"),
        "actions_taken": state.get("actions_taken"),
        "knowledge_used": [a["title"] for a in state.get("knowledge", [])],
    }
    done = completed_actions(state.get("actions_taken"))
    language = state.get("classification", {}).get("language", "English")
    handoff: EscalationHandoff = get_llm().with_structured_output(EscalationHandoff).invoke([
        SystemMessage(content=ESCALATION_PROMPT.format(
            language=language, reason=reason, completed="; ".join(d["text"] for d in done) or "none")),
        HumanMessage(content=f"Customer context:\n{customer_context_block(state)}\n\n"
                             f"Conversation:\n{conversation_text(state)}\n\n"
                             f"Work done so far:\n{json.dumps(evidence, indent=1, default=str)}"),
    ])
    h = handoff.model_dump()
    # Code checks the reply told the customer about every completed change, and adds what it left out
    missing = [d["text"] for d in done if d["must_mention"].lower() not in h["customer_message"].lower()]
    if missing:
        from agentic.agents.supervisor import _in_customer_language
        note = _in_customer_language("Already done: " + " ".join(missing), language)
        h["customer_message"] = f"{h['customer_message'].rstrip()}\n\n{note}"
    ticket_tools.add_message(state["ticket_id"], "system", "ESCALATION HANDOFF\n" + json.dumps(
        {"reason": reason, **{k: v for k, v in h.items() if k != "customer_message"}}, indent=1))
    return {
        "escalation": {"reason": reason, **h},
        "final_response": h["customer_message"],
        "trace": add_trace(state, f"[escalation_agent] -> {h['team']} ({h['priority']}): {reason}"
                                  + (f" (added to the reply: {len(missing)} completed action(s))" if missing else ""),
                           event="escalate", team=h["team"], priority=h["priority"], reason=reason,
                           completed_actions=len(done), added_to_reply=len(missing)),
    }
