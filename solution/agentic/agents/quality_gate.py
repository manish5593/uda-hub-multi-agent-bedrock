"""Quality gate: decides whether a draft answer is good enough to send or must be escalated."""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from agentic import config
from agentic.agents.common import add_trace, conversation_text
from agentic.llm import get_llm
from agentic.state import ResolutionAssessment
from agentic.tools.cultpass_tools import ESCALATING_TOOLS
from agentic.tools.knowledge_tools import format_articles

GATE_PROMPT = """You review draft replies written by an AI support agent for CultPass before they are sent.
Judge the draft against the evidence (the customer's account, knowledge base articles and tool results) only.
- confidence: how sure you are that the draft correctly and fully addresses the customer's request.
- grounded: false if the draft states facts, policies, steps or suggestions not supported by the evidence.
  Advice such as "check the venue's website" or "look at the event details" is NOT grounded unless an
  article says it.
- answered_by_evidence: true only if the evidence actually answers what the customer asked. Articles that are
  on a related topic but don't answer the question (e.g. an article about QR codes for a question about
  parking) mean false. A draft that can't answer and offers to pass the question on also means false.
- policy_requires_human: true only for refund approval, a blocked account, a legal or data-protection issue,
  or identity verification.
- customer_says_steps_failed: true only if the customer says they already tried the documented steps and they
  didn't work. A draft that gives the steps and offers more help *if they don't work* is false: the customer
  hasn't tried them yet.
- A claim about anyone other than this customer (another person's reservations or account) is NOT grounded:
  the tools only show this customer's account.
- awaiting_customer: true if the draft asks a clarifying question that is genuinely needed to proceed.
  A good clarifying question is a valid outcome and deserves high confidence. A draft that answers the
  request and then offers more help ("feel free to ask", "if you'd like, I can...") is NOT awaiting the
  customer: the request is done."""


def asks_a_question(draft: str) -> bool:
    """Whether the draft asks the customer anything. Questions end with '?' in the languages served, and
    Spanish opens them with '¿'; the full-width '？' covers CJK text."""
    return any(mark in draft for mark in ("?", "¿", "？"))


def gate_evidence(state: dict) -> list[str]:
    """What the draft may rely on: the articles retrieved, the tool calls and their results, previous tickets,
    the customer's stored preferences, and the customer's account as code loaded it from CultPass for this
    message (subscription, quota used and remaining, blocked status). The account agent is given the profile
    and the preferences too, so an answer taken from them is grounded even when no tool was called."""
    evidence = []
    customer = state.get("customer") or {}
    if customer.get("identified"):
        account = {k: customer.get(k) for k in ("name", "is_blocked", "subscription")}
        evidence.append("Customer account (loaded from CultPass for this message):\n"
                        + json.dumps(account, indent=1, default=str))
    if state.get("memories"):
        evidence.append("Known customer preferences (long-term memory):\n"
                        + "\n".join(f"- {m}" for m in state["memories"]))
    if state.get("knowledge"):
        evidence.append("Knowledge base:\n" + format_articles(state["knowledge"]))
    if state.get("actions_taken"):
        evidence.append("Tool calls and results:\n" + json.dumps(state["actions_taken"], indent=1, default=str))
    if state.get("prior_tickets"):
        evidence.append("Previous tickets:\n" + json.dumps(state["prior_tickets"], indent=1, default=str))
    return evidence


def assess_resolution(state: dict) -> dict:
    draft = state.get("draft_response")
    if not draft:
        if state.get("resolver") == "knowledge_agent":
            reason = "no relevant knowledge base article found"
        else:
            reason = "resolver could not produce an answer"
        assessment = {"confidence": 0.0, "grounded": False, "answered_by_evidence": False,
                      "needs_human": True, "awaiting_customer": False, "reason": reason}
        return {"assessment": assessment,
                "trace": add_trace(state, f"[quality_gate] {assessment['reason']}", event="assess", **assessment)}

    evidence = gate_evidence(state)
    result: ResolutionAssessment = get_llm().with_structured_output(ResolutionAssessment).invoke([
        SystemMessage(content=GATE_PROMPT),
        HumanMessage(content=f"Conversation:\n{conversation_text(state)}\n\n"
                             f"{chr(10).join(evidence) or 'No evidence.'}\n\nDraft reply:\n{draft}"),
    ])
    assessment = result.model_dump()
    # Code decides whether a human is needed, from the two facts the gate reports
    assessment["needs_human"] = assessment["policy_requires_human"] or assessment["customer_says_steps_failed"]
    if assessment["awaiting_customer"] and not asks_a_question(draft):
        # The model marks "let me know if..." offers as waiting for the customer, which left answered tickets
        # open. A reply can only wait for an answer if it asks something, so code decides.
        assessment["awaiting_customer"] = False
        assessment["reason"] += " (no question in the draft, so not awaiting the customer)"
    return {"assessment": assessment, "trace": add_trace(
        state, f"[quality_gate] confidence={assessment['confidence']:.2f} grounded={assessment['grounded']} "
               f"needs_human={assessment['needs_human']}: {assessment['reason']}",
        event="assess", **assessment)}


def should_escalate(state: dict) -> tuple[bool, str]:
    """Deterministic resolve-vs-escalate decision on top of the gate's assessment."""
    a = state.get("assessment") or {}
    used = {x["tool"] for x in state.get("actions_taken", [])
            if isinstance(x.get("result"), dict) and x["result"].get("success")}
    if used & ESCALATING_TOOLS:
        return True, f"policy requires human approval for {sorted(used & ESCALATING_TOOLS)}"
    if a.get("needs_human"):
        return True, a.get("reason", "human required")
    if not a.get("grounded", False):
        return True, "draft is not grounded in the knowledge base or account data"
    # A necessary clarifying question is a valid reply even though nothing is answered yet
    if not a.get("answered_by_evidence", False) and not a.get("awaiting_customer"):
        return True, "the knowledge base and account data don't answer this question"
    if a.get("confidence", 0) < config.RESOLUTION_MIN_CONFIDENCE:
        return True, f"resolution confidence {a.get('confidence', 0):.2f} below {config.RESOLUTION_MIN_CONFIDENCE}"
    return False, "draft approved"


def approve(state: dict) -> dict:
    return {"final_response": state["draft_response"],
            "trace": add_trace(state, "[quality_gate] approved draft", event="approve")}
