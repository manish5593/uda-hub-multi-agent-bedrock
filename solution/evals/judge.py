"""LLM judge for what code can't check: is the reply supported by the evidence, and does it make the points
the case expects.

The judge sees only the customer's messages, the reply, and the evidence the agents had (the knowledge base
articles they retrieved, the tool calls and results, the customer's profile and stored preferences). It uses the configured
chat model unless UDAHUB_EVAL_JUDGE_MODEL names another one, from the same provider or from the one in
UDAHUB_EVAL_JUDGE_PROVIDER ("openai" or "bedrock"). A judge from a stronger model is more reliable, avoids a model
grading its own replies, and keeps scores comparable when the agents' provider changes.
"""
import json
import os
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic import config
from agentic.tracing import call_config
from agentic.tools.knowledge_tools import format_articles

FAITHFULNESS_PROMPT = """You check replies written by an AI customer support agent for CultPass.
List every factual claim in the reply that the evidence does not support: policies, prices, steps, dates,
numbers, account details, or actions the reply says were taken. The evidence is the only source of truth.
Do not count greetings, empathy, offers to help further, questions to the customer, or hedged suggestions
such as "you might enjoy" as claims.
A claim the evidence contradicts is unsupported.
Everything in a knowledge base article is evidence, its "Suggested phrasing" included: the agent is told to use it.
In tool results, "already_booked": true on an experience means this customer has booked it, so a reply saying
they already booked it (with that experience's date) is supported.
"remaining_this_month" in a booking, cancellation or change result is how many experiences the customer has left
this month after that action: a reply giving that number, or saying it stayed the same when the profile showed the
same number before, is supported. A different number is not.
"dates_with_free_slots" on an experience lists every date that experience runs with free slots, and
"other_dates_with_free_slots" the others. When a tool result has "runs_only_on_this_date": true, the experience runs
on no other date: "there are no other dates", "it isn't available on <any other date>" and "the only date is <its
date>" are all supported by that result. Don't list such a claim as unsupported."""

KEY_POINTS_PROMPT = """You check replies written by an AI customer support agent for CultPass.
For each expected point, decide whether the reply makes it. Judge the meaning, not the wording; the reply
may be in any language. A point that says the reply must NOT do something is met when the reply doesn't
do it."""


class Faithfulness(BaseModel):
    unsupported_claims: list[str] = Field(description="Claims in the reply the evidence doesn't support; empty if none")


class PointVerdict(BaseModel):
    point: str
    met: bool
    reason: str = Field(description="One short sentence")


class KeyPoints(BaseModel):
    verdicts: list[PointVerdict] = Field(description="One verdict per expected point, in the same order")


def judge_provider() -> str:
    return (os.getenv("UDAHUB_EVAL_JUDGE_PROVIDER") or config.LLM_PROVIDER).lower()


@lru_cache(maxsize=None)
def judge_llm() -> BaseChatModel:
    model = os.getenv("UDAHUB_EVAL_JUDGE_MODEL") or config.CHAT_MODEL
    if judge_provider() == "bedrock":
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(model=model, region_name=config.AWS_REGION, temperature=0, max_tokens=2048)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=0)


def judge_model_name() -> str:
    return os.getenv("UDAHUB_EVAL_JUDGE_MODEL") or config.CHAT_MODEL


def evidence_text(state: dict) -> str:
    parts = []
    customer = state.get("customer") or {}
    if customer.get("identified"):
        parts.append("Customer profile:\n" + json.dumps(
            {k: customer.get(k) for k in ("name", "email", "is_blocked", "subscription")}, default=str))
    if state.get("memories"):
        parts.append("Known customer preferences (long-term memory):\n"
                     + "\n".join(f"- {m}" for m in state["memories"]))
    if state.get("knowledge"):
        parts.append("Knowledge base articles:\n" + format_articles(state["knowledge"]))
    if state.get("actions_taken"):
        parts.append("Tool calls and results:\n" + json.dumps(state["actions_taken"], indent=1, default=str))
    return "\n\n".join(parts) or "No evidence."


def _trace(check: str, session_id: str | None) -> dict:
    """Each judge call is a Langfuse trace (when tracing is on) in the session of the ticket it judges."""
    return call_config(f"udahub-eval-judge-{check}", session_id, ["eval-judge", judge_model_name()])


def check_faithfulness(customer_messages: list[str], reply: str, state: dict, llm=None,
                       session_id: str | None = None) -> Faithfulness:
    llm = llm or judge_llm()
    return llm.with_structured_output(Faithfulness).invoke([
        SystemMessage(content=FAITHFULNESS_PROMPT),
        HumanMessage(content="Customer messages:\n" + "\n".join(customer_messages)
                             + f"\n\nEvidence:\n{evidence_text(state)}\n\nReply:\n{reply}"),
    ], config=_trace("faithfulness", session_id or state.get("ticket_id")))


def check_key_points(customer_messages: list[str], reply: str, points: list[str], llm=None,
                     session_id: str | None = None) -> list[PointVerdict]:
    llm = llm or judge_llm()
    result: KeyPoints = llm.with_structured_output(KeyPoints).invoke([
        SystemMessage(content=KEY_POINTS_PROMPT),
        HumanMessage(content="Customer messages:\n" + "\n".join(customer_messages) + f"\n\nReply:\n{reply}\n\n"
                             "Expected points:\n" + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(points))),
    ], config=_trace("key-points", session_id))
    verdicts = list(result.verdicts)
    if len(verdicts) != len(points):
        raise ValueError(f"judge returned {len(verdicts)} verdicts for {len(points)} points")
    # Keep the case's own wording of each point, whatever the judge echoed back
    return [PointVerdict(point=p, met=v.met, reason=v.reason) for p, v in zip(points, verdicts)]
