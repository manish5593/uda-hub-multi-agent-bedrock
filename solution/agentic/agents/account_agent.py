"""Account resolver: a ReAct agent that uses CultPass tools scoped to the current customer."""
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

from agentic.agents.common import add_trace, conversation, customer_context_block, text_of
from agentic.agents.quality_gate import asks_a_question
from agentic.llm import get_llm
from agentic.tools.cultpass_tools import build_account_tools
from agentic.tools.knowledge_tools import build_knowledge_tool

ACCOUNT_PROMPT = """You are a CultPass customer support agent with access to the customer's account tools.

Rules:
- Use the tools to check real account data before answering; never guess reservation ids, dates or quota.
- Earlier messages say what was done then, not what is true now. To answer whether a booking, cancellation or
  subscription change went through, or anything else about the account's current state, check it with a tool
  in this turn (get_my_reservations, get_my_account) and answer from the result.
- Never invent ids. Get reservation ids from get_my_reservations and experience ids from find_experiences
  BEFORE calling book_experience or cancel_my_reservation.
- For recommendations, call recommend_experiences. If the customer says in this conversation what they like,
  pass it as `interests` in their words; never pass interests they didn't state. It returns only experiences
  the customer can book that fit a stored or stated preference, each with the preference it fits. Recommend those (at most 3), saying which preference
  each fits. If it returns no matches, say that nothing coming up matches their known interests, then offer
  its other_options as alternatives, without saying they fit their taste. Describe each alternative by its facts
  only (what it is, where, when): no reason it would suit them, and nothing about them the tools didn't say. Use find_experiences when the customer
  asks for something specific.
- Look up policies with search_knowledge_base before explaining them (cancellation, refunds, quota...).
- Each find_experiences result says whether the customer can_book it, and if not, why
  (cannot_book_because). Never offer or attempt a booking the result says they can't make; explain why.
- Never say you will do something later ("I will book it"): call the tool now and report its result, or ask
  the question you need answered first.
- To change a reservation's date (or move it to another experience): first find the other dates with
  find_experiences (each date is a separate experience with the same title; each result lists its
  other_dates_with_free_slots, and runs_only_on_this_date). If there is no other date the
  customer can book, say so and keep their reservation. Otherwise use change_reservation_date, which books the new
  one and cancels the old one together. Never cancel a reservation as part of a change.
- Only perform changes (book, cancel, change subscription) that the customer clearly asked for.
  If the request is ambiguous (e.g. which reservation), ask a short clarifying question instead.
- Refunds: never promise one. Use request_refund to register it for support-lead approval and tell
  the customer the billing team will review it.
- The tools only show the account of the customer you are talking to. Never say anything about another
  person's reservations, subscription or account; if asked, say you can't see or share other customers'
  details, and help with this customer's own account instead.
- If a tool returns an error, explain it plainly and offer the next best option.
- Describe experiences in plain words (what it is, where, when, and why it fits); don't list tool fields
  such as can_book, slots or fits_preference.
- Reply in the customer's language ({language}), be warm and concise, and confirm what you did.

Customer context:
{context}

Ticket summary: {summary}"""


def _actions(messages: list) -> list[dict]:
    """Extract the tool calls (and their results) the agent made, for auditing and escalation."""
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    actions = []
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                raw = results.get(call["id"], "")
                try:
                    result = json.loads(raw)
                except (TypeError, ValueError):
                    result = raw
                actions.append({"tool": call["name"], "args": call["args"], "result": result})
    return actions


# Reservation and experience ids are 6 hex characters (uuid4 prefixes). A token counts as an id only if it
# mixes digits and letters a-f, so words like "cafe" and years like "2026" never do.
ID_LIKE = re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{4,8}\b")


def _ids_in(value) -> set[str]:
    """Every reservation or experience id in a tool call's arguments or result."""
    if isinstance(value, dict):
        found = {str(v) for k, v in value.items() if k in ("reservation_id", "experience_id") and v}
        return found.union(*(_ids_in(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_ids_in(v) for v in value))
    return set()


def correct_ids(draft: str, actions: list[dict]) -> tuple[str, list[str]]:
    """Ids are copied by the model, which once wrote 23e89 for a 6-character reservation id. Every id-like
    token in the draft must be one the tools returned. A wrong one is replaced when the right one is
    unambiguous (the turn made exactly one booking); the replacements are returned for the trace."""
    known = set().union(*(_ids_in(a.get("args")) | _ids_in(a.get("result")) for a in actions)) if actions else set()
    booked = [a["result"]["reservation_id"] for a in actions
              if a["tool"] == "book_experience" and isinstance(a.get("result"), dict) and a["result"].get("success")]
    unknown = [t for t in dict.fromkeys(ID_LIKE.findall(draft)) if t not in known]
    if not unknown or len(booked) != 1:
        return draft, []
    for token in unknown:
        draft = re.sub(rf"\b{token}\b", booked[0], draft)
    return draft, [f"{token} -> {booked[0]}" for token in unknown]


# Sent to the agent (never to the customer) when its reply used no tool and asks nothing
CHECK_WITH_TOOLS = ("(Support system note, not from the customer: your reply used no tool. Answers about the account, "
                    "reservations or experiences must come from this turn's tool results. Check with your tools now, then "
                    "reply to the customer's last message.)")


def resolve_with_account_tools(state: dict) -> dict:
    customer = state["customer"]
    classification = state.get("classification", {})
    tools = build_account_tools(customer["external_user_id"], state["ticket_id"], state.get("memories"))
    tools.append(build_knowledge_tool(customer["account_id"]))

    agent = create_agent(
        model=get_llm(),
        tools=tools,
        system_prompt=SystemMessage(content=ACCOUNT_PROMPT.format(
            language=classification.get("language", "English"),
            context=customer_context_block(state),
            summary=classification.get("summary", ""),
        )),
    )
    try:
        result = agent.invoke({"messages": conversation(state)}, {"recursion_limit": 15})
    except GraphRecursionError:
        # The agent looped without converging: hand over to the quality gate with no draft
        return {
            "draft_response": None,
            "resolver": "account_agent",
            "trace": add_trace(state, "[account_agent] gave up after too many steps", event="tools", tools=[],
                               error="recursion_limit"),
        }
    actions = _actions(result["messages"])
    first_draft = text_of(result["messages"][-1])
    if not actions and not asks_a_question(first_draft):
        # The agent answered from memory, e.g. from earlier messages: the quality gate then saw no evidence and
        # escalated a correct reply, or the agent announced a lookup instead of making it. One retry, with tools.
        try:
            retry = agent.invoke({"messages": [*conversation(state), HumanMessage(content=CHECK_WITH_TOOLS)]},
                                 {"recursion_limit": 15})
            retry_actions = _actions(retry["messages"])
            if retry_actions:
                result, actions = retry, retry_actions
            state = {**state, "trace": add_trace(
                state, f"[account_agent] reply used no tool; retried with tools: {[a['tool'] for a in retry_actions]}",
                event="retry_with_tools", tools=[a["tool"] for a in retry_actions])}
        except GraphRecursionError:
            pass  # keep the first draft
    draft, fixed = correct_ids(text_of(result["messages"][-1]), actions)
    if fixed:
        state = {**state, "trace": add_trace(state, f"[account_agent] corrected ids in the draft: {fixed}",
                                             event="correct_ids", fixed=fixed)}
    return {
        "draft_response": draft,
        "actions_taken": actions,
        "resolver": "account_agent",
        "trace": add_trace(state, f"[account_agent] tools used: {[a['tool'] for a in actions]}", event="tools",
                           tools=[{"tool": a["tool"], "args": a["args"],
                                   "success": a["result"].get("success") if isinstance(a["result"], dict) else True}
                                  for a in actions]),
    }
