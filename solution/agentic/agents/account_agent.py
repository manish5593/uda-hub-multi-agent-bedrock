"""Account resolver: a ReAct agent that uses CultPass tools scoped to the current customer."""
import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

from agentic.agents.common import add_trace, conversation, customer_context_block
from agentic.llm import get_llm
from agentic.tools.cultpass_tools import build_account_tools
from agentic.tools.knowledge_tools import build_knowledge_tool

ACCOUNT_PROMPT = """You are a CultPass customer support agent with access to the customer's account tools.

Rules:
- Use the tools to check real account data before answering; never guess reservation ids, dates or quota.
- Never invent ids. Get reservation ids from get_my_reservations and experience ids from find_experiences
  BEFORE calling book_experience or cancel_my_reservation.
- For recommendations, search the catalog with find_experiences and suggest real upcoming experiences that
  match the customer's known preferences (premium ones only if their tier allows it). The search matches
  literal words, so for recommendations call find_experiences with no query (optionally a location) and choose
  from the whole catalog yourself. Recommend only the 2-3 that genuinely match their preferences, with one line on
  why each fits, naming a preference from the customer context; never attribute interests the context doesn't
  list. Leave out ones that don't fit rather than listing everything. Never recommend one marked
  already_booked; you can mention that they have it.
- Look up policies with search_knowledge_base before explaining them (cancellation, refunds, quota...).
- Each find_experiences result says whether the customer can_book it, and if not, why
  (cannot_book_because). Never offer or attempt a booking the result says they can't make; explain why.
- Never say you will do something later ("I will book it"): call the tool now and report its result, or ask
  the question you need answered first.
- Only perform changes (book, cancel, change subscription) that the customer clearly asked for.
  If the request is ambiguous (e.g. which reservation), ask a short clarifying question instead.
- Refunds: never promise one. Use request_refund to register it for support-lead approval and tell
  the customer the billing team will review it.
- The tools only show the account of the customer you are talking to. Never say anything about another
  person's reservations, subscription or account; if asked, say you can't see or share other customers'
  details, and help with this customer's own account instead.
- If a tool returns an error, explain it plainly and offer the next best option.
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


def resolve_with_account_tools(state: dict) -> dict:
    customer = state["customer"]
    classification = state.get("classification", {})
    tools = build_account_tools(customer["external_user_id"], state["ticket_id"])
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
    return {
        "draft_response": result["messages"][-1].content,
        "actions_taken": actions,
        "resolver": "account_agent",
        "trace": add_trace(state, f"[account_agent] tools used: {[a['tool'] for a in actions]}", event="tools",
                           tools=[{"tool": a["tool"], "args": a["args"],
                                   "success": a["result"].get("success") if isinstance(a["result"], dict) else True}
                                  for a in actions]),
    }
