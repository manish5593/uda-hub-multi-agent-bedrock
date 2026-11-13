# Agents

Each module provides one or more LangGraph nodes, which `agentic/workflow.py` wires into the graph. A node reads the
shared `UdaHubState` and returns the fields it changes. Every node also adds a line to `state["trace"]`, so each
decision can be audited. The full flow and routing rules are in [`../../../design`](../../../design/README.md), and its
[Safeguards](../../../design/README.md#safeguards-the-llm-proposes-code-decides) section lists which decisions are enforced in code.

| Module | Node(s) | Uses the LLM | What it does |
|---|---|---|---|
| `memory_agent.py` | `load_context` | No | Runs first on every turn. It resolves the ticket from `thread_id`, creating it if needed, and identifies the customer from metadata. It stores the inbound message, recalls long-term memory (preferences and previous tickets), and resets the per-turn fields. |
| `classifier.py` | `classifier` | Yes, structured output | Produces a `TicketClassification`: issue type (11 categories), urgency, sentiment, language, whether account action is needed, whether a human was requested, any email mentioned, and confidence. Recommendations count as account work, because they need the real catalog. |
| `supervisor.py` | `supervisor`, `request_info` | No | When a guest types an email, sends a 6-digit code to it (`agentic/verification.py`) and links the ticket to that customer only once the code is typed back; a message that is just the code reuses the request's classification. It then routes with `route_ticket()`, a pure function of the classification, customer and ticket status. It escalates before `knowledge_agent` or `account_agent` runs when the customer asks for a human, the issue is legal, the urgency is critical (classified, or reported in the ticket metadata), the classifier isn't confident (< 0.5), a blocked customer has an account issue, or the customer reports a problem they reported before (`repeat_issue_types()`). A guest asking how reservations, the subscription or billing work goes to `knowledge_agent` without being asked who they are. `request_info` asks a guest for their email when an account action is needed, then for the code sent to it, in their language (only non-English messages need an LLM call, to translate). |
| `knowledge_agent.py` | `knowledge_agent` | Yes | Retrieves the top 3 relevant knowledge base articles (cosine similarity of at least 0.30) and answers using only those, in the customer's language. With no relevant article, it produces no draft, and the ticket is escalated. |
| `account_agent.py` | `account_agent` | Yes, ReAct agent | A `create_agent` (LangChain) with the CultPass tools bound to the current customer, plus knowledge base search. The prompt has it look ids up before acting, recommend with `recommend_experiences` (only bookable experiences that fit the customer's stored preferences), never offer a booking a search result says the customer can't make, ask a clarifying question when the request is ambiguous, and never promise a refund. The tools enforce the ids part: booking and cancellation refuse ids that no lookup returned in the same turn, so a guessed id fails and the agent looks it up. Every reservation or experience id in the draft must be one the tools returned; a miscopied one is corrected when the turn made exactly one booking (`correct_ids()`). It records every tool call in `actions_taken`, and gives up with no draft (so the ticket escalates) if it hasn't finished after 15 steps. |
| `quality_gate.py` | `quality_gate`, `approve` | Yes, structured output | Scores the draft against the evidence (the customer's account and stored preferences, articles, tool results, and what the account agent can do) for confidence, grounding, whether the evidence actually answers the question, and whether it waits on the customer. Instead of asking whether a human is needed, it asks two facts, whether policy requires one and whether the customer says the steps already failed, and code sets `needs_human` from them. A draft with no question mark can't be waiting on the customer (`asks_a_question()`). `should_escalate()` then decides in plain code. It escalates if a refund was requested, a human is needed, the draft isn't supported by the evidence, the evidence doesn't answer the question, or confidence is below 0.7. |
| `escalation_agent.py` | `escalation_agent` | Yes, structured output | Picks the team (Trust & Safety, billing, technical support, privacy/legal or general support) and the priority. It saves a handoff note on the ticket and writes an empathetic reply. It never reveals why an account is blocked and never promises an outcome. If the turn already booked, cancelled or changed something, the reply says so: code checks it and adds anything left out. |
| `memory_agent.py` | `finalize`, `write_memory` | `write_memory` only | `finalize` sends the reply, logs it and sets the ticket status (`resolved`, `awaiting_customer` or `escalated`). `write_memory` extracts durable facts about an identified customer, as described below. |
| `common.py` | | No | Shared helpers for traces, the session's conversation, and the customer-context block used in prompts. |

## Long-term memory rules (`write_memory`)

The LLM proposes memories, and code then decides what is stored:

- **Input:** the memory writer sees only the customer's own messages (`customer_messages_text()`). The assistant's replies contain internal details, such as team names and block status, which must never become memories.
- **Wording:** the prompt asks for third-person summaries of what the customer said, such as "Reported <problem>". It rules out quotes, inferences, and restating a known memory in other words.
- **Filter:** `drop_reason()` rejects:
  - memories about reservations, bookings, cancellations or quota, or that name a specific catalog experience, because the system can look these up live and they go stale. General preferences are exempt from the keyword check.
  - one-off requests, such as asking to talk to a real person.
  - contact details: email addresses and phone numbers, which the account already holds.
- **Duplicates:** `save_memory()` in `../tools/ticket_tools.py` ignores case, punctuation and spacing when it checks for an existing memory.
- **Trace:** the trace lists what was stored and what was dropped, and why.

## Adding an agent

1. Write a node function that takes the state and returns only the fields it changes. Include
   `"trace": add_trace(state, "[my_agent] ...", event="my_event", **fields)`. That records the decision in
   the trace and as a structured event in the searchable log.
2. Add any new fields to `UdaHubState` in `../state.py`, and reset per-turn fields in `load_context`.
3. Register the node and its edges in `build_graph()` in `../workflow.py`. For a new route, add it to
   `route_ticket()` and to the supervisor's conditional edges.
4. Add tests: unit tests for any rule in plain code, and a path through the graph in `tests/test_workflow.py`
   using the scripted `FakeLLM`.
