# Tools

Each module provides plain Python functions, which hold the business rules and are unit-tested without an LLM.
The tools the agents call are thin LangChain wrappers around them. Because the rules live in the tools, they hold
whatever the model decides. The [Safeguards](../design/README.md#safeguards-the-llm-proposes-code-decides) section
of the design README lists them.

## `cultpass_tools.py`: CultPass account tools

`build_account_tools(user_id, ticket_id)` returns the account agent's tools with the customer's id bound in, so the
LLM can never act on another customer's account. The tools are rebuilt on every turn.

| Tool | What it does | Rules enforced |
|---|---|---|
| `get_my_account` | Profile, blocked status, subscription, and quota used and remaining this month | |
| `get_my_reservations` | The customer's reservations: id, status, experience and date | |
| `find_experiences` | Upcoming experiences with free slots, by keyword and location. Each result says whether the customer has `already_booked` it, so they aren't recommended something they already have. | |
| `book_experience` | Reserves an experience | The id must have come from `find_experiences` or `get_my_reservations` in this turn. The account must not be blocked and the subscription must be active. Premium experiences need the premium tier. There must be free slots and quota left. No double booking. |
| `cancel_my_reservation` | Cancels a reservation and returns the credit if it's at least 24h before the event | The id must have come from `get_my_reservations` in this turn, or from a booking made in this turn. The reservation must belong to the customer and still be active. |
| `change_subscription` | Pauses, resumes or cancels the subscription | Only valid state changes are allowed. Blocked accounts can't make changes. |
| `request_refund` | Registers a refund request as a system message on the ticket and returns a reference such as `RF-1A2B3C` | It doesn't issue a refund. Its use always escalates the ticket (`ESCALATING_TOOLS`), because refunds need support-lead approval. |

An id that wasn't looked up gets an error telling the agent which lookup to call. If the agent has already done
the lookup, the error lists the valid ids with their labels, because the usual mistake is passing a list position
such as `"0"` instead of the id. The error is returned rather than raised, so the agent retries with the right id.

### Error handling and validation

Every tool returns structured data, and every failure comes back as `{"success": false, "error": "..."}`:

| Kind of failure | How it's handled |
|---|---|
| Invalid arguments, such as an unknown subscription action or a missing id | Rejected by the tool's typed schema before it runs. LangGraph returns the validation message to the agent. |
| Business rules: blocked account, full event, quota used, double booking, ownership, invalid state change | Checked by the underlying function and returned as an error |
| Unusable input: an id that wasn't looked up this turn, a refund request with no reason | Checked by the tool wrapper and returned as an error |
| Unexpected faults, such as the database being unavailable | Caught by `_safe()`, logged, and returned as an error. LangGraph's `ToolNode` only catches invalid-argument errors, so without this one fault would crash the whole run and leave the customer without a reply. |

Notebook 03's "Support tools on sample operations" cell runs each tool on sample operations, including these failures.

The underlying functions (`get_customer_profile`, `find_customer_by_email`, `list_reservations`,
`search_experiences`, `reserve_experience`, `cancel_reservation`, `manage_subscription`) are also used directly
by the supervisor and the memory nodes.

## `knowledge_tools.py`: RAG over the knowledge base

- **Retriever:** `KnowledgeRetriever` indexes the account's `knowledge` articles in an `InMemoryVectorStore` with the configured embedding model: OpenAI `text-embedding-3-small` by default, Amazon Bedrock Titan, or a SageMaker endpoint (see `agentic/config.py`). The knowledge service in `service/` uses a subclass that keeps the embeddings in Postgres with pgvector instead (`service/knowledge_service/pgvector_index.py`); the search and its thresholds are the same.
- **Search:** `search()` returns the top 3 articles scoring at least 0.30 cosine similarity. The threshold can be changed with the `UDAHUB_RAG_MIN_SCORE` environment variable.
- **Fallback:** if embeddings are unavailable, for example offline or without `numpy`, it falls back to IDF keyword scoring with light stemming and a 0.10 threshold. A warning is logged when this happens. The fallback also lets the tests run offline.
- **Caching:** `get_retriever()` builds each account's index once.
- **Tool:** `search_knowledge_base` gives the account agent access to the same retriever.

## `ticket_tools.py`: tickets and long-term memory in `udahub.db`

| Function | What it does |
|---|---|
| `get_or_create_ticket` | Loads a ticket, or creates it with its user. Unknown customers get a per-account `guest` user. |
| `link_ticket_to_customer` | Moves a guest ticket to the identified customer |
| `add_message`, `get_ticket_messages` | The ticket's message log: user, AI and system messages, including handoff notes and refund requests |
| `update_ticket_metadata` | Status, main issue type, and tags (merged without duplicates) |
| `get_prior_tickets` | Summaries of the customer's previous tickets, recalled in new tickets |
| `get_memories`, `save_memory` | The `customer_memory` table. `save_memory` skips duplicates, ignoring case, punctuation and spacing. |

## `db.py`

Cached SQLAlchemy engines and a session context manager for `udahub.db` and `cultpass.db`. The paths come from
`agentic/config.py` and can be overridden with the `UDAHUB_DB_PATH` and `CULTPASS_DB_PATH` environment variables.

## Adding a tool

1. Write the logic as a plain function that takes ids and returns a dict, with `success` and an `error`
   message on failure. Returning errors instead of raising lets the agent explain the problem or recover.
2. Expose it to the account agent inside `build_account_tools()`. Take the customer's id from the closure,
   not from the LLM, and give it a docstring: that's the description the model sees.
3. If it should accept only ids the agent looked up, check them against `seen_reservation_ids` or
   `seen_experience_ids`. If a successful call should always reach a human, add its name to `ESCALATING_TOOLS`.
4. Add unit tests for its rules to `tests/test_tools.py`.
