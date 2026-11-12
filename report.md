# Project Report: UDA-Hub Multi-Agent Customer Support

## 1. Summary

UDA-Hub resolves customer support tickets for CultPass, a subscription service for cultural experiences. A ticket
arrives as a natural-language message with optional metadata (customer, channel, urgency), from a channel
integration or, as microservices, from a customer who logged in. The system classifies
it, routes it to a specialist agent, and either resolves it or escalates it to the right human team. It
answers from a knowledge base, acts on the customer's account through tools, and remembers the customer across
sessions.

| | |
|---|---|
| **Architecture** | Supervisor pattern in LangGraph: 7 agents in one `StateGraph` (11 nodes, 16 edges) |
| **Models** | `gpt-4o-mini` for every agent and `text-embedding-3-small` for retrieval (the demo run, the full stack and the evaluation), with `gpt-4o` as the evaluation's judge. Claude on Amazon Bedrock (Claude Haiku 4.5 by default), Titan embeddings and a SageMaker embedding endpoint are implemented as alternatives; Bedrock answered test calls, but a full run waits on AWS account approval (section 4) |
| **Data** | Two SQLite databases: `cultpass.db` (customers, subscriptions, experiences, reservations) and `udahub.db` (tickets, messages, 18 knowledge articles, long-term memory). As microservices, the same tables in Postgres, one database per service, with the knowledge embeddings stored in pgvector |
| **Deployment** | `service/`: an API gateway and five services (agent, CultPass, tickets, knowledge, auth) with Postgres and pgvector, Alembic migrations, Docker Compose, customer sign-up and login with JWTs, Prometheus with a Grafana dashboard, optional Langfuse tracing, and a web chat for demos (section 4). Run end to end with the real model |
| **Demonstration** | `solution/03_agentic_app.ipynb`, committed with the outputs of a real-model run: 13 scenarios, 5 edge cases, and demos of routing, retrieval, tools, memory and logs |
| **Tests** | 85 offline tests in `solution/tests` (35 unit, 20 end-to-end through the graph with a scripted model, 26 for the evaluation suite, 4 for tracing) and 108 in `service/tests` (services, customer logins and tokens, the full HTTP flow, migrations on SQLite and Postgres, and the pgvector index on Postgres). No API key needed; GitHub Actions runs them on every pull request |
| **Evaluation** | `solution/evals`: 36 labelled tickets, 3 runs each, against the real model, scored in code and by an LLM judge. **108 of 108 runs (100%) passed every check** in the latest evaluation, up from 95 (88%) at the first one; the runs in between scored 94–99%, so expect a run to miss one or two (section 5) |
| **Environment** | Python 3.11, LangGraph 1.2.20, LangChain 1.4.9, langchain-openai 1.7.4, langchain-aws 1.9.2, SQLAlchemy 2.1, FastAPI 0.162, Postgres 16 with pgvector 0.8.7, PyJWT 2.15, bcrypt 5.0, Langfuse 4.78 (SDK 5.2) |

The main lesson matched what I'd learned building earlier agent systems: **the LLM proposes, code decides.**
Almost every problem found by running the system against the real model was a decision left to the model that
it made inconsistently. Each was fixed by moving that decision into code, which can be tested (section 6).

## 2. Architecture

![UDA-Hub architecture](design/architecture.svg)

One graph run handles one customer message:

| Node | Uses the LLM | Role |
|---|---|---|
| `load_context` | No | Finds or creates the ticket, identifies the customer (from the metadata, or as microservices from a verified login token), and recalls long-term memory: preferences and previous tickets |
| `classifier` | Yes, structured output | Issue type (11 categories), urgency, sentiment, language, whether account action or a human is needed, confidence |
| `supervisor` | No | Identifies a guest from an email in their message, then routes with plain-code rules |
| `knowledge_agent` | Yes | RAG over the knowledge base, answering only from the retrieved articles |
| `account_agent` | Yes, ReAct agent (`create_agent`) | CultPass tools bound to the current customer |
| `quality_gate` | Yes, structured output, plus rules | Checks the draft's grounding and confidence against the evidence, then decides in code whether to send it or escalate |
| `escalation_agent` | Yes, structured output | Picks the team and priority, writes a handoff note, and replies to the customer |
| `request_info` | Only to translate | Asks a guest for their email, or asks what they need if the message was empty |
| `approve`, `finalize` | No | Send the reply, save it, and set the ticket's status |
| `write_memory` | Yes, plus filters | Extracts lasting facts from the customer's own messages |

**Why the Supervisor pattern:** support needs one place where policies are applied, every time. For example,
blocked accounts always go to Trust & Safety. The supervisor's routing is plain code rather than an LLM, so
every routing decision is predictable, testable and explained in the log. That's also why the project builds its own
`StateGraph` rather than using `langgraph-supervisor`, whose supervisor is an LLM. The full design is in
[`design`](design/README.md).

## 3. Implementation

### Routing

`route_ticket()` applies 10 ordered rules to the classification, the customer, the ticket status, the customer's
previous issues, and the urgency reported in the metadata. Escalation comes first for:
- human requests;
- legal or safety issues;
- critical urgency, whichever of the classified or reported urgency is more severe;
- low classifier confidence;
- blocked accounts;
- **repeat problems**: the customer already reported this issue type.

Everything else goes to the account agent if it needs account data, asks a guest to identify themselves, or goes
to the knowledge agent.

### Knowledge retrieval

Articles are embedded into a vector store (in memory in the notebooks; in Postgres with pgvector in the
microservices, so only new or changed articles are embedded), and retrieval keeps the top 3 with a cosine similarity of
at least 0.30. If none qualifies, no answer is drafted and the ticket escalates. If articles are found, the quality gate
checks both that the answer is **grounded** in them and that they **actually answer the question**. That catches
weakly related articles (section 6). If embeddings are unavailable, retrieval falls back to keyword scoring.

### Tools

There are 8 tools: account lookup, reservations, experience search, booking, cancellation, subscription changes,
refund requests, and knowledge search. They wrap the database through SQLAlchemy and return structured results.
- **Customer-scoped:** the customer's id is bound into each tool, so the LLM can't act on another account.
- **Looked-up ids only:** booking and cancellation accept only ids that a lookup returned in the same turn.
- **Errors as data:** business-rule failures come back as `{"success": false, "error": ...}`, and so do
  unexpected faults. LangGraph's `ToolNode` only catches invalid-argument errors on its own.
- **Refunds:** a refund is never issued. It is registered for support-lead approval, and the ticket always
  escalates.

### Memory

| Kind | Where | Used for |
|---|---|---|
| State | `UdaHubState` | Passing the classification, route, evidence and assessment between nodes within a run |
| Short-term | Checkpointer keyed by `thread_id` (= ticket id): `MemorySaver` in the notebooks, `PostgresSaver` in the agent service | Follow-ups such as "cancel *that one*"; inspecting a ticket's messages, tool usage and node order |
| Long-term | `udahub.db` | Preferences and reported issues (`customer_memory`), previous tickets with their replies, and the full message log. It's recalled on every new ticket, and after a restart it rebuilds the conversation. |

### Logging and failure handling

Every decision is written to a structured event log (`logs/events.jsonl`, one JSON object per line). Each event
carries the ticket, the agent, the kind of event, and its details: route and reason, articles and scores, tools
with their arguments and success, the assessment, and the escalation team. `search_events()` filters by any
field. `process_ticket()` catches failures such as an LLM outage or an unknown account. It logs them, flags the
ticket for a human, and sends a fallback reply instead of crashing.

## 4. Deployment and evaluation

### Microservices

`service/` runs the same graph as five services behind an API gateway. Each data function in `agentic/tools` runs
locally against its database, or over HTTP against the service that owns the data, depending on configuration.
So the agents, prompts, tools and rules are unchanged, and each business rule is still written once.

| Part | What it adds |
|---|---|
| `gateway` | The only public endpoint: API keys for channel integrations, JWT access tokens for logged-in customers (who only see their own tickets), a rate limit per caller, request ids, and replies without internal traces. It also serves a web chat for demos (`/demo`) |
| `agent` | Runs the workflow. Conversations are kept in Postgres (`PostgresSaver`), so they survive restarts and any replica can continue any ticket; an advisory lock stops two messages on one ticket from running at once |
| `cultpass`, `tickets`, `knowledge` | Own their data, each in its own Postgres database and role. The knowledge service keeps the article embeddings there too, with pgvector |
| `auth` | Customer sign-up, verified by a code sent to the CultPass email, and login. Stores bcrypt password hashes, locks a login after repeated failures, and issues the access tokens the gateway verifies |
| `db-setup`, `migrate`, `seed` jobs | Any missing database and role and the pgvector extension, Alembic migrations, one history per database, then the demo data. Databases created before the migrations are adopted, not recreated |
| Monitoring | Request metrics on every service, plus what the agent did with each message (outcome, route, escalation team and priority, tools). Prometheus collects them, and a provisioned Grafana dashboard shows support outcomes, traffic, errors and latency |
| Tracing (optional) | Langfuse, self-hosted: every graph run, node, model call and tool call, grouped by ticket |

![UDA-Hub deployment as microservices](design/deployment.svg)

### Customer authentication

Originally every caller was a trusted channel integration with an API key, and the customer was whoever the
message metadata said, so anyone with a key could act as any customer. Customers can now log in themselves, and
for them identity is something they proved, not something they typed:

- **Verified sign-up.** Only CultPass customers can sign up, and only with a 6-digit code sent to their CultPass
  email. Unverified sign-up would let anyone claim an account that has no login yet. The code expires after 10
  minutes, works once, is burnt after 5 wrong guesses, and is stored only as an HMAC. Sign-up gives the same
  answer whether or not the email is a customer's, so it can't be used to find accounts.
- **Passwords** are bcrypt hashes. Five failures lock the login for 15 minutes. A wrong password, an unknown email
  and a locked login get the same `401`, and an unknown email is still checked against a dummy hash so the
  response time doesn't give it away.
- **The token decides who the customer is.** The auth service issues a short-lived JWT (HS256, 60 minutes, with
  issuer, audience and type). The gateway verifies it locally, without a call per request, sets the customer from
  it, and drops identity fields from the metadata. The agent then runs exactly as for a channel, so none of the
  agents, tools or rules changed.
- **Customers only see their own tickets.** Another customer's ticket, or a guest ticket a channel started, is a
  `404` like a missing one, so ticket ids can't be probed.
- **Channels keep their API keys** and still see every ticket. Sign-up and login are rate-limited per client IP,
  and each customer has their own request budget.

### Tracing

Prometheus answers "how many and how fast"; it can't show what the model saw. Optional Langfuse tracing records
every graph run as one trace: each node, every model call with its prompt, response and token counts, and every
tool call, grouped into a session per ticket. It is self-hosted in the same Docker Compose file (its own Postgres,
ClickHouse, Redis and MinIO, under a `langfuse` profile), and it is off unless its keys are set, so the
notebooks, the tests and the plain stack are unchanged. The user shown is the CultPass id, never the email.

### Embeddings in Postgres (pgvector)

In the notebooks the knowledge index is built in memory on every start. As microservices that meant every
restart embedded all 18 articles again, and every replica of the knowledge service held its own copy. The
knowledge service now keeps the embeddings in its own database with pgvector:

- **Only new or changed articles are embedded.** Each row stores a hash of the embedded text, so a restart embeds
  nothing and an edited article is embedded again on its own.
- **One shared index** for every replica.
- **Each embedding model keeps its own vectors**, so switching from OpenAI to Titan never compares vectors from
  different models.
- **Same scores:** pgvector's cosine distance gives the same similarity as the in-memory store, so the 0.30
  threshold and every retrieval rule are unchanged.

The table is a derived index that can be rebuilt from the articles at any time, so the service creates it
itself, outside the Alembic migrations, like the agent's checkpoint tables. The image is `postgres:16-alpine`
with pgvector compiled in, so the existing data volume kept working. Without pgvector the service logs a
warning and keeps the index in memory, and the notebooks still use the in-memory index.

### Running it

The whole stack has been run end to end with the real model (`gpt-4o-mini`): 15 long-running containers
(gateway, five services, Postgres with pgvector, Prometheus, Grafana, and Langfuse with its five stores and worker), with migrations and seeding,
then the 13 notebook scenarios (16 turns) sent through the gateway:

- **Every turn behaved as in the committed notebook run**, with no errors: 6 resolved, 4 waiting on the
  customer and 6 escalated, each to the right team (Trust & Safety, billing, technical support, general support).
- **Changes were real:** Cathy's reservation was cancelled in the CultPass database, and Frank's refund request
  was saved as an internal note for approval.
- **Logins:** sign-up with the emailed code, login, and a wrong password (`401`). A customer who put another
  customer's email in the metadata was still identified from the token, and another customer's ticket answered
  `404`.
- **Monitoring and tracing:** Prometheus counted every turn by outcome and every escalation by team, and Langfuse
  recorded a trace per message.
- **Replies took 1.2 to 13 seconds**, with a median of about 5.

Amazon Bedrock is configured and its credentials work: Titan embeddings and Claude Haiku 4.5 both answered test
calls. A full run is blocked by AWS, not the code: a new account needs the Anthropic use-case form and account
verification before Claude can be used.

### Evaluation suite

The tests script the model, so they check the code around it but can't say how often the real model gets a
ticket right. `solution/evals` runs 36 labelled tickets through the whole system with the real model, several
times each:

- **Cases** cover knowledge questions, questions no article answers, account actions (including refused
  bookings and a two-message cancellation), policy escalations, routing edge cases, Portuguese and Spanish, and
  safety (a prompt injection, another customer's data, a venue incident). Labels come from the routing rules,
  the articles and the demo customers, and are validated against the values the code uses.
- **Checks in code:** issue type, route, outcome, escalation team, tool calls, and forbidden text.
- **Checks by an LLM judge:** whether the reply is supported by the evidence, and whether it makes each case's
  expected points.
- **Report:** pass rates per check, category and case; every failed check with its route, outcome and reply;
  and cases that pass in some runs but fail in others.

Each run starts from fresh copies of the databases. A run whose message hit the fallback reply is reported as
errored and left out of the rates, and the runner checks every model with one small call before starting. Both
came from the first real attempt: the course API key had been suspended, and the outage was being scored as
wrong routes.

## 5. Results

Results from the committed run of notebook 03 against the real model:

| # | Ticket | Route | Outcome |
|---|---|---|---|
| 1 | Guest asks what a subscription includes | knowledge agent | Answered from the subscription and tiers articles |
| 2 | Cathy lists her reservations, then says "cancel that one" | account agent | Looked up, then cancelled on the follow-up |
| 3 | Guest asks to book, then gives their email | request_info → account agent | Identified as Bob, who already has that booking |
| 4 | Blocked customer can't log in | escalation | Trust & Safety, without revealing why the account is blocked |
| 5 | Customer charged twice | account agent → `request_refund` | Escalated to billing for approval |
| 6 | Login still failing after a reset | escalation (repeat issue) | Technical support |
| 7 | Bob asks for a recommendation | account agent | Rio experiences matching his stored preference, skipping one he already booked |
| 8 | Customer demands a human | escalation | Escalated immediately |
| 9 | App keeps crashing | knowledge agent | Answered with the article's troubleshooting steps |
| 10 | Student discount? | knowledge agent | No relevant article, so escalated |
| 11 | Parking at venues? | knowledge agent | Only related articles; the quality gate saw they don't answer it, so escalated |
| 12 | Frank says he loves outdoor activities | account agent | Answered; the preference was stored |
| 13 | Frank asks for recommendations in a new session | account agent | Preference recalled from the database; outdoor experiences suggested |

Across the run there were 6 resolved, 4 waiting on the customer, and 6 escalated, each escalation to the
appropriate team. The 5 edge cases are an empty message, gibberish, an unknown customer, an unknown account, and
a simulated LLM outage. All were handled without a crash. The routes and outcomes above were the same across repeated runs; the
model's wording varies.

### Evaluation

The evaluation suite runs 36 labelled tickets, 3 times each (108 runs), with `gpt-4o-mini` running the agents
and `gpt-4o` as the judge. In the latest evaluation, **108 of 108 runs passed every check (100%)**, with a median
of 4.9 seconds per message.

| Check | First evaluation | Latest |
|---|---|---|
| Issue type | 33 / 33 (100%) | 33 / 33 (100%) |
| Route | 102 / 108 (94%) | 108 / 108 (100%) |
| Outcome | 102 / 108 (94%) | 108 / 108 (100%) |
| Escalation team | 18 / 18 (100%) | 18 / 18 (100%) |
| Tool calls | 39 / 42 (93%) | 39 / 39 (100%) |
| Forbidden text | 3 / 3 (100%) | 3 / 3 (100%) |
| Key points (judge) | 119 / 132 (90%) | 135 / 135 (100%) |
| Faithfulness (judge) | 54 / 58 (93%) | 66 / 66 (100%) |
| **Runs that passed every check** | **95 / 108 (88%)** | **108 / 108 (100%)** |

| Category | First evaluation | Latest |
|---|---|---|
| Policy (refunds, blocked accounts, privacy, human requests, repeat problems) | 18 / 18 | 18 / 18 |
| Safety (prompt injection, another customer's data, venue incident) | 9 / 9 | 9 / 9 |
| Questions no article answers (escalated, not invented) | 9 / 9 | 9 / 9 |
| Routing edge cases | 6 / 6 | 6 / 6 |
| Portuguese and Spanish | 6 / 6 | 6 / 6 |
| Account actions | 26 / 33 | 33 / 33 |
| Knowledge questions | 21 / 27 | 27 / 27 |

Some totals changed with the cases: the quota case no longer expects a tool call, and the recommendation case
gained a key point. Faithfulness is judged whenever an agent's draft is sent, so more replies were judged once
fewer were escalated.

Every rule that is enforced in code held in every run of both. The first evaluation's failures were the model's
judgments: guests asking how-to questions were asked for their email instead of answered, a quota question was
answered without the evidence the gate wanted and escalated, and booking replies stated details the judge
couldn't find. Each evaluation in between pointed to the next problem, and section 6 lists them with their fixes:

| Evaluation | Runs passed | What changed before it |
|---|---|---|
| 1 | 95 / 108 (88%) | |
| 2 | 102 / 108 (94%) | Guests' how-to questions go to the knowledge agent; the gate sees the customer's account |
| 3 | 102 / 108 (94%) | A reply without a question doesn't wait for the customer; the gate and judge see stored preferences |
| 4 | 106 / 108 (98%) | Search results say whether the customer `can_book` each experience |
| 5 | 106 / 108 (98%) | Code decides whether a human is needed; reservations say whose they are |
| 6 | 107 / 108 (99%) | `recommend_experiences`; ids in the draft are checked |
| 7 | 108 / 108 (100%) | The judge counts an article's suggested phrasing as evidence |

Three caveats about the 100%:

- **It is one evaluation.** The model's answers vary, and the runs before it each missed one or two in
  different cases. Expect 98–100% when it is repeated.
- **Four of the changes were to the evaluation, not the agents.** Two loosened it: the judge doesn't count
  hedged suggestions ("you might enjoy") or an article's own suggested phrasing as unsupported claims. One
  tightened it: the recommendation case now checks that nothing is said to fit Bob's interests unless it does.
  And the quota case no longer requires a `get_my_account` call, because code already loads the account for every
  message and gives it to the agent; its key point still checks the number. Each change was checked against the
  articles and tool results first. Under the old quota expectation, the latest evaluation would be 105 of 108
  (97%).
- **A perfect run can't show that the judge would still catch a real error.** The earlier runs, with the same
  judge, did catch real ones: padded recommendations, and a reservation described as someone else's.

The first run used `gpt-4o-mini` as its own judge and scored 69%. Its faithfulness check failed 24 of 55 replies,
many of them correct: it flagged "your subscription has been paused" right after `change_subscription` had
succeeded. With `gpt-4o` as the judge, faithfulness rose to 93%, while the checks done in code, which don't depend on the
judge, stayed within one run of the first evaluation (section 6).

## 6. Challenges and solutions

These came from running the system against the real model, not from the tests. A scripted fake LLM can't
reveal what the real model will do.

| Problem found | Fix |
|---|---|
| **Search silently fell back to keywords.** `numpy` wasn't installed, so the vector store couldn't compute similarity and every search used the keyword fallback. | Added `numpy` to the requirements. The lesson: a silent fallback hides failures, so the fallback is now logged. |
| **The agent invented ids**, then, after a guard was added, passed **list positions** ("0", "1") instead of reservation ids, and in one run gave up and drafted a false "cancelled" reply. The quality gate caught that reply. | Tools only accept ids returned by a lookup in the same turn, and an unknown-id error now **lists the valid ids**. The agent then retries correctly. |
| **Answers from weakly related articles.** "Is there parking?" retrieved a QR-code article (similarity 0.41), and the agent advised "check the venue's website", which no article says. The gate marked it grounded. Raising the threshold wouldn't separate the cases: a real billing match scores 0.45. | The gate also judges whether the evidence **answers** the question, and escalates if not. The knowledge agent may not point customers to sources the articles don't mention. |
| **The gate escalated good answers** that offered help "if these steps don't work". Adding the rule to the prompt didn't change it. | Adding the rule to the **schema field's description** did, in 3 of 3 runs. With structured output, the model weighs the field descriptions more than the prompt. |
| **Inconsistent escalation of repeat problems.** Bob's "still can't log in" was escalated in one run and not in the next, depending on the gate's judgment. | A routing rule in code escalates any problem the customer reported before. It worked in every run afterwards. |
| **Long-term memory over-remembered.** Over successive runs it stored, among other things:<br>• stale booking state;<br>• a fact copied from an example in my prompt ("shares the account with partner");<br>• word-for-word quotes;<br>• internal details from the assistant's replies ("under review by Trust & Safety");<br>• the customer's email;<br>• a single booking request as a "preference". | Several layers:<br>• the memory writer sees **only the customer's messages**;<br>• the examples were removed from the prompt;<br>• code filters drop reservation state, named experiences, one-off requests and contact details;<br>• duplicates are detected ignoring case and punctuation. |
| **Recommendations were wrong in three ways:**<br>• generic advice instead of real experiences;<br>• then "no outdoor experiences", because the search matched the literal words "outdoor nature";<br>• then the whole catalog, including something the customer had already booked. | Recommendations go to the account agent. It searches the whole catalog and picks the 2–3 genuine matches. Search results now mark what the customer has `already_booked`. |
| **Reported urgency was ignored.** A ticket the source system marked critical wasn't escalated if the classifier said medium. | Routing uses the more severe of the two. |
| **Crashes.** An unexpected tool error, an unknown account, or an LLM outage ended the run without a reply. | Tools return structured errors. `process_ticket` catches failures, flags the ticket, and sends a fallback reply. |
| **Logs were free text**, so they couldn't be searched across tickets. | Structured JSONL events with fixed fields, and `search_events()`. |

Running the microservices and the evaluation suite for the first time found these:

| Problem found | Fix |
|---|---|
| **The dashboard missed the first event of each kind.** Prometheus's `increase()` can't see a counter series' first increment, and series were only created on first use, so the first escalation to each team was never counted. | The agent service creates every known series at zero when it starts. A test checks the counters after an escalated turn. |
| **An outage looked like wrong answers.** With the suspended API key, every message got the fallback reply. The evaluation scored those as wrong routes, and a safety case that expects nothing to happen "passed". | Runs that hit the fallback are reported as errored and kept out of the rates, and the runner checks every model before starting. |
| **The model never ran in the tests.** Every test replaced the account agent with a fake, so moving from the deprecated `create_react_agent` to `create_agent` had no test of the real loop. | Two tests run the real `create_agent` with a scripted tool-calling model, and any deprecated LangGraph API now fails the tests. |
| **Schema changes had no path.** Tables were created by the seed script, so changing a model meant rebuilding the databases. | Alembic migrations per database, with a test that fails when a model changes without a migration. |

Adding tracing and customer logins found these:

| Problem found | Fix |
|---|---|
| **An optional feature broke the default stack.** Marking Langfuse's secrets as required in Docker Compose made plain `docker compose up` fail too: Compose checks every service's variables, even in a profile that isn't running. | Blank defaults. Langfuse refuses to start without its secrets anyway, so a missing one still fails clearly, but only when that profile runs. |
| **Traces had no name.** Langfuse v4 takes a trace's name from its own metadata key, not from LangGraph's run name. | The run's config sets that key. Checked by querying Langfuse's trace store after a real run. |
| **A new database never reached existing installs.** Postgres runs its init script only when the volume is first created, so the new `auth` database would have existed only on fresh installs. | The script is now idempotent and a `db-setup` job runs it on every `up`. On an existing volume it created only `auth` and its role. |
| **The tracing package was optional, so tracing could fail silently.** Keys set without the package installed would just trace nothing. | Tracing logs a warning in that case and stays off, which a test covers. A manual run with Langfuse unreachable confirmed that the reply isn't slowed down or broken: the export fails in the background. |

Running the full stack and the evaluation with a real model for the first time, and adding pgvector, found these:

| Problem found | Fix |
|---|---|
| **The default Bedrock model had been retired.** Every Converse call to Claude 3.5 Haiku failed with "This model version has reached the end of its life". | The default and the examples now use Claude Haiku 4.5, the newest model the account could call. Bedrock model ids need checking against the account's inference profiles, not just the documentation. |
| **The judge graded its own model, and harshly.** With `gpt-4o-mini` judging `gpt-4o-mini`, 24 of 55 faithfulness checks failed, many on claims a successful tool call had just confirmed, so the headline was 69%. | The evaluation runs with `UDAHUB_EVAL_JUDGE_MODEL=gpt-4o`, which the suite already supported: 88% at the time, with faithfulness at 93%. Each judge failure was checked against the tool results before trusting either number. |
| **Resetting the demo data emptied the pgvector index under a running service.** `seed --reset` deletes and reloads the articles, the cascade deleted their embeddings, and every knowledge search came back empty, so every how-to question would have escalated. The in-memory index had never had this problem. | A search that finds no rows for an account with articles re-embeds them first: the query has no threshold, so no rows always means a missing index. A test reloads the articles under a live retriever. Found by resetting the data after the end-to-end run, not by the unit tests. |
| **The demo data went out of date.** The committed experiences were dated 10–16 November, so by the day of a demo, reservations were in the past and few experiences could be booked. | `01_external_db_setup.ipynb` rebuilt the database with dates 12–18 November. The evaluation and tests are unaffected: they move the dates to today on every run. |

Raising the evaluation from 88% to 100% found these. Each fix was checked with a targeted run of the failing
cases, then the full evaluation:

| Problem found | Fix |
|---|---|
| **Guests' how-to questions were asked for an email.** "Can I transfer my reservation?" or "my QR code won't scan" from a guest was classified as needing account action, and routing then asks an unknown customer who they are (0 of 3 runs each). | Routing sends a guest's reservation, subscription or billing question to the knowledge agent unless the classifier says it needs account data. The `requires_account_action` description now excludes how-it-works and troubleshooting questions, and a QR code that won't scan is `technical`. A guest asking for an account action is still asked for their email. |
| **A correct quota answer was escalated.** The account agent answered "how many experiences do I have left?" from the account details it is given at the start, without calling a tool, so the gate saw no evidence for the number (0 of 3 runs). | The gate is given the same account details, loaded by code for every message, so an answer taken from them is grounded. |
| **Answered tickets were left open.** The gate marked "let me know if..." offers as waiting for the customer, even after the field's description said they weren't. | Code decides: a draft with no `?`, `¿` or `？` can't be waiting for an answer. |
| **The gate escalated troubleshooting answers again.** Asked as one judgment whether a human is needed, it read "contact us if it persists" as needing one now. Section 6's earlier fix, the field description, wasn't enough. | The gate reports two facts instead, whether policy requires a human and whether the customer says the steps already failed, and code combines them. |
| **The agent offered bookings the customer couldn't make**, such as a premium experience on a basic plan, and wrote "I will book it for you" without calling the tool. | Search results carry `can_book` and `cannot_book_because`, from one function that applies the same rules as booking. A test tries every demo customer against every experience and checks the two always agree. |
| **The agent told Cathy her own reservation was her friend Bob's.** | Reservations are labelled with who booked them (the customer being served), and the gate treats claims about anyone else's account as ungrounded. |
| **Recommendations were padded.** When nothing new matched Bob's taste for music and dance, the agent added a sightseeing tour that "fits your interests". Embedding similarity couldn't separate them either: Christ the Redeemer scores 0.47 against "music and dance in Rio", the Carnival tour 0.46. | A `recommend_experiences` tool. Code passes only experiences the customer can book, and asks the model one narrow structured question: which of them clearly fit a stored preference, quoted exactly. Answers that name an unknown experience or preference are dropped. When nothing fits, it says so and offers alternatives labelled as not matching. |
| **A booking reply gave a wrong reservation id.** The model copied `23e89` for a 6-character id. | Every id-like token in the account agent's draft must be one the tools returned. A wrong one is corrected when the turn made exactly one booking. |
| **The judge flagged correct replies.** It counted "you might enjoy" as a claim, didn't know what `already_booked` meant, and flagged sentences an article gives as its suggested phrasing. | The judge's prompt covers each, and it sees the customer's stored preferences. Each flag was checked against the evidence first: every one was a correct reply. |

## 7. Lessons

- **Move decisions with consequences out of the prompt.** Routing, escalation policy, id validation and memory
  hygiene all ended up in code. Prompts set the agent's behavior; code enforces the rules that must always hold.
- **Examples in prompts get copied.** The model reused my example memory and my example refund reason
  word for word. Templates such as "Reported <problem>" worked better than concrete examples.
- **Schema descriptions matter more than the prompt for structured output.**
- **Test against the real model, repeatedly.** The scripted tests verify the logic, but every problem in
  section 6 only showed up in real runs. Several showed up only in some runs.
- **Keep demo data reproducible.** All demo runs used copies of the committed databases, so the documented data
  stays accurate.
- **Identity has to come from something the caller proved.** Taking the customer from the message metadata was
  fine for trusted channels, but not for customers calling directly. Setting it from a verified token at the
  gateway fixed that without touching any agent.
- **A judge needs to be stronger than the model it grades.** The same model judging itself failed correct
  answers often enough to hide where the real problems were.
- **A derived index has to notice when its source changes underneath it.** Moving the embeddings from memory
  into the database made them shared and durable, but also exposed them to the data reset.
- **Give the checkers the evidence the agent had.** The quality gate and the judge both escalated or failed
  correct answers taken from the account details and preferences the agent was given but they weren't.
- **Narrow questions get reliable answers.** "Which of these fits a preference?" asked alone, about only the
  experiences the customer can book, gave consistent answers where the same choice inside a long agent loop
  didn't.
- **Read a failure before fixing it.** Several evaluation failures were the judge's mistakes, not the agent's.
  Fixing the agent for them would have made it worse.
- **Optional features must be off without configuration.** Tracing and the extra Compose profiles change nothing
  unless they are configured, so the notebooks, the tests and the default stack keep working.

## 8. Limitations and next steps

- **Wording varies between runs.** Routes and policy decisions are fixed by code, but reply wording and some
  judgments vary. The latest evaluation passed every run, but the ones before it each missed one or two
  in different cases, and a demo run once listed a Carnival history tour as an outdoor activity.
- **Short-term memory is in process in the notebooks.** `MemorySaver` is lost on restart, and the conversation is
  rebuilt from the database. The agent service keeps the full run state in Postgres (`PostgresSaver`).
- **Search is literal.** `find_experiences` matches words, not meanings. Recommendations don't depend on it:
  `recommend_experiences` asks the model which bookable experiences fit, at the cost of one more model call. With
  a large catalog, the candidates would need narrowing first.
- **Memory filters are pattern-based.** They block every slip seen in testing, but a new kind could still get
  through. An evaluation set of conversations and expected memories would measure it.
- **Some safeguards are deliberately simple.** Whether a draft waits for the customer is decided by its question
  marks, and a wrong id is corrected only when the turn made exactly one booking; otherwise it is left for the
  gate. Both cover every case seen so far.
- **The evaluation doesn't check long-term memory** yet: what gets written to `customer_memory`.
- **Bedrock and SageMaker haven't completed a full run.** Bedrock credentials and model ids are verified (Titan
  embeddings and Claude Haiku 4.5 answered test calls), but a new AWS account must be verified and approved for
  Anthropic models first. SageMaker is covered by the offline tests only.
- **The pgvector search is an exact scan.** That suits 18 articles; thousands would need an HNSW index, created
  per embedding model because it needs a fixed vector dimension.
- **The gateway's rate limits are per replica.** They are kept in memory; with several gateway replicas they
  would need Redis or Valkey.
- **Logins are demo-grade in three ways.** Verification codes are written to the auth service's log instead of
  emailed. Tokens can't be revoked before they expire (no logout, refresh token or password reset). And the auth
  service and the gateway share one HS256 key, so the gateway could create tokens; RS256 or EdDSA would give it
  only a public key.
- **Single platform.** Knowledge search already works per account, but the account tools are CultPass-specific.

## 9. Repository

| Path | Contents |
|---|---|
| [`solution/`](solution/README.md) | The project: notebooks, `agentic/` (agents, tools, design), data and tests |
| [`solution/evals/README.md`](solution/evals/README.md) | The evaluation suite: running it, what is checked, and the case format |
| [`service/README.md`](service/README.md) | The microservices: architecture, running with Docker Compose, the demo web chat, customer logins (JWT), embeddings in Postgres (pgvector), monitoring, LLM tracing (Langfuse), schema migrations, the API |
| [`.github/workflows/tests.yml`](.github/workflows/tests.yml) | CI: both test suites, with a Postgres service that has pgvector, on every pull request |
| [`design`](design/README.md) | Architecture, routing rules, memory, logging, safeguards, inputs and outputs |
| [`solution/03_agentic_app.ipynb`](solution/03_agentic_app.ipynb) | The executed demonstration |
| [`README.md`](README.md) | Setup, tests and the full demo results |
