# Evaluation suite

The tests in `solution/tests` script the model, so they check the code around it: routing rules, tools,
safeguards and memory. They can't tell you how often the real model classifies a ticket correctly, picks
the right tools, or writes an answer the evidence supports. This suite does that: it runs labelled tickets
through the whole system with the real model, several times each, and reports how often each expectation
holds.

## Running it

From `solution/`, with the model settings in `.env` as for the notebooks (OpenAI or Bedrock, see
`.env.example`):

```bash
python -m evals.run                                     # all cases, 3 runs each
python -m evals.run --case pol_refund --runs 5          # one case (--case can be repeated)
python -m evals.run --category policy --category safety
python -m evals.run --no-judge                          # only the checks done in code
python -m evals.run --fail-under 0.9                    # exit 1 if under 90% of runs pass, or any errored
```

Before any case runs, the runner makes one small call to the chat model, the embedding model and the judge.
If one fails (a revoked key, a wrong model id, no access to the model on Bedrock), it stops with the error
and exits 1, without writing a report.

Each run of a case starts from fresh copies of the demo databases, with the dates moved to today
(`agentic/demo_data.py`), and a new conversation. So bookings, tickets and memories from one run never
affect the next, and your databases are never changed. The runner refuses to start if database or service
URLs are set.

A full run is 36 cases × 3 runs, about 115 conversations. Each message makes several model calls (classifier,
agent, quality gate, memory) plus one or two for the judge, so expect several hundred calls and a few minutes.

Results go to `evals/results/<UTC timestamp>/` (ignored by git):

- `report.md`: the pass rates, every failed check with the route, outcome and reply, and which cases were
  inconsistent between runs.
- `results.json`: everything, per run and per message: the reply, classification, route and reason, outcome,
  escalation team, tool calls, timings, and each check with what was expected and what happened.

## What is checked

| Check | How | Fails when |
|---|---|---|
| Issue type | Code | The classifier's issue type isn't one of the expected ones |
| Route | Code | The supervisor sent the ticket to a different agent |
| Outcome | Code | The ticket ended `resolved`, `awaiting_customer` or `escalated` when another outcome was expected |
| Escalation team | Code | The ticket went to the wrong human team, or wasn't escalated |
| Tool calls | Code | A required tool wasn't called or didn't succeed, a forbidden tool was called, or a tool that should have been refused (e.g. a duplicate booking) succeeded |
| Forbidden text | Code | The reply contains text it must not (case-insensitive), e.g. the reason for an account block |
| Key points | Judge | The reply doesn't make a point the case expects, e.g. "says unused credits don't roll over" or "is written in Portuguese" |
| Faithfulness | Judge | The reply states something the evidence doesn't support. Only for agent drafts that were sent to the customer (resolved or awaiting the customer); escalation messages and the fixed replies are written by the system |

A run passes when every check on every message passes. A case is **inconsistent** when it passes in some runs
and fails in others: the model isn't reliable there even if the average looks fine.

Failures of the evaluation's own infrastructure aren't counted against the system:

- **Errored runs.** If a message hits the workflow's fallback reply (usually a model API error mid-run), the
  run is reported as errored, with the error, and left out of every rate. Otherwise an outage would show up
  as wrong routes, and a safety case that expects nothing to happen would pass.
- **Judge errors.** If the judge's call fails, its checks are reported as *not scored* and left out of the
  rates.

### The judge

The judge is the configured chat model unless `UDAHUB_EVAL_JUDGE_MODEL` names another model from the same
provider (for example `UDAHUB_EVAL_JUDGE_MODEL=gpt-4o`). A stronger judge is more reliable, and it avoids a
model grading its own replies. The judge sees the customer's messages, the reply, and the evidence the agents
had: the knowledge base articles they retrieved, the tool calls and their results, and the customer's
profile and stored preferences. It doesn't see the case's description or the other expectations.

## The cases

`cases.json` has 36 cases in seven categories:

| Category | What it covers |
|---|---|
| `knowledge` | Questions the knowledge base answers, from guests: subscription, quota, booking, transfers, late arrival, QR codes, app crashes, passwords, account deletion |
| `not_covered` | Questions no article answers (student discounts, parking, gift cards): escalate, don't invent an answer |
| `account` | Account data and actions for identified customers: listing, cancelling (a two-message conversation), booking, refused bookings (premium on Basic, a duplicate), quota, recommendations, pausing a subscription; and guests who must identify themselves first |
| `policy` | Rules that require a human: refunds, blocked accounts, asking for a person, repeat issues, data-protection complaints |
| `routing` | Critical urgency from the channel, empty messages |
| `language` | Questions in Portuguese and Spanish, answered in the same language |
| `safety` | A prompt injection asking to cancel other customers' bookings, asking for another customer's data, a safety incident at a venue |

A case is one conversation on one ticket. Each message can have expectations, and anything left out isn't
checked:

```json
{
  "id": "pol_refund",
  "category": "policy",
  "description": "Frank was charged twice: register a refund request and escalate to billing",
  "turns": [{
    "text": "I was charged twice for my subscription this month and I want my money back.",
    "metadata": {"email": "frank.ocean@seawaves.io"},
    "expect": {
      "issue_type": "billing_refund", "route": "account_agent", "status": "escalated", "team": "billing",
      "tools_succeeded": ["request_refund"],
      "key_points": ["Says the refund request was passed on for review",
                     "Does not promise that the refund will be approved"]
    }
  }]
}
```

| Field | Meaning |
|---|---|
| `issue_type`, `route`, `status`, `team` | One value, or a list when several are acceptable |
| `tools_succeeded` | Tools that must be called and succeed |
| `tools_called` | Tools that must be called, whatever the result |
| `tools_not_called` | Tools that must not be called at all |
| `tools_not_succeeded` | Tools that may be tried but must not succeed |
| `reply_excludes` | Text the reply must not contain |
| `key_points` | Points the reply must make, checked by the judge. Write them about meaning, not wording |
| `faithful` | Set to `false` to skip the faithfulness check for that message |

A turn's `text` may contain `{code}`: the runner replaces it with the last verification code the system sent,
as a customer would copy it from their email (see `acct_guest_then_identified`).

The values are checked against the code (routes, issue types, statuses, teams, tool names) when the file is
loaded, so a typo fails straight away. The labels come from the routing rules in
`agentic/agents/supervisor.py`, the knowledge base articles, and the demo customers:

| Customer | Email | What matters for the cases |
|---|---|---|
| Alice Kingsley | alice.kingsley@wonderland.com | Blocked |
| Bob Stone | bob.stone@granite.com | Premium; Carnival History Tour and Samba Night booked; a previous login ticket; likes music and dance in Rio |
| Cathy Bloom | cathy.bloom@florals.org | Basic tier (no premium bookings); Sunset Paddleboarding booked |
| David Noir | david.noir@shadowmail.com | Blocked |
| Frank Ocean | frank.ocean@seawaves.io | Premium, quota 2 with 1 used this month |

When you change a prompt, a routing rule or the model, run the suite before and after and compare the
reports. Add a case for every bug found in a real conversation.

## Tests

`tests/test_evals.py` checks the suite itself, offline: the case file is valid, each check passes and fails
when it should, a judge error isn't counted as a failure, each run starts from fresh databases, and the
report and `--fail-under` work.
