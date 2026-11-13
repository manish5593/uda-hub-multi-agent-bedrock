"""Score one turn of a case against its expectations.

Deterministic checks (classification, route, outcome, escalation team, tools, forbidden text) are plain code.
Only faithfulness and key points need the LLM judge, and they are skipped when `judge` is None.
"""
from dataclasses import asdict, dataclass
from typing import Optional

from evals.cases import Expect

# Replies the system writes itself (no agent draft): nothing for the faithfulness judge to check
DRAFTING_ROUTES = {"knowledge_agent", "account_agent"}
SENT_DRAFT_STATUSES = {"resolved", "awaiting_customer"}


@dataclass
class Check:
    name: str                 # issue_type, route, status, team, tools, reply_excludes, key_point, faithfulness
    passed: Optional[bool]    # None: couldn't be scored (the judge failed); left out of the pass rates
    expected: object = None
    actual: object = None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def tool_calls(state: dict) -> list[dict]:
    """The tools called this turn, each with whether it succeeded (a dict result without success: False)."""
    calls = []
    for action in state.get("actions_taken") or []:
        result = action.get("result")
        success = not (isinstance(result, dict) and (result.get("success") is False or "error" in result))
        calls.append({"tool": action["tool"], "success": success})
    return calls


def _one_of(name: str, expected: Optional[list], actual) -> Optional[Check]:
    if expected is None:
        return None
    return Check(name, actual in expected, expected if len(expected) > 1 else expected[0], actual)


def deterministic_checks(expect: Expect, state: dict) -> list[Check]:
    checks = [
        _one_of("issue_type", expect.issue_type, (state.get("classification") or {}).get("issue_type")),
        _one_of("route", expect.route, state.get("route")),
        _one_of("status", expect.status, state.get("ticket_status")),
        _one_of("team", expect.team, (state.get("escalation") or {}).get("team")),
    ]
    checks = [c for c in checks if c is not None]

    calls = tool_calls(state)
    called = {c["tool"] for c in calls}
    succeeded = {c["tool"] for c in calls if c["success"]}
    problems = ([f"{t} not called successfully" for t in expect.tools_succeeded if t not in succeeded]
                + [f"{t} not called" for t in expect.tools_called if t not in called]
                + [f"{t} was called" for t in expect.tools_not_called if t in called]
                + [f"{t} succeeded" for t in expect.tools_not_succeeded if t in succeeded])
    if expect.tools_succeeded or expect.tools_called or expect.tools_not_called or expect.tools_not_succeeded:
        wanted = {k: v for k, v in (("succeeded", expect.tools_succeeded), ("called", expect.tools_called),
                                    ("not_called", expect.tools_not_called),
                                    ("not_succeeded", expect.tools_not_succeeded)) if v}
        checks.append(Check("tools", not problems, wanted, calls, "; ".join(problems)))

    reply = (state.get("final_response") or "").lower()
    if expect.reply_excludes:
        found = [text for text in expect.reply_excludes if text.lower() in reply]
        checks.append(Check("reply_excludes", not found, expect.reply_excludes, found,
                            f"reply contains {found}" if found else ""))
    return checks


def needs_faithfulness(expect: Expect, state: dict) -> bool:
    """Judge an agent's draft that was sent to the customer. Escalation messages, requests for the email
    and the fallback reply are written by the system itself."""
    return (expect.faithful and state.get("route") in DRAFTING_ROUTES
            and state.get("ticket_status") in SENT_DRAFT_STATUSES and bool(state.get("final_response")))


def judged_checks(expect: Expect, state: dict, customer_messages: list[str], judge) -> list[Check]:
    """`judge` is the evals.judge module (or a stand-in with the same two functions)."""
    checks = []
    reply = state.get("final_response") or ""
    if expect.key_points:
        try:
            for verdict in judge.check_key_points(customer_messages, reply, expect.key_points,
                                                  session_id=state.get("ticket_id")):
                checks.append(Check("key_point", verdict.met, verdict.point, None, verdict.reason))
        except Exception as exc:  # the judge failing is not the system failing
            checks += [Check("key_point", None, p, None, f"judge error: {type(exc).__name__}: {exc}")
                       for p in expect.key_points]
    if needs_faithfulness(expect, state):
        try:
            result = judge.check_faithfulness(customer_messages, reply, state)
            checks.append(Check("faithfulness", not result.unsupported_claims, "no unsupported claims",
                                result.unsupported_claims,
                                "; ".join(result.unsupported_claims)))
        except Exception as exc:
            checks.append(Check("faithfulness", None, "no unsupported claims", None,
                                f"judge error: {type(exc).__name__}: {exc}"))
    return checks
