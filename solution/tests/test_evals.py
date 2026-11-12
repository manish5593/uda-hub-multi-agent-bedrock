"""The evaluation suite's own machinery (offline): the case file, the scoring, and a full run with a
scripted model and a scripted judge. The evaluation itself runs against the real model: python -m evals.run
"""
import json

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from agentic import config
from agentic.agents import (account_agent, classifier, escalation_agent, knowledge_agent, memory_agent,
                            quality_gate, supervisor)
from agentic.agents.escalation_agent import EscalationHandoff
from agentic.agents.memory_agent import MemoryExtraction
from agentic.state import ResolutionAssessment, TicketClassification
import agentic.tools.knowledge_tools as kt
from evals import run as runner
from evals.cases import Case, Expect, load_cases
from evals.judge import Faithfulness, PointVerdict
from evals.scoring import deterministic_checks, judged_checks, needs_faithfulness


def test_the_case_file_is_valid_and_every_case_checks_something():
    cases = load_cases()
    assert len(cases) >= 30
    for case in cases:
        for turn in case.turns:
            e = turn.expect
            assert any([e.issue_type, e.route, e.status, e.team, e.tools_succeeded, e.tools_called,
                        e.tools_not_called, e.tools_not_succeeded, e.reply_excludes, e.key_points]), case.id


@pytest.mark.parametrize("field, value", [("route", "billing_agent"), ("status", "closed"), ("team", "sales"),
                                          ("issue_type", "refund"), ("tools_called", ["delete_everything"])])
def test_labels_must_be_values_the_code_uses(field, value):
    with pytest.raises(ValidationError, match="unknown"):
        Expect(**{field: value})


def test_duplicate_case_ids_are_refused(tmp_path):
    case = {"id": "a", "category": "knowledge", "description": "x", "turns": [{"text": "hi"}]}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case, case]))
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


STATE = {
    "classification": {"issue_type": "billing_refund"},
    "route": "account_agent",
    "ticket_status": "escalated",
    "escalation": {"team": "billing"},
    "actions_taken": [
        {"tool": "get_my_reservations", "args": {}, "result": [{"reservation_id": "r1"}]},
        {"tool": "request_refund", "args": {}, "result": {"success": True, "refund_request_id": "RF-1"}},
        {"tool": "book_experience", "args": {}, "result": {"success": False, "error": "Already reserved."}},
    ],
    "final_response": "I've passed your refund request to our billing team.",
}


def by_name(checks):
    return {c.name: c for c in checks}


def test_deterministic_checks_pass_when_everything_matches():
    expect = Expect(issue_type="billing_refund", route="account_agent", status=["escalated"], team="billing",
                    tools_succeeded=["request_refund"], tools_called=["book_experience"],
                    tools_not_called=["cancel_my_reservation"], tools_not_succeeded=["book_experience"],
                    reply_excludes=["guaranteed"])
    checks = deterministic_checks(expect, STATE)
    assert [c.name for c in checks] == ["issue_type", "route", "status", "team", "tools", "reply_excludes"]
    assert all(c.passed for c in checks)


def test_deterministic_checks_report_what_went_wrong():
    expect = Expect(route="knowledge_agent", team="trust_and_safety", tools_succeeded=["book_experience"],
                    tools_not_called=["request_refund"], reply_excludes=["BILLING TEAM"])
    checks = by_name(deterministic_checks(expect, STATE))
    assert not checks["route"].passed and checks["route"].actual == "account_agent"
    assert not checks["team"].passed
    assert checks["tools"].detail == "book_experience not called successfully; request_refund was called"
    assert checks["reply_excludes"].actual == ["BILLING TEAM"]  # case-insensitive


def test_a_dict_with_an_error_counts_as_a_failed_tool_call():
    state = {**STATE, "actions_taken": [{"tool": "get_my_account", "args": {}, "result": {"error": "Customer not found."}}]}
    [check] = deterministic_checks(Expect(tools_succeeded=["get_my_account"]), state)
    assert not check.passed


def test_unspecified_expectations_are_not_checked():
    assert deterministic_checks(Expect(), STATE) == []


@pytest.mark.parametrize("route, status, judged", [("knowledge_agent", "resolved", True),
                                                   ("account_agent", "awaiting_customer", True),
                                                   ("account_agent", "escalated", False),
                                                   ("escalation_agent", "escalated", False),
                                                   ("request_info", "awaiting_customer", False)])
def test_faithfulness_is_judged_only_for_agent_drafts_that_were_sent(route, status, judged):
    state = {"route": route, "ticket_status": status, "final_response": "reply"}
    assert needs_faithfulness(Expect(), state) is judged
    assert needs_faithfulness(Expect(faithful=False), state) is False


class FakeJudge:
    def __init__(self, unsupported=(), met=True, fail=False):
        self.unsupported, self.met, self.fail = list(unsupported), met, fail

    def check_key_points(self, messages, reply, points):
        if self.fail:
            raise TimeoutError("judge timed out")
        return [PointVerdict(point=p, met=self.met, reason="checked") for p in points]

    def check_faithfulness(self, messages, reply, state):
        if self.fail:
            raise TimeoutError("judge timed out")
        return Faithfulness(unsupported_claims=self.unsupported)


def test_judged_checks():
    state = {"route": "knowledge_agent", "ticket_status": "resolved", "final_response": "Refunds take 2 days."}
    expect = Expect(key_points=["Mentions refunds", "Is polite"])
    checks = judged_checks(expect, state, ["Refund?"], FakeJudge(unsupported=["Refunds take 2 days"], met=True))
    assert [(c.name, c.passed) for c in checks] == [("key_point", True), ("key_point", True), ("faithfulness", False)]
    assert checks[2].detail == "Refunds take 2 days"


def test_a_judge_failure_is_not_scored_rather_than_failed():
    state = {"route": "knowledge_agent", "ticket_status": "resolved", "final_response": "reply"}
    checks = judged_checks(Expect(key_points=["x"]), state, ["q"], FakeJudge(fail=True))
    assert [c.passed for c in checks] == [None, None]
    assert "judge timed out" in checks[0].detail


# ---------------------------------------------------------------- a full run with a scripted model


def classification(**kw):
    base = dict(issue_type="general_info", urgency="low", sentiment="neutral", summary="question",
                requires_account_action=False, human_requested=False, customer_email=None,
                language="English", confidence=0.9)
    return TicketClassification(**{**base, **kw})


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = outputs

    def invoke(self, messages):
        return AIMessage(content="Here is the answer.")

    def with_structured_output(self, schema):
        outputs = self.outputs

        class _Runner:
            def invoke(self, messages):
                return outputs[schema]
        return _Runner()


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr(kt, "get_embeddings", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    kt._retrievers.clear()
    llm = FakeLLM({
        TicketClassification: classification(issue_type="login_access"),
        ResolutionAssessment: ResolutionAssessment(confidence=0.9, grounded=True, answered_by_evidence=True,
                                                   awaiting_customer=False, reason="ok"),
        EscalationHandoff: EscalationHandoff(team="trust_and_safety", priority="high", summary="blocked",
                                             steps_taken=[], recommended_next_step="review",
                                             customer_message="A specialist will review your case."),
        MemoryExtraction: MemoryExtraction(memories=[]),
    })
    for module in (classifier, knowledge_agent, quality_gate, escalation_agent, memory_agent, account_agent,
                   supervisor):
        monkeypatch.setattr(module, "get_llm", lambda *a, **k: llm)
    return llm


def blocked_login_case(**expect):
    return Case(id="blocked", category="policy", description="Alice is blocked",
                turns=[{"text": "I can't log in.", "metadata": {"email": "alice.kingsley@wonderland.com"},
                        "expect": {"route": "escalation_agent", "status": "escalated", **expect}}])


def test_run_case_scores_a_real_workflow_run(scripted):
    paths = (config.UDAHUB_DB_PATH, config.CULTPASS_DB_PATH, config.EVENT_LOG_PATH)
    record = runner.run_case(blocked_login_case(team="trust_and_safety", key_points=["Says a specialist will review"]),
                             run=1, judge=FakeJudge())

    assert record["passed"]
    [turn] = record["turns"]
    assert (turn["route"], turn["status"], turn["team"]) == ("escalation_agent", "escalated", "trust_and_safety")
    assert {c["name"] for c in turn["checks"]} == {"route", "status", "team", "key_point"}
    # the run used its own copies, and the configured databases are back afterwards
    assert (config.UDAHUB_DB_PATH, config.CULTPASS_DB_PATH, config.EVENT_LOG_PATH) == paths


def test_runs_start_from_fresh_databases(scripted):
    """A ticket created in one run must not exist in the next (it would change routing, e.g. repeat issues)."""
    case = blocked_login_case()
    first = runner.run_case(case, run=1, judge=None)
    second = runner.run_case(case, run=1, judge=None)  # same ticket id on purpose
    assert first["turns"][0]["route_reason"] == second["turns"][0]["route_reason"]
    assert "already with the human team" not in second["turns"][0]["route_reason"]


def test_summary_and_report(scripted, tmp_path):
    good = blocked_login_case(team="trust_and_safety")
    bad = Case(**{**blocked_login_case(team="billing").model_dump(), "id": "wrong_team"})
    records = [runner.run_case(good, 1, None), runner.run_case(good, 2, None),
               runner.run_case(bad, 1, None), runner.run_case(bad, 2, None)]
    summary = runner.summarize([good, bad], records)

    assert (summary["runs_passed"], summary["runs"], summary["pass_rate"]) == (2, 4, 0.5)
    assert summary["checks"]["team"] == {"passed": 2, "total": 4, "not_scored": 0}
    assert summary["by_case"]["wrong_team"]["consistent"]  # fails every time: consistent, not flaky

    meta = {"date": "today", "provider": "openai", "chat_model": "m", "embedding_model": "e", "judge_model": None,
            "cases": 2, "turns": 2, "runs_per_case": 2, "commit": "abc"}
    report = runner.render_report(meta, summary, records)
    assert "**2 of 4 runs passed every check (50%).**" in report
    assert "| Escalation team | 2 / 4 | 50% |" in report
    assert "`wrong_team` run 1: **Escalation team**. Expected billing, got trust_and_safety." in report


def test_main_writes_results_and_applies_fail_under(scripted, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "check_models", lambda judge: None)
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps([blocked_login_case(team="billing").model_dump(exclude_defaults=True)]))
    out = tmp_path / "out"
    code = runner.main(["--cases", str(cases_file), "--runs", "1", "--no-judge", "--out", str(out),
                        "--fail-under", "0.5"])
    assert code == 1
    results = json.loads((out / "results.json").read_text())
    assert results["summary"]["runs"] == 1 and not results["runs"][0]["passed"]
    assert (out / "report.md").read_text().startswith("# UDA-Hub evaluation")


def test_the_evals_refuse_to_touch_real_databases_or_services(monkeypatch):
    monkeypatch.setattr(config, "TICKET_SERVICE_URL", "http://tickets:8000")
    with pytest.raises(SystemExit, match="TICKET_SERVICE_URL"):
        runner.refuse_remote_config()


def test_a_run_that_hits_the_fallback_reply_is_errored_not_scored(scripted, monkeypatch):
    """E.g. the API key was revoked: every message gets the fallback. That must not count as a pass or a fail."""
    def unreachable(*a, **k):
        raise ConnectionError("model API unreachable")
    monkeypatch.setattr(classifier, "get_llm", unreachable)
    case = blocked_login_case(team="trust_and_safety")
    records = [runner.run_case(case, 1, None), runner.run_case(case, 2, None)]

    assert all(r["errored"] and not r["passed"] for r in records)
    summary = runner.summarize([case], records)
    assert (summary["runs"], summary["errored_runs"], summary["pass_rate"]) == (0, 2, None)
    assert summary["checks"] == {}
    meta = {"date": "today", "provider": "openai", "chat_model": "m", "embedding_model": "e", "judge_model": None,
            "cases": 1, "turns": 1, "runs_per_case": 2, "commit": "abc"}
    report = runner.render_report(meta, summary, records)
    assert "**2 runs errored**" in report
    assert "## Errored runs" in report and "model API unreachable" in report
    assert "## Failed checks\n\nNone." in report


def test_fail_under_also_fails_when_runs_errored(scripted, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "check_models", lambda judge: None)
    monkeypatch.setattr(classifier, "get_llm", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps([blocked_login_case().model_dump(exclude_defaults=True)]))
    assert runner.main(["--cases", str(cases_file), "--runs", "1", "--no-judge", "--out", str(tmp_path / "o"),
                        "--fail-under", "0"]) == 1


def test_an_unusable_model_stops_the_run_before_any_case(monkeypatch):
    import agentic.llm

    class Revoked:
        def invoke(self, prompt):
            raise PermissionError("Key access is suspended")
    monkeypatch.setattr(agentic.llm, "get_llm", lambda *a, **k: Revoked())
    with pytest.raises(SystemExit, match="chat model .* isn't usable: PermissionError: Key access is suspended"):
        runner.check_models(judge=None)
