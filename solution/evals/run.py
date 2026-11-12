"""Run the labelled cases against the real model and report how often the system gets each one right.

From solution/, with the model credentials in .env (as for the notebooks):

    python -m evals.run                         # every case, 3 runs each
    python -m evals.run --runs 5 --case pol_refund --case acct_quota_left
    python -m evals.run --category policy --no-judge
    python -m evals.run --fail-under 0.9        # exit 1 if fewer than 90% of runs pass

Each run of a case starts from fresh copies of the demo databases (dates moved to today) and a new
conversation, so bookings and memories from one run never leak into another. The models are called for
real: a full run is about 115 conversations and several hundred model calls.

Writes results.json (every turn, every check) and report.md to evals/results/<UTC timestamp>/, and
results.partial.json after every run while it goes, so a stopped evaluation keeps the runs it finished.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

SOLUTION_DIR = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    # agentic.config reads the environment when it's imported, so load .env first. Only when run as a
    # command: importing this module (e.g. from the tests) must not load real credentials.
    load_dotenv(SOLUTION_DIR / ".env")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agentic import config, verification  # noqa: E402
from agentic.verification import deliver_code  # noqa: E402
from agentic.demo_data import copy_demo_databases  # noqa: E402
from agentic.tools import db  # noqa: E402
from agentic.workflow import build_graph, process_ticket  # noqa: E402
from evals import judge as llm_judge  # noqa: E402
from evals.cases import CASES_PATH, Case, load_cases  # noqa: E402
from evals.scoring import deterministic_checks, judged_checks, tool_calls  # noqa: E402

RESULTS_DIR = Path(__file__).with_name("results")
CHECK_ORDER = ["issue_type", "route", "status", "team", "tools", "reply_excludes", "key_point", "faithfulness"]
CHECK_LABELS = {"issue_type": "Issue type", "route": "Route", "status": "Outcome", "team": "Escalation team",
                "tools": "Tool calls", "reply_excludes": "Forbidden text", "key_point": "Key points (judge)",
                "faithfulness": "Faithfulness (judge)"}


def refuse_remote_config() -> None:
    """The evals run the workflow in this process against SQLite copies. With database or service URLs set,
    the tools would read and change a real database instead."""
    set_ = [name for name in ("UDAHUB_DB_URL", "CULTPASS_DB_URL", "CULTPASS_SERVICE_URL", "TICKET_SERVICE_URL",
                              "KNOWLEDGE_SERVICE_URL") if getattr(config, name)]
    if set_:
        raise SystemExit(f"Unset {', '.join(set_)}: the evals run against throwaway copies of the demo databases.")


def run_case(case: Case, run: int, judge=llm_judge) -> dict:
    """One conversation from fresh databases. Returns the run's record, with every turn and check."""
    saved = (config.UDAHUB_DB_PATH, config.CULTPASS_DB_PATH, config.EVENT_LOG_PATH)
    with tempfile.TemporaryDirectory(prefix=f"udahub-eval-{case.id}-") as tmp:
        config.UDAHUB_DB_PATH, config.CULTPASS_DB_PATH = copy_demo_databases(Path(tmp))
        config.EVENT_LOG_PATH = Path(tmp) / "events.jsonl"
        try:
            graph = build_graph(MemorySaver())
            ticket_id = f"eval-{case.id}-{run}"
            customer_messages, turns = [], []
            # The customer's inbox: a turn's "{code}" is replaced by the last verification code sent
            inbox = []
            verification.deliver_code = lambda email, code: inbox.append(code)
            for i, turn in enumerate(case.turns):
                text = turn.text.replace("{code}", inbox[-1]) if inbox else turn.text
                customer_messages.append(text)
                start = time.perf_counter()
                state = process_ticket(text, ticket_id, turn.metadata, agent=graph)
                seconds = time.perf_counter() - start
                checks = deterministic_checks(turn.expect, state)
                if judge is not None:
                    checks += judged_checks(turn.expect, state, customer_messages, judge)
                turns.append({
                    "turn": i + 1, "text": text, "reply": state.get("final_response", ""),
                    "issue_type": (state.get("classification") or {}).get("issue_type"),
                    "route": state.get("route"), "route_reason": state.get("route_reason"),
                    "status": state.get("ticket_status"),
                    "team": (state.get("escalation") or {}).get("team"),
                    "tools": tool_calls(state), "error": state.get("error"),
                    # Why the reply was sent or escalated, so a failure can be explained from the results
                    "assessment": {k: (state.get("assessment") or {}).get(k) for k in
                                   ("confidence", "grounded", "answered_by_evidence", "needs_human",
                                    "awaiting_customer", "reason")} if state.get("assessment") else None,
                    "escalation_reason": (state.get("escalation") or {}).get("reason"),
                    "seconds": round(seconds, 2), "checks": [c.to_dict() for c in checks],
                })
        finally:
            verification.deliver_code = deliver_code
            config.UDAHUB_DB_PATH, config.CULTPASS_DB_PATH, config.EVENT_LOG_PATH = saved
            db._engine.cache_clear()  # drop the engines of the deleted copies
    scored = [c for t in turns for c in t["checks"] if c["passed"] is not None]
    # A message that hit the workflow's fallback (e.g. the model API failed) says nothing about the system's
    # decisions, so the run is reported as errored and left out of the pass rates
    errored = any(t["error"] for t in turns)
    return {"case": case.id, "category": case.category, "run": run, "turns": turns, "errored": errored,
            "passed": not errored and all(c["passed"] for c in scored)}


def check_models(judge) -> None:
    """One small call to each model before the runs, so a bad key or model id stops the run straight away
    instead of producing a report full of fallback replies."""
    from agentic.llm import get_embeddings, get_llm
    targets = [("chat model", config.CHAT_MODEL, lambda: get_llm().invoke("Reply with the word OK.")),
               ("embedding model", config.EMBEDDING_MODEL, lambda: get_embeddings().embed_query("test"))]
    if judge is not None:
        targets.append(("judge model", judge.judge_model_name(), lambda: judge.judge_llm().invoke("Reply with the word OK.")))
    for what, name, call in targets:
        try:
            call()
        except Exception as exc:
            raise SystemExit(f"The {what} ({config.LLM_PROVIDER}/{name}) isn't usable: {type(exc).__name__}: "
                             f"{str(exc)[:300]}\nCheck the model settings in solution/.env.") from None


def summarize(cases: list[Case], all_records: list[dict]) -> dict:
    records = [r for r in all_records if not r["errored"]]
    checks = defaultdict(lambda: {"passed": 0, "total": 0, "not_scored": 0})
    for record in records:
        for turn in record["turns"]:
            for c in turn["checks"]:
                if c["passed"] is None:
                    checks[c["name"]]["not_scored"] += 1
                else:
                    checks[c["name"]]["total"] += 1
                    checks[c["name"]]["passed"] += c["passed"]
    by_case = {}
    for case in cases:
        runs = [r for r in records if r["case"] == case.id]
        passed = sum(r["passed"] for r in runs)
        by_case[case.id] = {"category": case.category, "description": case.description, "passed": passed,
                            "runs": len(runs), "consistent": passed in (0, len(runs)),
                            "errored": sum(r["errored"] for r in all_records if r["case"] == case.id)}
    by_category = defaultdict(lambda: {"passed": 0, "runs": 0})
    for r in records:
        by_category[r["category"]]["passed"] += r["passed"]
        by_category[r["category"]]["runs"] += 1
    seconds = sorted(t["seconds"] for r in records for t in r["turns"])
    return {
        "runs_passed": sum(r["passed"] for r in records), "runs": len(records),
        "pass_rate": round(sum(r["passed"] for r in records) / len(records), 4) if records else None,
        "checks": {k: checks[k] for k in CHECK_ORDER if k in checks},
        "by_category": dict(by_category), "by_case": by_case,
        "errored_runs": sum(r["errored"] for r in all_records),
        "seconds_per_turn": {"median": seconds[len(seconds) // 2] if seconds else None,
                             "max": seconds[-1] if seconds else None},
    }


def _pct(passed: int, total: int) -> str:
    return f"{passed / total:.0%}" if total else "n/a"


def _short(value, limit=160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return (text[:limit] + "…") if len(text) > limit else text


def render_report(meta: dict, summary: dict, records: list[dict]) -> str:
    lines = [
        "# UDA-Hub evaluation",
        "",
        f"- **Date:** {meta['date']}",
        f"- **Model:** {meta['provider']} `{meta['chat_model']}`, embeddings `{meta['embedding_model']}`",
        f"- **Judge:** {'`' + meta['judge_model'] + '`' if meta['judge_model'] else 'off (--no-judge)'}",
        f"- **Cases:** {meta['cases']} ({meta['turns']} customer messages), {meta['runs_per_case']} "
        f"run{'s' if meta['runs_per_case'] != 1 else ''} each",
        f"- **Commit:** `{meta['commit']}`",
        "",
        "## Summary",
        "",
        f"**{summary['runs_passed']} of {summary['runs']} runs passed every check "
        f"({_pct(summary['runs_passed'], summary['runs'])}).** Median time per message "
        f"{summary['seconds_per_turn']['median']} s (max {summary['seconds_per_turn']['max']} s).",
        "",
    ]
    if summary["errored_runs"]:
        lines += [f"**{summary['errored_runs']} runs errored** (a message hit the fallback reply, usually because a "
                  "model call failed). They are left out of every rate below; see [Errored runs](#errored-runs).", ""]
    lines += ["| Check | Passed | Rate |", "|---|---|---|"]
    for name, c in summary["checks"].items():
        note = f" ({c['not_scored']} not scored)" if c["not_scored"] else ""
        lines.append(f"| {CHECK_LABELS[name]} | {c['passed']} / {c['total']}{note} | {_pct(c['passed'], c['total'])} |")
    lines += ["", "## By category", "", "| Category | Runs passed | Rate |", "|---|---|---|"]
    for cat, c in sorted(summary["by_category"].items()):
        lines.append(f"| {cat} | {c['passed']} / {c['runs']} | {_pct(c['passed'], c['runs'])} |")
    lines += ["", "## By case", "",
              "Inconsistent: the case passed in some runs and failed in others.", "",
              "| Case | Category | Runs passed | |", "|---|---|---|---|"]
    for case_id, c in summary["by_case"].items():
        mark = "" if c["passed"] == c["runs"] else ("inconsistent" if not c["consistent"] else "fails")
        if c["errored"]:
            mark = ", ".join(filter(None, [mark, f"{c['errored']} errored"]))
        lines.append(f"| `{case_id}` | {c['category']} | {c['passed']} / {c['runs']} | {mark} |")

    failures = [(r, t, c) for r in records if not r["errored"]
                for t in r["turns"] for c in t["checks"] if c["passed"] is False]
    lines += ["", "## Failed checks", ""]
    if not failures:
        lines.append("None.")
    for r, t, c in failures:
        where = f"`{r['case']}` run {r['run']}" + (f", message {t['turn']}" if len(r["turns"]) > 1 else "")
        lines.append(f"- {where}: **{CHECK_LABELS[c['name']]}**. Expected {_short(c['expected'])}, "
                     f"got {_short(c['actual'])}." + (f" {_short(c['detail'])}" if c["detail"] else ""))
        lines.append(f"  - Route `{t['route']}` ({_short(t['route_reason'] or '', 100)}), outcome `{t['status']}`. "
                     f"Reply: \"{_short(t['reply'])}\"")
    errored = [(r, t) for r in records if r["errored"] for t in r["turns"] if t["error"]]
    if errored:
        lines += ["", "## Errored runs", ""]
        lines += [f"- `{r['case']}` run {r['run']}" + (f", message {t['turn']}" if len(r["turns"]) > 1 else "")
                  + f": {_short(t['error'], 300)}" for r, t in errored]
    not_scored = [(r, t, c) for r in records if not r["errored"]
                  for t in r["turns"] for c in t["checks"] if c["passed"] is None]
    if not_scored:
        lines += ["", "## Not scored (judge errors)", ""]
        lines += [f"- `{r['case']}` run {r['run']}: {CHECK_LABELS[c['name']]}: {_short(c['detail'])}"
                  for r, t, c in not_scored]
    return "\n".join(lines) + "\n"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=SOLUTION_DIR, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=SOLUTION_DIR).stdout.strip()
        return out + (" (uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3, help="runs per case (default 3)")
    parser.add_argument("--case", action="append", default=[], help="only this case id (repeatable)")
    parser.add_argument("--category", action="append", default=[], help="only this category (repeatable)")
    parser.add_argument("--cases", type=Path, default=CASES_PATH, help="cases file (default evals/cases.json)")
    parser.add_argument("--out", type=Path, help="output directory (default evals/results/<timestamp>)")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge (key points, faithfulness)")
    parser.add_argument("--fail-under", type=float,
                        help="exit 1 if the share of scored runs passing is below this, or if any run errored")
    args = parser.parse_args(argv)

    refuse_remote_config()
    cases = load_cases(args.cases)
    if args.case:
        unknown = set(args.case) - {c.id for c in cases}
        if unknown:
            parser.error(f"unknown case ids: {sorted(unknown)}")
        cases = [c for c in cases if c.id in args.case]
    if args.category:
        cases = [c for c in cases if c.category in args.category]
    if not cases:
        parser.error("no cases selected")

    judge = None if args.no_judge else llm_judge
    check_models(judge)
    started = datetime.now(timezone.utc)
    out = args.out or RESULTS_DIR / started.strftime("%Y%m%dT%H%M%SZ")
    print(f"{len(cases)} cases x {args.runs} runs, model {config.LLM_PROVIDER}/{config.CHAT_MODEL}, "
          f"judge {llm_judge.judge_model_name() if judge else 'off'}")

    out.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        for run in range(1, args.runs + 1):
            record = run_case(case, run, judge)
            records.append(record)
            failed = [f"{c['name']}" for t in record["turns"] for c in t["checks"] if c["passed"] is False]
            seconds = sum(t["seconds"] for t in record["turns"])
            outcome = ("ERROR (fallback reply, not scored)" if record["errored"]
                       else "pass" if record["passed"] else "FAIL " + ", ".join(failed))
            print(f"  {case.id} run {run}: {outcome} ({seconds:.1f} s)", flush=True)
            # Every finished run is kept even if the evaluation is stopped part way (e.g. by throttling)
            (out / "results.partial.json").write_text(json.dumps({"runs": records}, indent=1, ensure_ascii=False,
                                                                 default=str), encoding="utf-8")

    meta = {"date": started.strftime("%Y-%m-%d %H:%M UTC"), "provider": config.LLM_PROVIDER,
            "chat_model": config.CHAT_MODEL, "embedding_model": config.EMBEDDING_MODEL,
            "judge_model": llm_judge.judge_model_name() if judge else None,
            "cases": len(cases), "turns": sum(len(c.turns) for c in cases), "runs_per_case": args.runs,
            "commit": git_commit(), "cases_file": str(args.cases)}
    summary = summarize(cases, records)
    (out / "results.partial.json").unlink(missing_ok=True)
    (out / "results.json").write_text(json.dumps({"meta": meta, "summary": summary, "runs": records},
                                                 indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "report.md").write_text(render_report(meta, summary, records), encoding="utf-8")
    errored = f", {summary['errored_runs']} errored and not scored" if summary["errored_runs"] else ""
    print(f"\n{summary['runs_passed']} of {summary['runs']} runs passed ({_pct(summary['runs_passed'], summary['runs'])})"
          f"{errored}. Report: {out / 'report.md'}")
    if args.fail_under is not None and ((summary["pass_rate"] or 0) < args.fail_under or summary["errored_runs"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
