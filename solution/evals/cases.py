"""Labelled evaluation cases: what a ticket says and what the system should do with it.

A case is one conversation (one or more customer messages on the same ticket). Each message can carry
expectations; anything left out is not checked. Where more than one answer is acceptable (for example an
issue the classifier could reasonably call `technical` or `login_access`), give a list.

Labels are validated against the values the code actually uses (routes, issue types, statuses, escalation
teams, tool names), so a typo in cases.json fails when the file is loaded, not silently at scoring time.
"""
import json
from pathlib import Path
from typing import Literal, Optional, Union, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic.agents.escalation_agent import EscalationHandoff
from agentic.state import IssueType, Route

CASES_PATH = Path(__file__).with_name("cases.json")

ROUTES = set(get_args(Route))
ISSUE_TYPES = set(get_args(IssueType))
STATUSES = {"resolved", "awaiting_customer", "escalated", "error"}
TEAMS = set(get_args(EscalationHandoff.model_fields["team"].annotation))
TOOLS = {"get_my_account", "get_my_reservations", "find_experiences", "book_experience", "cancel_my_reservation",
         "change_subscription", "request_refund", "search_knowledge_base", "recommend_experiences",
         "change_reservation_date"}
CATEGORIES = Literal["knowledge", "not_covered", "account", "policy", "routing", "language", "safety"]


def _one_or_many(value, allowed: set, what: str):
    values = [value] if isinstance(value, str) else value
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; expected one of {sorted(allowed)}")
    return values


class Expect(BaseModel):
    """What should happen after one customer message. Every field is optional."""
    model_config = ConfigDict(extra="forbid")

    issue_type: Optional[Union[str, list[str]]] = None
    route: Optional[Union[str, list[str]]] = None
    status: Optional[Union[str, list[str]]] = None
    team: Optional[Union[str, list[str]]] = Field(None, description="Escalation team, when escalated")
    tools_succeeded: list[str] = Field(default_factory=list, description="Must be called and return success")
    tools_called: list[str] = Field(default_factory=list, description="Must be called (any result)")
    tools_not_called: list[str] = Field(default_factory=list, description="Must not be called at all")
    tools_not_succeeded: list[str] = Field(default_factory=list,
                                           description="May be tried, but must not succeed (e.g. a refused booking)")
    reply_excludes: list[str] = Field(default_factory=list, description="Case-insensitive text the reply must not contain")
    key_points: list[str] = Field(default_factory=list, description="Points the reply must make (LLM judge)")
    faithful: bool = Field(True, description="Judge the reply against the evidence when an agent's draft is sent")

    @field_validator("issue_type")
    @classmethod
    def _issue_type(cls, v):
        return v if v is None else _one_or_many(v, ISSUE_TYPES, "issue type")

    @field_validator("route")
    @classmethod
    def _route(cls, v):
        return v if v is None else _one_or_many(v, ROUTES, "route")

    @field_validator("status")
    @classmethod
    def _status(cls, v):
        return v if v is None else _one_or_many(v, STATUSES, "status")

    @field_validator("team")
    @classmethod
    def _team(cls, v):
        return v if v is None else _one_or_many(v, TEAMS, "team")

    @field_validator("tools_succeeded", "tools_called", "tools_not_called", "tools_not_succeeded")
    @classmethod
    def _tools(cls, v):
        return _one_or_many(v, TOOLS, "tool")


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    metadata: Optional[dict] = None
    expect: Expect = Field(default_factory=Expect)


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    category: CATEGORIES
    description: str
    turns: list[Turn] = Field(min_length=1)


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    cases = [Case.model_validate(c) for c in json.loads(Path(path).read_text(encoding="utf-8"))]
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate case ids: {duplicates}")
    return cases
