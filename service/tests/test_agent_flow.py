"""End to end: client -> gateway -> agent service -> CultPass, Ticket and Knowledge services.

Only the LLM is scripted. Routing, tools, persistence and memory all run for real, and every data access
from the agent goes over HTTP to the service that owns the data.
"""
import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from agentic.agents import (account_agent, classifier, escalation_agent, knowledge_agent, memory_agent,
                            quality_gate, supervisor)
from agentic.agents.escalation_agent import EscalationHandoff
from agentic.agents.memory_agent import Memory, MemoryExtraction
from agentic.state import ResolutionAssessment, TicketClassification
from agentic.tools import ticket_tools

API_KEY = "test-api-key"


def classification(**kw):
    base = dict(issue_type="general_info", urgency="low", sentiment="neutral", summary="question",
                requires_account_action=False, human_requested=False, customer_email=None,
                language="English", confidence=0.9)
    return TicketClassification(**{**base, **kw})


GOOD = ResolutionAssessment(confidence=0.9, grounded=True, answered_by_evidence=True,
                            awaiting_customer=False, reason="grounded")
HANDOFF = EscalationHandoff(team="trust_and_safety", priority="high", summary="blocked account",
                            steps_taken=[], recommended_next_step="review", customer_message="A specialist will email you.")


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = outputs

    def invoke(self, messages):
        return AIMessage(content="Tap 'Forgot Password' on the login screen.")

    def with_structured_output(self, schema):
        outputs = self.outputs

        class _Runner:
            def invoke(self, messages):
                return outputs[schema]
        return _Runner()


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM({TicketClassification: classification(), ResolutionAssessment: GOOD,
                    EscalationHandoff: HANDOFF, MemoryExtraction: MemoryExtraction(memories=[])})
    for module in (classifier, knowledge_agent, quality_gate, escalation_agent, memory_agent, account_agent, supervisor):
        monkeypatch.setattr(module, "get_llm", lambda *a, **k: fake)
    return fake


@pytest.fixture
def agent_client(remote_services, llm, monkeypatch):
    from service.agent_service import main
    monkeypatch.setattr(main, "STATE_DB_URL", None)
    with TestClient(main.app, headers={"X-Internal-Token": "test-internal-token"}) as client:
        yield client


@pytest.fixture
def gateway(agent_client, ticket_client, auth_client, monkeypatch):
    """The gateway, with its upstream calls routed to the in-process services."""
    from service.gateway import main
    monkeypatch.setenv("GATEWAY_API_KEYS", f"other-key, {API_KEY}")
    monkeypatch.setenv("GATEWAY_RATE_LIMIT_PER_MINUTE", "5")
    main.limiter.hits.clear()
    apps = {"agent": agent_client.app, "tickets": ticket_client.app, "auth": auth_client.app}
    monkeypatch.setattr(main, "make_client", lambda name: httpx.AsyncClient(
        transport=httpx.ASGITransport(app=apps[name]), base_url=f"http://{name}"))
    with TestClient(main.app, headers={"X-API-Key": API_KEY}) as client:
        yield client


def test_agent_service_resolves_a_knowledge_question(agent_client, llm, remote_services):
    llm.outputs[TicketClassification] = classification(issue_type="login_access", summary="cannot log in")
    turn = agent_client.post("/tickets/t-kb/messages", json={"text": "I forgot my password"}).json()

    assert turn["status"] == "resolved"
    assert turn["route"] == "knowledge_agent"
    assert "Forgot Password" in turn["reply"]
    # Tickets and the knowledge base were reached over HTTP (a guest's how-to question needs no CultPass data)
    assert {name for name, _, _ in remote_services} == {"tickets", "knowledge"}
    assert ("knowledge", "POST", "/accounts/cultpass/search") in remote_services
    assert [m["role"] for m in ticket_tools.get_ticket_messages("t-kb")] == ["user", "ai"]


def test_conversation_continues_across_turns_and_identifies_the_customer(agent_client, llm, monkeypatch):
    llm.outputs[TicketClassification] = classification(issue_type="reservation", requires_account_action=True)
    first = agent_client.post("/tickets/t-guest/messages", json={"text": "Cancel my reservation"}).json()
    assert first["status"] == "awaiting_customer" and first["route"] == "request_info"

    llm.outputs[TicketClassification] = classification(issue_type="reservation", requires_account_action=True,
                                                       customer_email="bob.stone@granite.com")

    def fake_agent(*args, tools, **kwargs):
        class _Agent:
            def invoke(self, payload, config=None):
                by_name = {t.name: t for t in tools}
                reservations = by_name["get_my_reservations"].invoke({})  # a real call to the CultPass service
                call = {"id": "c1", "name": "get_my_reservations", "args": {}}
                return {"messages": [*payload["messages"], AIMessage(content="", tool_calls=[call]),
                                     ToolMessage(content=str(reservations), tool_call_id="c1"),
                                     AIMessage(content="Which reservation should I cancel?")]}
        return _Agent()
    monkeypatch.setattr(account_agent, "create_agent", fake_agent)
    llm.outputs[ResolutionAssessment] = ResolutionAssessment(
        confidence=0.9, grounded=True, answered_by_evidence=False, awaiting_customer=True,
        reason="clarifying question")
    from agentic import verification
    inbox = {}
    monkeypatch.setattr(verification, "deliver_code", lambda email, code: inbox.__setitem__(email, code))
    # The typed email alone identifies nobody: a code goes to it, and the ticket waits for it
    second = agent_client.post("/tickets/t-guest/messages", json={"text": "It's bob.stone@granite.com"}).json()
    assert second["route"] == "request_info" and second["status"] == "awaiting_customer"
    assert second["tools_used"] == [] and ticket_tools.get_ticket("t-guest")["external_user_id"] != "f556c0"

    third = agent_client.post("/tickets/t-guest/messages", json={"text": inbox["bob.stone@granite.com"]}).json()
    assert third["route"] == "account_agent"
    assert third["tools_used"] == ["get_my_reservations"]
    assert third["status"] == "awaiting_customer"
    assert any("identified customer Bob Stone" in line for line in third["trace"])
    ticket = ticket_tools.get_ticket("t-guest")
    assert ticket["external_user_id"] == "f556c0"


def test_blocked_customer_is_escalated_and_memory_is_written(agent_client, llm):
    llm.outputs[TicketClassification] = classification(issue_type="login_access")
    llm.outputs[MemoryExtraction] = MemoryExtraction(memories=[Memory(category="preference", content="Prefers email contact")])
    turn = agent_client.post("/tickets/t-blocked/messages",
                             json={"text": "I can't log in", "metadata": {"external_user_id": "a4ab87"}}).json()

    assert turn["status"] == "escalated"
    assert turn["escalation"] == {"team": "trust_and_safety", "priority": "high",
                                  "reason": "account is blocked: policy requires Trust & Safety review"}
    ticket = ticket_tools.get_ticket("t-blocked")
    assert "Prefers email contact" in [m["content"] for m in ticket_tools.get_memories(ticket["udahub_user_id"])]


def test_agent_outcomes_are_exported_as_metrics(agent_client, llm):
    from prometheus_client import REGISTRY

    def value(name, **labels):
        return REGISTRY.get_sample_value(name, labels) or 0.0

    turns = dict(route="escalation_agent", status="escalated")
    team = dict(team="trust_and_safety", priority="high")
    before = value("udahub_agent_turns_total", **turns), value("udahub_agent_escalations_total", **team)
    llm.outputs[TicketClassification] = classification(issue_type="login_access")
    agent_client.post("/tickets/t-metrics/messages",
                      json={"text": "I can't log in", "metadata": {"external_user_id": "a4ab87"}})

    assert value("udahub_agent_turns_total", **turns) == before[0] + 1
    assert value("udahub_agent_escalations_total", **team) == before[1] + 1
    assert 'udahub_agent_turns_total{route="escalation_agent",status="escalated"}' in agent_client.get("/metrics").text


def test_a_service_outage_still_gets_the_customer_a_reply(agent_client, llm, cultpass_client, monkeypatch):
    """The CultPass service failing mid-turn must not break the request: the workflow's fallback replies."""
    def broken(*args, **kwargs):
        raise httpx.ConnectError("cultpass is down")
    monkeypatch.setattr(cultpass_client, "request", broken)
    turn = agent_client.post("/tickets/t-down/messages",
                             json={"text": "hello", "metadata": {"email": "bob.stone@granite.com"}})
    assert turn.status_code == 200
    assert turn.json()["status"] == "error"
    assert "support team will follow up" in turn.json()["reply"]


def test_agent_service_validates_input(agent_client):
    assert agent_client.post("/tickets/bad id!/messages", json={"text": "hi"}).status_code == 422
    assert agent_client.post("/tickets/t-1/messages", json={"text": "x" * 5000}).status_code == 422
    agent_client.headers.pop("X-Internal-Token")
    assert agent_client.post("/tickets/t-1/messages", json={"text": "hi"}).status_code == 401


def test_agent_refuses_to_start_without_the_service_urls():
    from service.agent_service import main
    with pytest.raises(RuntimeError, match="needs the URLs"):
        main.check_services()


# --- Gateway ----------------------------------------------------------------

def test_gateway_round_trip_hides_internal_details(gateway, llm):
    llm.outputs[TicketClassification] = classification(issue_type="login_access")
    reply = gateway.post("/v1/tickets/t-gw/messages",
                         json={"text": "I can't log in", "metadata": {"external_user_id": "a4ab87"}})
    assert reply.status_code == 200
    assert reply.json() == {"ticket_id": "t-gw", "reply": HANDOFF.customer_message, "status": "escalated",
                            "escalated_to": "trust_and_safety"}

    assert gateway.get("/v1/tickets/t-gw").json() == {"ticket_id": "t-gw", "status": "escalated", "channel": "chat",
                                                       "issue_type": "login_access"}
    messages = gateway.get("/v1/tickets/t-gw/messages").json()
    # The internal escalation handoff note stays inside
    assert [m["role"] for m in messages] == ["user", "ai"]
    assert gateway.get("/v1/tickets/missing").status_code == 404


def test_gateway_requires_an_api_key(gateway):
    assert gateway.get("/v1/tickets/t-1", headers={"X-API-Key": "wrong"}).status_code == 401
    gateway.headers.pop("X-API-Key")
    assert gateway.get("/v1/tickets/t-1").status_code == 401


def test_demo_page_is_served_without_secrets(gateway):
    gateway.headers.pop("X-API-Key")
    page = gateway.get("/demo")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert "/v1/tickets/" in page.text
    # The page never embeds a key: the person using it types one in
    for key in [API_KEY, "other-key"]:
        assert key not in page.text
    # Customers can log in on the page with both steps, and then chat with their own token
    for path in ("/v1/auth/login", "/v1/auth/login/verify", "Authorization"):
        assert path in page.text


def test_gateway_rate_limits_each_key(gateway):
    statuses = [gateway.get("/v1/tickets/missing").status_code for _ in range(6)]
    assert statuses == [404] * 5 + [429]
    # Another key has its own budget
    assert gateway.get("/v1/tickets/missing", headers={"X-API-Key": "other-key"}).status_code == 404


def test_gateway_readiness(gateway):
    assert gateway.get("/ready").json() == {"ready": True,
                                            "services": {"agent": True, "tickets": True, "auth": True}}


# --- Customer logins through the gateway ------------------------------------------------------------

BOB, CATHY = "bob.stone@granite.com", "cathy.bloom@florals.org"


def customer_token(gateway, codes, email, password="a-good-password-1"):
    """Sign up and log in (password, then the emailed code) through the gateway, as a customer with no API key,
    and return the Authorization header."""
    api_key = gateway.headers.pop("X-API-Key", None)
    assert gateway.post("/v1/auth/register/start", json={"email": email}).status_code == 202
    created = gateway.post("/v1/auth/register/complete",
                           json={"email": email, "code": codes[email], "password": password})
    assert created.status_code == 201, created.text
    codes.pop(email, None)
    first = gateway.post("/v1/auth/login", json={"email": email, "password": password})
    # The message points to the gateway's path, not the auth service's
    assert first.status_code == 202 and "/v1/auth/login/verify within 10 minutes" in first.json()["detail"]
    login = gateway.post("/v1/auth/login/verify", json={"email": email, "code": codes[email]})
    assert login.status_code == 200, login.text
    if api_key:
        gateway.headers["X-API-Key"] = api_key
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_a_logged_in_customer_is_identified_by_their_token(gateway, llm, codes):
    bob = customer_token(gateway, codes, BOB)
    gateway.headers.pop("X-API-Key")
    # Metadata claiming to be someone else is ignored: the token decides who this is
    reply = gateway.post("/v1/tickets/t-bob/messages", headers=bob,
                         json={"text": "Hi", "metadata": {"email": CATHY, "external_user_id": "x"}})
    assert reply.status_code == 200
    ticket = ticket_tools.get_ticket("t-bob")
    assert ticket["user_name"] == "Bob Stone" and ticket["external_user_id"] != "x"

    assert gateway.get("/v1/tickets/t-bob", headers=bob).status_code == 200
    assert [m["role"] for m in gateway.get("/v1/tickets/t-bob/messages", headers=bob).json()] == ["user", "ai"]
    # And the conversation continues on the same ticket
    assert gateway.post("/v1/tickets/t-bob/messages", headers=bob, json={"text": "Thanks"}).status_code == 200


def test_customers_only_see_their_own_tickets(gateway, llm, codes):
    bob = customer_token(gateway, codes, BOB)
    cathy = customer_token(gateway, codes, CATHY)
    gateway.headers.pop("X-API-Key")
    gateway.post("/v1/tickets/t-bob/messages", headers=bob, json={"text": "Hi"})

    # Someone else's ticket looks exactly like a missing one
    for request in (lambda: gateway.get("/v1/tickets/t-bob", headers=cathy),
                    lambda: gateway.get("/v1/tickets/t-bob/messages", headers=cathy),
                    lambda: gateway.post("/v1/tickets/t-bob/messages", headers=cathy, json={"text": "From Cathy"})):
        response = request()
        assert response.status_code == 404 and response.json() == {"detail": "Ticket not found"}
    assert all(m["content"] != "From Cathy" for m in ticket_tools.get_ticket_messages("t-bob"))


def test_a_customer_cannot_take_over_a_channel_ticket(gateway, llm, codes):
    # A guest conversation started by a channel integration
    gateway.post("/v1/tickets/t-guest/messages", json={"text": "Hi"})
    bob = customer_token(gateway, codes, BOB)
    gateway.headers.pop("X-API-Key")
    assert gateway.get("/v1/tickets/t-guest", headers=bob).status_code == 404
    assert gateway.post("/v1/tickets/t-guest/messages", headers=bob, json={"text": "Hi"}).status_code == 404


def test_api_keys_still_work_for_customer_tickets(gateway, llm, codes):
    bob = customer_token(gateway, codes, BOB)
    gateway.post("/v1/tickets/t-bob/messages", headers={**bob, "X-API-Key": ""}, json={"text": "Hi"})
    # A trusted channel integration sees every ticket
    assert gateway.get("/v1/tickets/t-bob").status_code == 200


@pytest.mark.parametrize("token", ["not-a-jwt", "a.b.c"])
def test_gateway_rejects_bad_tokens(gateway, token):
    gateway.headers.pop("X-API-Key")
    response = gateway.get("/v1/tickets/t-1", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401 and response.headers["WWW-Authenticate"] == "Bearer"


def test_gateway_rejects_an_expired_token(gateway, codes, monkeypatch):
    monkeypatch.setenv("JWT_TTL_MINUTES", "-1")
    bob = customer_token(gateway, codes, BOB)
    gateway.headers.pop("X-API-Key")
    assert gateway.get("/v1/tickets/t-1", headers=bob).status_code == 401


def test_gateway_relays_login_failures(gateway, codes):
    customer_token(gateway, codes, BOB)
    gateway.headers.pop("X-API-Key")
    response = gateway.post("/v1/auth/login", json={"email": BOB, "password": "wrong-password-1"})
    assert response.status_code == 401 and "Incorrect email or password" in response.json()["detail"]
    weak = gateway.post("/v1/auth/register/complete", json={"email": CATHY, "code": "123456", "password": "short"})
    assert weak.status_code == 422
    bad_code = gateway.post("/v1/auth/login/verify", json={"email": BOB, "code": "123456"})
    assert bad_code.status_code == 401 and "Invalid or expired login code" in bad_code.json()["detail"]


def test_sign_up_and_login_are_rate_limited_per_client(gateway, monkeypatch):
    monkeypatch.setenv("GATEWAY_AUTH_RATE_LIMIT_PER_MINUTE", "3")
    statuses = [gateway.post("/v1/auth/login", json={"email": BOB, "password": "x"}).status_code for _ in range(4)]
    assert statuses == [401] * 3 + [429]


def test_customers_are_rate_limited_separately(gateway, llm, codes):
    bob = customer_token(gateway, codes, BOB)
    gateway.headers.pop("X-API-Key")
    statuses = [gateway.get("/v1/tickets/missing", headers=bob).status_code for _ in range(6)]
    assert statuses == [404] * 5 + [429]
