import pytest
"""Tests for the deterministic building blocks: tools, routing, retrieval and escalation rules."""
from agentic.agents.quality_gate import should_escalate
from agentic.agents.supervisor import route_ticket
from agentic.tools import cultpass_tools as cp
from agentic.tools import ticket_tools as tt
from agentic.tools.knowledge_tools import KnowledgeRetriever

BOB, ALICE, EVA = "f556c0", "a4ab87", "f1f10d"
GOOD = {"issue_type": "general_info", "urgency": "low", "confidence": 0.9,
        "requires_account_action": False, "human_requested": False}
KNOWN = {"identified": True, "is_blocked": False}


# --- CultPass tools ---------------------------------------------------------

def test_profile_and_lookup_by_email():
    profile = cp.get_customer_profile(BOB)
    assert profile["subscription"]["tier"] == "premium"
    assert profile["subscription"]["remaining_this_month"] == 4  # quota 6, two reservations
    assert cp.find_customer_by_email("BOB.STONE@granite.com")["user_id"] == BOB
    assert cp.find_customer_by_email("nobody@example.com") is None


def test_reserve_and_cancel_round_trip():
    exp = cp.search_experiences()[-1]
    booked = cp.reserve_experience(BOB, exp["experience_id"])
    assert booked["success"], booked
    assert cp.reserve_experience(BOB, exp["experience_id"])["success"] is False  # no double booking
    cancelled = cp.cancel_reservation(BOB, booked["reservation_id"])
    assert cancelled["success"] and cancelled["credit_returned"]


def test_reservation_business_rules():
    exp = cp.search_experiences()[0]
    assert "blocked" in cp.reserve_experience(ALICE, exp["experience_id"])["error"]
    assert "not active" in cp.reserve_experience(EVA, exp["experience_id"])["error"]


def test_cannot_cancel_someone_elses_reservation():
    alice_res = cp.list_reservations(ALICE)[0]["reservation_id"]
    assert cp.cancel_reservation(BOB, alice_res)["success"] is False


def test_account_tools_are_bound_to_one_customer():
    tools = {t.name: t for t in cp.build_account_tools(BOB, "t-1")}
    assert "user_id" not in tools["cancel_my_reservation"].args
    assert tools["get_my_account"].invoke({})["user_id"] == BOB


def test_subscription_transitions():
    assert cp.manage_subscription(BOB, "pause")["new_status"] == "paused"
    assert cp.manage_subscription(BOB, "pause")["success"] is False
    assert cp.manage_subscription(BOB, "resume")["new_status"] == "active"


# --- Tickets and memory -----------------------------------------------------

def test_ticket_creation_linking_and_memory():
    ticket = tt.get_or_create_ticket("t-new", {"channel": "email"})
    assert ticket["external_user_id"] == "guest" and ticket["status"] == "open"
    uid = tt.link_ticket_to_customer("t-new", BOB, "Bob Stone")
    assert tt.get_or_create_ticket("t-new")["external_user_id"] == BOB

    tt.add_message("t-new", "user", "hello")
    tt.update_ticket_metadata("t-new", status="resolved", issue_type="general_info", tags="a, b")
    tt.update_ticket_metadata("t-new", tags="b, c")
    info = tt.get_or_create_ticket("t-new")
    assert info["status"] == "resolved" and info["tags"] == "a, b, c"

    # Bob's previous ticket and seeded preference are recalled
    prior = tt.get_prior_tickets(uid, exclude_ticket_id="t-new")
    assert prior[0]["ticket_id"] == "past-bob-login"
    assert any("music" in m["content"] for m in tt.get_memories(uid))
    assert tt.save_memory("cultpass", uid, "preference", "Likes jazz", "t-new") is True
    assert tt.save_memory("cultpass", uid, "preference", "Likes jazz", "t-new") is False


# --- Retrieval --------------------------------------------------------------

def test_keyword_retrieval_finds_relevant_articles(monkeypatch):
    import agentic.tools.knowledge_tools as kt
    monkeypatch.setattr(kt, "get_embeddings", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    retriever = KnowledgeRetriever("cultpass")
    assert retriever.mode == "keyword" and len(retriever.articles) >= 14
    assert retriever.search("I forgot my password and can't log in")[0]["title"] == "How to Handle Login Issues?"
    assert retriever.search("how do refunds work")[0]["title"] == "Refund Policy"
    assert retriever.search("xyzzy quantum") == []


# --- Routing and escalation decisions ----------------------------------------

def test_routing_rules():
    assert route_ticket(GOOD, KNOWN, "open")[0] == "knowledge_agent"
    assert route_ticket({**GOOD, "issue_type": "reservation"}, KNOWN, "open")[0] == "account_agent"
    guest = {"identified": False}
    # A guest asking for an account action is asked who they are first...
    action = {**GOOD, "issue_type": "reservation", "requires_account_action": True}
    assert route_ticket(action, guest, "open")[0] == "request_info"
    # ...but a guest asking how reservations, the subscription or billing work gets the knowledge base's answer
    for issue in ("reservation", "subscription", "billing_refund"):
        assert route_ticket({**GOOD, "issue_type": issue}, guest, "open") == (
            "knowledge_agent", f"{issue} question that needs no account data")
    # ...and so does a guest whose request no account tool could act on, even marked as an account action
    # (Claude marked "I forgot my password", "my QR code won't scan" and "delete my account" that way)
    for issue in ("login_access", "technical", "account_privacy", "general_info"):
        route, reason = route_ticket({**GOOD, "issue_type": issue, "requires_account_action": True}, guest, "open")
        assert route == "knowledge_agent" and "no account tool" in reason
    # An identified customer with the same request still goes to the account agent
    assert route_ticket({**GOOD, "issue_type": "account_privacy", "requires_account_action": True},
                        KNOWN, "open")[0] == "account_agent"
    assert route_ticket({**GOOD, "human_requested": True}, KNOWN, "open")[0] == "escalation_agent"
    assert route_ticket({**GOOD, "urgency": "critical"}, KNOWN, "open")[0] == "escalation_agent"
    assert route_ticket({**GOOD, "confidence": 0.2}, KNOWN, "open")[0] == "escalation_agent"
    assert route_ticket(GOOD, KNOWN, "escalated")[0] == "escalation_agent"
    blocked = {"identified": True, "is_blocked": True}
    assert route_ticket({**GOOD, "issue_type": "login_access"}, blocked, "open")[0] == "escalation_agent"
    assert route_ticket(GOOD, blocked, "open")[0] == "knowledge_agent"  # general questions are still answered


def test_gate_sees_the_account_the_agent_was_given():
    """The account agent gets the customer's profile, quota included, in its context. An answer taken from
    it must count as grounded, so the gate gets the same profile; a guest has none."""
    from agentic.agents.quality_gate import gate_evidence
    sub = {"tier": "premium", "monthly_quota": 2, "used_this_month": 1, "remaining_this_month": 1}
    customer = {"identified": True, "name": "Frank Ocean", "email": "frank.ocean@seawaves.io",
                "is_blocked": False, "subscription": sub}
    evidence = gate_evidence({"customer": customer})
    assert len(evidence) == 1 and evidence[0].startswith("Customer account")
    assert '"remaining_this_month": 1' in evidence[0]
    assert "frank.ocean@seawaves.io" not in evidence[0]  # only what the answer may rely on
    assert gate_evidence({"customer": {"identified": False}}) == []


def test_a_reply_without_a_question_is_not_awaiting_the_customer(monkeypatch):
    """The gate model marked answered replies that end with "let me know if..." as awaiting the customer,
    which left the ticket open. Code overrides it when the draft asks nothing."""
    from agentic.agents import quality_gate
    from agentic.state import ResolutionAssessment
    assert quality_gate.asks_a_question("Which reservation do you mean?")
    assert quality_gate.asks_a_question("¿Cuál es tu correo")
    assert not quality_gate.asks_a_question("You have 1 left this month. Let me know if you need anything!")

    verdict = ResolutionAssessment(confidence=0.9, grounded=True, answered_by_evidence=True,
                                   awaiting_customer=True, reason="offers more help")

    class Gate:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return verdict

    monkeypatch.setattr(quality_gate, "get_llm", lambda: Gate())
    state = {"messages": [], "trace": [], "ticket_id": "t-gate"}
    answered = quality_gate.assess_resolution({**state, "draft_response": "Done. Let me know if you need more!"})
    assert answered["assessment"]["awaiting_customer"] is False
    asking = quality_gate.assess_resolution({**state, "draft_response": "Which of your two reservations?"})
    assert asking["assessment"]["awaiting_customer"] is True


@pytest.mark.parametrize("policy, steps_failed, escalates", [
    (False, False, False),  # troubleshooting steps offered: the customer tries them first
    (True, False, True),    # refund approval, blocked account, legal...
    (False, True, True),    # the customer already tried the documented steps
])
def test_code_decides_whether_a_human_is_needed(monkeypatch, policy, steps_failed, escalates):
    """The gate model reports two facts; code combines them into needs_human. Asked directly, the model
    escalated troubleshooting answers that offer help if the steps don't work."""
    from agentic.agents import quality_gate
    from agentic.state import ResolutionAssessment
    verdict = ResolutionAssessment(confidence=0.9, grounded=True, answered_by_evidence=True,
                                   policy_requires_human=policy, customer_says_steps_failed=steps_failed,
                                   awaiting_customer=False, reason="checked")

    class Gate:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return verdict

    monkeypatch.setattr(quality_gate, "get_llm", lambda: Gate())
    state = {"messages": [], "trace": [], "ticket_id": "t-human",
             "draft_response": "Update the app, then restart it. If it keeps crashing, we can take a closer look."}
    assessment = quality_gate.assess_resolution(state)["assessment"]
    assert assessment["needs_human"] is escalates
    assert should_escalate({"assessment": assessment})[0] is escalates


def test_reservations_say_whose_they_are():
    """The agent once told Cathy that her own reservation was her friend Bob's."""
    cathy = {t.name: t for t in cp.build_account_tools("88382b", "t-owner")}
    reservations = cathy["get_my_reservations"].invoke({})
    assert reservations and all(r["booked_by"] == "Cathy Bloom (the customer you are talking to)"
                                for r in reservations)


def test_escalation_rules():
    ok = {"confidence": 0.9, "grounded": True, "answered_by_evidence": True, "needs_human": False}
    assert should_escalate({"assessment": ok})[0] is False
    assert should_escalate({"assessment": {**ok, "confidence": 0.5}})[0] is True
    assert should_escalate({"assessment": {**ok, "grounded": False}})[0] is True
    # related articles that don't answer the question
    escalate, reason = should_escalate({"assessment": {**ok, "answered_by_evidence": False}})
    assert escalate and "don't answer" in reason
    # ...unless the draft is a clarifying question
    clarifying = {**ok, "answered_by_evidence": False, "awaiting_customer": True}
    assert should_escalate({"assessment": clarifying})[0] is False
    assert should_escalate({"assessment": {**ok, "needs_human": True}})[0] is True
    refund = [{"tool": "request_refund", "args": {}, "result": {"success": True}}]
    assert should_escalate({"assessment": ok, "actions_taken": refund})[0] is True


def test_memory_filter_drops_account_state():
    from agentic.agents.memory_agent import Memory, is_account_state
    stale = [
        Memory(category="issue_history", content="Requested to cancel a reservation."),
        Memory(category="personal_fact", content="Has an upcoming reservation for Sunset Paddleboarding."),
        Memory(category="personal_fact", content="Booked the Samba Night at Lapa."),
        Memory(category="issue_history", content="Used all of the monthly quota."),
    ]
    kept = [
        Memory(category="issue_history", content="Requested a refund for a double charge on subscription."),
        Memory(category="preference", content="Likes to book concerts and jazz nights."),
        Memory(category="preference", content="Prefers replies in Portuguese."),
        Memory(category="personal_fact", content="Uses a wheelchair; needs accessible venues."),
    ]
    assert all(is_account_state(m) for m in stale)
    assert not any(is_account_state(m) for m in kept)


def test_account_tools_refuse_ids_not_looked_up():
    tools = {t.name: t for t in cp.build_account_tools(BOB, "t-1")}
    reservation_id = cp.list_reservations(BOB)[0]["reservation_id"]
    experience_id = cp.search_experiences()[-1]["experience_id"]

    # Real ids, but this turn hasn't looked them up yet
    assert "get_my_reservations first" in tools["cancel_my_reservation"].invoke({"reservation_id": reservation_id})["error"]
    assert "find_experiences first" in tools["book_experience"].invoke({"experience_id": experience_id})["error"]

    tools["find_experiences"].invoke({})
    booked = tools["book_experience"].invoke({"experience_id": experience_id})
    assert booked["success"], booked
    # a reservation the agent just made can be cancelled without another lookup
    assert tools["cancel_my_reservation"].invoke({"reservation_id": booked["reservation_id"]})["success"]

    tools["get_my_reservations"].invoke({})
    assert tools["cancel_my_reservation"].invoke({"reservation_id": reservation_id})["success"]

    # a fresh turn starts with nothing looked up
    fresh = {t.name: t for t in cp.build_account_tools(BOB, "t-1")}
    assert fresh["cancel_my_reservation"].invoke({"reservation_id": "abc123"})["success"] is False


def test_memory_filter_drops_one_off_human_requests():
    from agentic.agents.memory_agent import Memory, drop_reason
    for text in ["Wants to talk to a real person for assistance.", "Asked to speak with a human agent",
                 "Demanded a live person"]:
        assert drop_reason(Memory(category="personal_fact", content=text)) == "one-off request"
    for text in ["Prefers to be contacted by email", "Reported being charged twice (refund requested)",
                 "Is a human rights lawyer"]:
        assert drop_reason(Memory(category="personal_fact", content=text)) is None


def test_duplicate_memories_ignore_case_and_punctuation():
    uid = tt.get_or_create_ticket("t-dup", {"external_user_id": BOB, "user_name": "Bob Stone"})["udahub_user_id"]
    seeded = "Enjoys music and dance experiences, especially in Rio de Janeiro"
    for variant in [seeded + ".", seeded.upper(), "  enjoys music and dance experiences especially in rio de janeiro!! ", ""]:
        assert tt.save_memory("cultpass", uid, "preference", variant, "t-dup") is False, variant
    assert tt.save_memory("cultpass", uid, "preference", "Enjoys jazz in Sao Paulo", "t-dup") is True
    assert sum("music and dance" in m["content"] for m in tt.get_memories(uid)) == 1


def test_repeat_issues_escalate_but_routine_requests_do_not():
    from agentic.agents.supervisor import repeat_issue_types
    login = {**GOOD, "issue_type": "login_access"}
    assert route_ticket(login, KNOWN, "open", frozenset({"login_access"}))[0] == "escalation_agent"
    assert route_ticket(login, KNOWN, "open", frozenset({"billing_refund"}))[0] == "knowledge_agent"
    # booking again is normal, not a failed fix
    booking = {**GOOD, "issue_type": "reservation", "requires_account_action": True}
    assert route_ticket(booking, KNOWN, "open", frozenset({"reservation"}))[0] == "account_agent"

    prior = [{"issue_type": "login_access"}, {"issue_type": None}]
    assert repeat_issue_types({"prior_tickets": prior}) == {"login_access"}
    # an earlier answer on this ticket counts only if the ticket was marked resolved
    assert repeat_issue_types({"ticket_status": "resolved", "ticket_issue_type": "technical"}) == {"technical"}
    assert repeat_issue_types({"ticket_status": "awaiting_customer", "ticket_issue_type": "technical"}) == set()


def test_memory_filter_drops_contact_details():
    from agentic.agents.memory_agent import Memory, drop_reason
    for text in ["The customer's email is bob.stone@granite.com.", "Phone number is +55 11 91234-5678"]:
        assert drop_reason(Memory(category="personal_fact", content=text)) == "contact details", text
    # a preference about how to be contacted is a real preference, not a contact detail
    assert drop_reason(Memory(category="preference", content="Prefers to be contacted by email")) is None
    assert drop_reason(Memory(category="issue_history", content="Reported being charged twice in 2026")) is None


def test_memory_filter_drops_specific_experiences():
    from agentic.agents.memory_agent import Memory, drop_reason
    for memory in [Memory(category="personal_fact", content="The customer is interested in Samba Night at Lapa events."),
                   Memory(category="preference", content="Wants to see the Modern Art at MASP tour")]:
        assert drop_reason(memory) == "account state", memory.content
    # a general taste is still a preference
    assert drop_reason(Memory(category="preference", content="Enjoys samba and live music")) is None


def test_reported_urgency_can_raise_but_not_lower_the_routing_urgency():
    from agentic.agents.supervisor import effective_urgency
    assert effective_urgency("medium", "critical") == "critical"
    assert effective_urgency("critical", "low") == "critical"
    assert effective_urgency("low", None) == "low"
    assert effective_urgency(None, "bogus") == "medium"

    # the source system marks it critical, the classifier only says medium: still escalated
    route, reason = route_ticket({**GOOD, "urgency": "medium"}, KNOWN, "open", reported_urgency="critical")
    assert route == "escalation_agent" and "metadata" in reason
    # high reported urgency alone doesn't escalate
    assert route_ticket({**GOOD, "urgency": "low"}, KNOWN, "open", reported_urgency="high")[0] == "knowledge_agent"


def test_tools_return_structured_errors_instead_of_raising(monkeypatch):
    tools = {t.name: t for t in cp.build_account_tools(BOB, "t-1")}
    # the schema the LLM sees is unchanged by the error wrapper
    assert tools["change_subscription"].args["action"]["enum"] == ["pause", "resume", "cancel"]
    assert "support-lead approval" in tools["request_refund"].description

    assert tools["request_refund"].invoke({"reason": "  "})["success"] is False

    def broken(*args, **kwargs):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(cp, "list_reservations", broken)
    result = tools["get_my_reservations"].invoke({})
    assert result["success"] is False and "Internal error" in result["error"]


def test_experience_search_marks_what_the_customer_already_booked():
    tools = {t.name: t for t in cp.build_account_tools(BOB, "t-1")}
    results = {e["title"]: e["already_booked"] for e in tools["find_experiences"].invoke({})}
    assert results["Samba Night at Lapa"] is True  # in Bob's seeded reservations
    assert results["Modern Art at MASP"] is False


def test_unknown_id_error_lists_the_valid_ids():
    tools = {t.name: t for t in cp.build_account_tools(BOB, "t-1")}
    reservations = tools["get_my_reservations"].invoke({})
    # the agent passing a list position instead of an id
    error = tools["cancel_my_reservation"].invoke({"reservation_id": "0"})["error"]
    assert "not a position" in error
    assert all(r["reservation_id"] in error for r in reservations)


def test_model_provider_is_a_configuration_switch(monkeypatch):
    from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
    from langchain_openai import ChatOpenAI
    from agentic import config, llm
    from agentic.sagemaker_embeddings import SageMakerEndpointEmbeddings
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")  # no AWS call is made: clients are only built
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    try:
        monkeypatch.setattr(config, "LLM_PROVIDER", "bedrock")
        monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bedrock")
        monkeypatch.setattr(config, "CHAT_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        monkeypatch.setattr(config, "EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
        llm.get_llm.cache_clear(); llm.get_embeddings.cache_clear()
        assert isinstance(llm.get_llm(), ChatBedrockConverse)
        assert isinstance(llm.get_embeddings(), BedrockEmbeddings)

        # Embeddings can come from a SageMaker endpoint while chat stays on Bedrock
        monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "sagemaker")
        llm.get_embeddings.cache_clear()
        embeddings = llm.get_embeddings()
        assert isinstance(embeddings, SageMakerEndpointEmbeddings)
        assert embeddings.endpoint_name == config.SAGEMAKER_ENDPOINT

        monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
        monkeypatch.setattr(config, "CHAT_MODEL", "gpt-4o-mini")
        llm.get_llm.cache_clear()
        assert isinstance(llm.get_llm(), ChatOpenAI)
    finally:
        llm.get_llm.cache_clear(); llm.get_embeddings.cache_clear()


class _FakeSageMakerRuntime:
    """Stands in for boto3's sagemaker-runtime client: returns one 2-d vector per input and records requests."""
    def __init__(self):
        self.requests = []

    def invoke_endpoint(self, **request):
        import io, json
        self.requests.append(request)
        inputs = json.loads(request["Body"])["inputs"]
        return {"Body": io.BytesIO(json.dumps([[float(len(t)), 1.0] for t in inputs]).encode())}


def test_sagemaker_embeddings_batch_requests_to_the_endpoint():
    import json
    from agentic.sagemaker_embeddings import SageMakerEndpointEmbeddings
    runtime = _FakeSageMakerRuntime()
    embeddings = SageMakerEndpointEmbeddings("kb-embeddings", region_name="us-east-1", batch_size=2, client=runtime)

    vectors = embeddings.embed_documents(["a", "bb", "ccc"])
    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]  # order kept across batches
    assert [json.loads(r["Body"])["inputs"] for r in runtime.requests] == [["a", "bb"], ["ccc"]]
    assert all(r["EndpointName"] == "kb-embeddings" for r in runtime.requests)
    assert embeddings.embed_query("dddd") == [4.0, 1.0]
    assert embeddings.embed_documents([]) == []


def test_sagemaker_deploy_creates_model_config_and_endpoint(monkeypatch):
    from deploy import sagemaker_embeddings as deploy
    calls = []

    class FakeSageMaker:
        def __getattr__(self, name):
            return lambda **kwargs: calls.append((name, kwargs))

        def get_waiter(self, name):
            return type("Waiter", (), {"wait": lambda _, **kwargs: calls.append(("wait", kwargs))})()

    monkeypatch.setattr(deploy, "sagemaker_client", lambda region: FakeSageMaker())
    deploy.main(["deploy", "--endpoint", "kb", "--region", "us-east-1", "--model", "BAAI/bge-small-en-v1.5",
                 "--role-arn", "arn:aws:iam::123456789012:role/sm"])

    assert [name for name, _ in calls] == ["create_model", "create_endpoint_config", "create_endpoint", "wait"]
    model = calls[0][1]
    assert model["ModelName"] == "kb-model"
    assert model["PrimaryContainer"]["Image"].startswith("683313688378.dkr.ecr.us-east-1.amazonaws.com/tei-cpu:")
    assert model["PrimaryContainer"]["Environment"] == {"HF_MODEL_ID": "BAAI/bge-small-en-v1.5"}
    assert calls[1][1]["ProductionVariants"][0]["ModelName"] == "kb-model"
    assert calls[2][1] == {"EndpointName": "kb", "EndpointConfigName": "kb-config"}


def test_booking_blocker_agrees_with_reserve_experience(tmp_path, monkeypatch):
    """can_book in search results must say exactly what a booking would answer, for every demo customer and
    experience. Each attempt runs on fresh databases, so one booking can't change the next answer."""
    from agentic import config
    from agentic.demo_data import copy_demo_databases
    from agentic.tools import cultpass_tools as ct

    users = [u["user_id"] for u in map(ct.get_customer_profile, ["a4ab87", "f556c0", "88382b", "888fb2",
                                                                    "f1f10d", "e6376d"])]
    experiences = [e["experience_id"] for e in ct.search_experiences(only_available=False)]
    checked = {True: 0, False: 0}
    for user_id in users:
        for experience_id in experiences:
            udahub_db, cultpass_db = copy_demo_databases(tmp_path / f"{user_id}-{experience_id}")
            monkeypatch.setattr(config, "UDAHUB_DB_PATH", udahub_db)
            monkeypatch.setattr(config, "CULTPASS_DB_PATH", cultpass_db)
            exp = next(e for e in ct.search_experiences(only_available=False) if e["experience_id"] == experience_id)
            booked = any(r["experience_id"] == experience_id and r["status"] == "reserved"
                         for r in ct.list_reservations(user_id))
            blocker = ct.booking_blocker(ct.get_customer_profile(user_id), exp, booked)
            result = ct.reserve_experience(user_id, experience_id)
            assert result["success"] == (blocker is None), (user_id, exp["title"], blocker, result)
            checked[result["success"]] += 1
    # The demo data covers both answers (blocked, basic tier, quota and duplicates on one side)
    assert checked[True] > 0 and checked[False] > 0


def test_search_results_say_whether_the_customer_can_book():
    from agentic.tools import cultpass_tools as ct
    cathy = {t.name: t for t in ct.build_account_tools("88382b", "t-can-book")}  # basic tier
    results = cathy["find_experiences"].invoke({})
    premium = [e for e in results if e["is_premium"]]
    assert premium and all(not e["can_book"] and "premium" in e["cannot_book_because"] for e in premium)
    assert any(e["can_book"] and "cannot_book_because" not in e for e in results)


def test_wrong_ids_in_a_draft_are_corrected():
    """The model once wrote 23e89 for a 6-character reservation id."""
    from agentic.agents.account_agent import correct_ids
    booking = [{"tool": "find_experiences", "args": {}, "result": [{"experience_id": "b0e4dd"}]},
               {"tool": "book_experience", "args": {"experience_id": "b0e4dd"},
                "result": {"success": True, "reservation_id": "23e89a"}}]
    draft, fixed = correct_ids("Booked! Your reservation ID is 23e89.", booking)
    assert draft == "Booked! Your reservation ID is 23e89a." and fixed == ["23e89 -> 23e89a"]
    # Right ids, words that look like hex, and years are left alone
    same = "Reservation 23e89a for b0e4dd at the cafe, 2026, decade ahead."
    assert correct_ids(same, booking) == (same, [])
    # No booking this turn: nothing to correct it to, so the draft is unchanged
    lookup = booking[:1]
    assert correct_ids("Your reservation ID is 23e89.", lookup) == ("Your reservation ID is 23e89.", [])


class _ScriptedMatcher:
    def __init__(self, matches):
        self.matches = matches

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        from agentic.tools.recommendations import Match, Matches
        return Matches(matches=[Match(**m) for m in self.matches])


def test_recommendations_only_return_bookable_matches(monkeypatch):
    from agentic.tools import recommendations
    bob_prefs = ["(preference) Enjoys music and dance experiences, especially in Rio de Janeiro",
                 "(issue_history) Reported a login problem."]
    tools = {t.name: t for t in cp.build_account_tools(BOB, "t-rec", bob_prefs)}
    catalog = {e["title"]: e for e in tools["find_experiences"].invoke({})}
    # Bob has Samba Night and the Carnival tour booked already
    samba, carnival, christ, walk = (catalog[t]["experience_id"] for t in (
        "Samba Night at Lapa", "Carnival History Tour in Olinda", "Christ the Redeemer Experience",
        "Pelourinho Colonial Walk"))
    preference = "Enjoys music and dance experiences, especially in Rio de Janeiro"
    monkeypatch.setattr(recommendations, "get_llm", lambda: _ScriptedMatcher([
        {"experience_id": walk, "preference": preference, "why": "street music in Salvador"},
        {"experience_id": samba, "preference": preference, "why": "samba"},          # already booked: not a candidate
        {"experience_id": carnival, "preference": preference, "why": "carnival"},    # already booked: not a candidate
        {"experience_id": christ, "preference": "Loves sightseeing", "why": "sight"},  # not a stored preference
        {"experience_id": "zzz999", "preference": preference, "why": "made up"},      # not an experience
    ]))
    result = tools["recommend_experiences"].invoke({})
    assert result["preferences"] == [preference]
    assert [m["experience_id"] for m in result["matches"]] == [walk]
    assert result["matches"][0]["fits_preference"] == preference and "note" not in result

    # Nothing fits: the agent is told so, with nothing else to offer as a fit
    monkeypatch.setattr(recommendations, "get_llm", lambda: _ScriptedMatcher([]))
    empty = tools["recommend_experiences"].invoke({})
    assert empty["matches"] == [] and "None of the experiences" in empty["note"]
    assert 0 < len(empty["other_options"]) <= 3
    assert all(o["can_book"] and o["fits_preference"] is None for o in empty["other_options"])
    assert samba not in {o["experience_id"] for o in empty["other_options"]}


def test_recommendations_without_preferences_list_bookable_options():
    tools = {t.name: t for t in cp.build_account_tools(BOB, "t-rec", [])}
    result = tools["recommend_experiences"].invoke({})
    assert result["preferences"] == [] and "without saying they fit" in result["note"]
    assert result["experiences"] and all(e["can_book"] for e in result["experiences"])


def test_recommendations_use_interests_stated_in_the_conversation(monkeypatch):
    """A preference the customer states in this message is only stored after the reply, so the agent passes it
    in: without it, Frank's first "I love outdoor activities" got the unfiltered catalog."""
    from agentic.tools import recommendations
    tools = {t.name: t for t in cp.build_account_tools(BOB, "t-rec", [])}
    walk = next(e["experience_id"] for e in tools["find_experiences"].invoke({})
                if e["title"] == "Pelourinho Colonial Walk")
    monkeypatch.setattr(recommendations, "get_llm", lambda: _ScriptedMatcher([
        {"experience_id": walk, "preference": "historic walking tours", "why": "a walk"},
        {"experience_id": walk, "preference": "made-up interest", "why": "not stated"},
    ]))
    result = tools["recommend_experiences"].invoke({"interests": " historic walking tours "})
    assert result["preferences"] == ["historic walking tours"]
    assert [(m["experience_id"], m["fits_preference"]) for m in result["matches"]] == [(walk, "historic walking tours")]


def test_replies_made_of_content_blocks_are_read_as_text():
    """Bedrock's Converse API can return a reply as a list of blocks, which crashed the agents."""
    from langchain_core.messages import AIMessage
    from agentic.agents.common import text_of
    blocks = AIMessage(content=[{"type": "text", "text": "Your booking "}, {"type": "tool_use", "id": "t1"},
                                {"type": "text", "text": "is confirmed."}])
    assert text_of(blocks) == "Your booking is confirmed."
    assert text_of(AIMessage(content="plain")) == "plain"


class _FakeSES:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send_email(self, **kwargs):
        if self.fail:
            raise RuntimeError("MessageRejected")
        self.sent.append(kwargs)


def test_codes_are_emailed_with_ses_and_never_logged(monkeypatch, caplog):
    from agentic import mailer
    ses = _FakeSES()
    monkeypatch.setattr(mailer, "_ses", lambda: ses)
    monkeypatch.setenv("UDAHUB_EMAIL_PROVIDER", "ses")
    monkeypatch.setenv("UDAHUB_EMAIL_FROM", "support@example.com")
    with caplog.at_level("INFO"):
        assert mailer.send_code("bob.stone@granite.com", "482913", "login") is True
    email = ses.sent[0]
    assert email["FromEmailAddress"] == "support@example.com"
    assert email["Destination"] == {"ToAddresses": ["bob.stone@granite.com"]}
    assert "login code" in email["Content"]["Simple"]["Subject"]["Data"]
    assert "482913" in email["Content"]["Simple"]["Body"]["Text"]["Data"]
    assert "482913" not in caplog.text


def test_a_failed_email_falls_back_to_the_log_only_when_allowed(monkeypatch, caplog):
    from agentic import mailer
    monkeypatch.setattr(mailer, "_ses", lambda: _FakeSES(fail=True))
    monkeypatch.setenv("UDAHUB_EMAIL_PROVIDER", "ses")
    monkeypatch.setenv("UDAHUB_EMAIL_FROM", "support@example.com")
    with caplog.at_level("INFO"):
        assert mailer.send_code("bob.stone@granite.com", "482913", "chat") is False
    assert "482913" in caplog.text and "MessageRejected" in caplog.text   # demo: logged so the demo goes on
    caplog.clear()
    monkeypatch.setenv("UDAHUB_EMAIL_LOG_FALLBACK", "false")
    with caplog.at_level("INFO"):
        mailer.send_code("bob.stone@granite.com", "482913", "chat")
    assert "482913" not in caplog.text                                        # production: never logged


def test_codes_are_logged_by_default(monkeypatch, caplog):
    from agentic import mailer
    monkeypatch.delenv("UDAHUB_EMAIL_PROVIDER", raising=False)
    with caplog.at_level("INFO"):
        assert mailer.send_code("bob.stone@granite.com", "482913") is False
    assert "Verification code for bob.stone@granite.com: 482913" in caplog.text
