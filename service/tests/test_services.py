"""The CultPass, Ticket and Knowledge service APIs, and the agentic functions that call them."""
from agentic.tools import cultpass_tools as cp
from agentic.tools import ticket_tools as tt
from agentic.tools.knowledge_tools import get_retriever

BOB, ALICE = "f556c0", "a4ab87"


# --- Internal authentication -------------------------------------------------

def test_internal_token_is_required(cultpass_client):
    assert cultpass_client.get(f"/customers/{BOB}", headers={"X-Internal-Token": "wrong"}).status_code == 401
    cultpass_client.headers.pop("X-Internal-Token")
    assert cultpass_client.get(f"/customers/{BOB}").status_code == 401
    # Health and metrics stay open for the orchestrator and Prometheus
    assert cultpass_client.get("/health").json() == {"status": "ok", "service": "cultpass"}
    assert cultpass_client.get("/metrics").status_code == 200


def test_services_refuse_requests_when_no_token_is_configured(cultpass_client, monkeypatch):
    monkeypatch.delenv("INTERNAL_API_TOKEN")
    assert cultpass_client.get(f"/customers/{BOB}").status_code == 503


def test_request_id_is_echoed(cultpass_client):
    response = cultpass_client.get(f"/customers/{BOB}", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"


# --- CultPass service -------------------------------------------------------

def test_customer_lookup(cultpass_client):
    assert cultpass_client.get(f"/customers/{BOB}").json()["name"] == "Bob Stone"
    assert cultpass_client.get("/customers", params={"email": "BOB.STONE@granite.com"}).json()["user_id"] == BOB
    assert cultpass_client.get("/customers/nobody").status_code == 404
    assert cultpass_client.get("/customers", params={"email": "nobody@example.com"}).status_code == 404


def test_booking_rules_are_enforced_by_the_service(cultpass_client):
    experiences = cultpass_client.get("/experiences").json()
    assert experiences, "conftest moves the experiences into the future"
    standard = next(e for e in experiences if not e["is_premium"])

    booked = cultpass_client.post(f"/customers/{BOB}/reservations", json={"experience_id": standard["experience_id"]}).json()
    assert booked["success"] is True
    again = cultpass_client.post(f"/customers/{BOB}/reservations", json={"experience_id": standard["experience_id"]}).json()
    assert again == {"success": False, "error": f"Already reserved (reservation {booked['reservation_id']})."}

    # Alice's account is blocked
    blocked = cultpass_client.post(f"/customers/{ALICE}/reservations", json={"experience_id": standard["experience_id"]}).json()
    assert blocked["success"] is False and "blocked" in blocked["error"]

    cancelled = cultpass_client.post(f"/customers/{BOB}/reservations/{booked['reservation_id']}/cancel").json()
    assert cancelled["success"] is True
    assert cultpass_client.post(f"/customers/{BOB}/subscription/explode").json()["success"] is False


def test_cultpass_functions_go_over_http_when_remote(remote_services):
    profile = cp.find_customer_by_email("bob.stone@granite.com")
    assert profile["name"] == "Bob Stone"
    assert cp.get_customer_profile("nobody") is None
    assert cp.experience_titles()
    assert ("cultpass", "GET", "/customers") in remote_services
    assert ("cultpass", "GET", "/experiences/titles") in remote_services


def test_account_tools_work_through_the_service(remote_services):
    tools = {t.name: t for t in cp.build_account_tools(BOB, "t-remote")}
    experiences = tools["find_experiences"].invoke({})
    target = next(e for e in experiences if not e["is_premium"] and not e["already_booked"])
    booked = tools["book_experience"].invoke({"experience_id": target["experience_id"]})
    assert booked["success"] is True
    assert any(r["reservation_id"] == booked["reservation_id"] for r in tools["get_my_reservations"].invoke({}))
    # The id guard still runs in the agent, before any HTTP call
    assert tools["cancel_my_reservation"].invoke({"reservation_id": "made-up"})["success"] is False
    assert ("cultpass", "POST", f"/customers/{BOB}/reservations") in remote_services

    # Changing a date is one call to the service, which books the new one and cancels the old one together
    another = next(e for e in experiences if not e["is_premium"] and not e["already_booked"]
                   and e["experience_id"] != target["experience_id"])
    moved = tools["change_reservation_date"].invoke({"reservation_id": booked["reservation_id"],
                                                     "experience_id": another["experience_id"]})
    assert moved["success"] is True and moved["replaced_reservation_id"] == booked["reservation_id"]
    statuses = {r["reservation_id"]: r["status"] for r in tools["get_my_reservations"].invoke({})}
    assert statuses[booked["reservation_id"]] == "cancelled" and statuses[moved["reservation_id"]] == "reserved"
    assert ("cultpass", "POST", f"/customers/{BOB}/reservations/{booked['reservation_id']}/change") in remote_services


# --- Ticket service ---------------------------------------------------------

def test_ticket_lifecycle(ticket_client):
    created = ticket_client.put("/tickets/t-1", json={"metadata": {"channel": "email"}}).json()
    assert created["status"] == "open" and created["external_user_id"] == "guest"
    assert ticket_client.post("/tickets/t-1/messages", json={"role": "user", "content": "hi"}).status_code == 201
    assert ticket_client.get("/tickets/t-1/messages").json() == [{"role": "user", "content": "hi"}]
    updated = ticket_client.patch("/tickets/t-1/metadata", json={"status": "resolved", "tags": "a, b"}).json()
    assert updated["status"] == "resolved" and updated["tags"] == "a, b"
    linked = ticket_client.put("/tickets/t-1/customer", json={"external_user_id": BOB, "user_name": "Bob Stone"}).json()
    assert ticket_client.get("/tickets/t-1").json()["udahub_user_id"] == linked["udahub_user_id"]


def test_ticket_errors(ticket_client):
    assert ticket_client.get("/tickets/missing").status_code == 404
    assert ticket_client.post("/tickets/missing/messages", json={"role": "user", "content": "x"}).status_code == 404
    assert ticket_client.post("/tickets/missing/messages", json={"role": "hacker", "content": "x"}).status_code == 422
    unknown = ticket_client.put("/tickets/t-2", json={"metadata": {"account_id": "nope"}})
    assert unknown.status_code == 422 and "Unknown account" in unknown.json()["detail"]


def test_ticket_functions_go_over_http_when_remote(remote_services):
    ticket = tt.get_or_create_ticket("t-3", {"external_user_id": BOB, "user_name": "Bob Stone"})
    tt.add_message("t-3", "user", "hello")
    assert tt.ticket_exists("t-3") and not tt.ticket_exists("nope")
    assert tt.save_memory("cultpass", ticket["udahub_user_id"], "preference", "Likes opera", "t-3") is True
    assert tt.save_memory("cultpass", ticket["udahub_user_id"], "preference", "likes opera.", "t-3") is False
    assert {"category": "preference", "content": "Likes opera"} in tt.get_memories(ticket["udahub_user_id"])
    assert tt.get_prior_tickets(ticket["udahub_user_id"], exclude_ticket_id="t-3")[0]["ticket_id"] == "past-bob-login"
    # A bad request surfaces as ValueError, like the local function
    try:
        tt.get_or_create_ticket("t-4", {"account_id": "nope"})
    except ValueError as exc:
        assert "Unknown account" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    assert {name for name, _, _ in remote_services} == {"tickets"}


# --- Knowledge service ------------------------------------------------------

def test_knowledge_search(knowledge_client):
    result = knowledge_client.post("/accounts/cultpass/search", json={"query": "I forgot my password"}).json()
    assert result["mode"] == "keyword"
    assert result["articles"][0]["title"] == "How to Handle Login Issues?"
    assert knowledge_client.post("/accounts/cultpass/search", json={"query": "x", "k": 99}).status_code == 422


def test_retriever_is_remote_when_configured(remote_services):
    articles = get_retriever("cultpass").search("I forgot my password")
    assert articles[0]["title"] == "How to Handle Login Issues?"
    assert remote_services == [("knowledge", "POST", "/accounts/cultpass/search")]
