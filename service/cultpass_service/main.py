"""CultPass service: owns the CultPass database (customers, subscriptions, experiences, reservations).

It stands in for CultPass's own system of record. Its business rules (quota, premium tier, blocked accounts,
free slots) are the functions in agentic/tools/cultpass_tools.py, run here against this service's database.
Rule failures such as "Monthly quota already used" are normal results ({"success": false, "error": ...},
HTTP 200): the agent explains them to the customer. Unknown ids are 404.
"""
from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from agentic import config
from agentic.tools import cultpass_tools as cp
from agentic.tools.db import cultpass_engine
from service.common.app import create_app, ping_database, refuse_own_url

refuse_own_url("cultpass", config.CULTPASS_SERVICE_URL)

app = create_app("cultpass", "CultPass service",
                 health_check=lambda: ping_database(cultpass_engine()), owns_data=True)

ID = Field(min_length=1, max_length=64)


def _customer_or_404(user_id: str) -> dict:
    profile = cp.get_customer_profile(user_id)
    if profile is None:
        raise HTTPException(404, "Customer not found")
    return profile


@app.get("/customers")
def find_customer(email: str = Query(min_length=3, max_length=254)):
    profile = cp.find_customer_by_email(email)
    if profile is None:
        raise HTTPException(404, "No customer with that email")
    return profile


@app.get("/customers/{user_id}")
def get_customer(user_id: str):
    return _customer_or_404(user_id)


@app.get("/customers/{user_id}/reservations")
def list_reservations(user_id: str):
    _customer_or_404(user_id)
    return cp.list_reservations(user_id)


class ReservationIn(BaseModel):
    experience_id: str = ID


@app.post("/customers/{user_id}/reservations")
def reserve(user_id: str, body: ReservationIn):
    return cp.reserve_experience(user_id, body.experience_id)


@app.post("/customers/{user_id}/reservations/{reservation_id}/cancel")
def cancel(user_id: str, reservation_id: str):
    return cp.cancel_reservation(user_id, reservation_id)


@app.post("/customers/{user_id}/subscription/{action}")
def change_subscription(user_id: str, action: str):
    # An unknown action is a rule failure returned as data, as in the local function
    return cp.manage_subscription(user_id, action)


@app.get("/experiences")
def search_experiences(query: str = Query("", max_length=200), location: str = Query("", max_length=100),
                       only_available: bool = True):
    return cp.search_experiences(query=query, location=location, only_available=only_available)


@app.get("/experiences/titles")
def experience_titles():
    return cp.experience_titles()
