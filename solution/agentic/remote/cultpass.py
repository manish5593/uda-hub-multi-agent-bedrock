"""Remote versions of the CultPass data functions in agentic/tools/cultpass_tools.py (CultPass service)."""
from typing import Optional
from urllib.parse import quote

from agentic.remote.http import call


def _id(value: str) -> str:
    return quote(str(value), safe="")


def get_customer_profile(user_id: str) -> Optional[dict]:
    return call("cultpass", "GET", f"/customers/{_id(user_id)}", missing_ok=True)


def find_customer_by_email(email: str) -> Optional[dict]:
    return call("cultpass", "GET", "/customers", params={"email": email}, missing_ok=True)


def list_reservations(user_id: str) -> list[dict]:
    return call("cultpass", "GET", f"/customers/{_id(user_id)}/reservations")


def experience_titles() -> list[str]:
    return call("cultpass", "GET", "/experiences/titles")


def search_experiences(query: str = "", location: str = "", only_available: bool = True) -> list[dict]:
    return call("cultpass", "GET", "/experiences",
                params={"query": query, "location": location, "only_available": only_available})


def reserve_experience(user_id: str, experience_id: str) -> dict:
    return call("cultpass", "POST", f"/customers/{_id(user_id)}/reservations", json={"experience_id": experience_id})


def cancel_reservation(user_id: str, reservation_id: str) -> dict:
    return call("cultpass", "POST", f"/customers/{_id(user_id)}/reservations/{_id(reservation_id)}/cancel")


def manage_subscription(user_id: str, action: str) -> dict:
    return call("cultpass", "POST", f"/customers/{_id(user_id)}/subscription/{_id(action)}")


def change_reservation(user_id: str, reservation_id: str, experience_id: str) -> dict:
    return call("cultpass", "POST", f"/customers/{_id(user_id)}/reservations/{_id(reservation_id)}/change",
                json={"experience_id": experience_id})
