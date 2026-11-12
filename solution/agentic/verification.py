"""Proving an email typed in the chat belongs to the person typing it.

A guest who types an email isn't trusted with that account straight away: a 6-digit code goes to the email,
and only the right code links the ticket to the customer. Before this, typing someone's email was enough to
read or cancel their bookings. Channels that pass the customer in the ticket metadata are vouched for by their
API key, and customers who log in prove themselves to the auth service; this is for the rest.

The pending check is kept in the ticket's graph state (`pending_verification`), which the checkpointer
persists between messages, so it works the same in the notebooks and in the agent service. Only an HMAC of the
code is stored. The reply is the same whether or not the email has an account, so the chat can't be used to
find out who is a customer: for an unknown email the pending check has no customer, and every code fails.
"""
import hashlib
import hmac
import logging
import os
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import Optional

logger = logging.getLogger("udahub")

CODE_TTL = timedelta(minutes=int(os.getenv("UDAHUB_VERIFY_CODE_TTL_MINUTES", "10")))
CODE_MAX_ATTEMPTS = int(os.getenv("UDAHUB_VERIFY_CODE_MAX_ATTEMPTS", "5"))
RESEND_SECONDS = int(os.getenv("UDAHUB_VERIFY_RESEND_SECONDS", "60"))
# After this many codes on one ticket, a human checks the customer's identity instead
MAX_CODES_PER_TICKET = int(os.getenv("UDAHUB_VERIFY_MAX_CODES", "3"))

# The key for the codes' HMAC: stable across restarts in the services, random per process otherwise
_SECRET = (os.getenv("UDAHUB_VERIFICATION_SECRET") or os.getenv("JWT_SECRET_KEY")
           or os.getenv("INTERNAL_API_TOKEN") or secrets.token_hex(32))

CODE_IN_TEXT = re.compile(r"(?<!\d)(\d{6})(?!\d)")
CODE_ONLY = re.compile(r"\s*\D{0,20}?(\d{6})\D{0,20}\s*")


def _now() -> datetime:
    return datetime.now(UTC)


def _hash(email: str, code: str) -> str:
    return hmac.new(_SECRET.encode(), f"chat:{email}:{code}".encode(), hashlib.sha256).hexdigest()


def deliver_code(email: str, code: str) -> None:
    """Demo delivery: log the code, as the auth service does. Replace with an email provider in production."""
    logger.warning("Verification code for %s: %s (demo delivery: logged, not emailed)", email, code)


def find_code(text: str) -> Optional[str]:
    """The 6-digit code in a message, if it has exactly one."""
    codes = CODE_IN_TEXT.findall(text or "")
    return codes[0] if len(codes) == 1 else None


def is_code_only(text: str) -> bool:
    """A message that is just a code, maybe with a few words around it ("it's 123456")."""
    return bool(CODE_ONLY.fullmatch(text or ""))


def start(email: str, profile: Optional[dict], classification: dict, pending: Optional[dict]) -> tuple[dict, str]:
    """Send a code to `email` and return (the new pending check, outcome). The outcome is "sent", "already_sent"
    (a code went to this email within the resend wait; that code still works) or "too_many" (the ticket used
    up its codes: a human checks identity). `profile` is None for an email with no account: nothing is sent,
    but the customer sees the same reply."""
    email = email.strip().lower()
    sent_count = (pending or {}).get("codes_sent", 0)
    if pending and pending.get("email") == email and \
            (_now() - datetime.fromisoformat(pending["sent_at"])).total_seconds() < RESEND_SECONDS:
        return pending, "already_sent"
    if sent_count >= MAX_CODES_PER_TICKET:
        return {**(pending or {}), "codes_sent": sent_count}, "too_many"
    code = f"{secrets.randbelow(10**6):06d}"
    new = {
        "email": email,
        "user_id": profile["user_id"] if profile else None,
        "user_name": profile["name"] if profile else None,
        "code_hash": _hash(email, code) if profile else secrets.token_hex(32),  # unknown email: nothing matches
        "expires_at": (_now() + CODE_TTL).isoformat(),
        "sent_at": _now().isoformat(),
        "attempts": 0,
        "codes_sent": sent_count + 1,
        # The request the customer was making when asked who they are, to continue it once verified
        "classification": classification,
    }
    if profile:
        deliver_code(email, code)
    return new, "sent"


def check(pending: dict, code: str) -> tuple[Optional[dict], str]:
    """Check a code against the pending verification. Returns (the pending check to keep, outcome): "verified"
    (keep None), "wrong" (one attempt used), "burnt" (too many wrong codes: keep None, ask for the email
    again) or "expired" (keep None)."""
    if datetime.fromisoformat(pending["expires_at"]) < _now():
        return _cleared(pending), "expired"
    if pending.get("user_id") and hmac.compare_digest(pending["code_hash"], _hash(pending["email"], code)):
        return None, "verified"
    attempts = pending.get("attempts", 0) + 1
    if attempts >= CODE_MAX_ATTEMPTS:
        return _cleared(pending), "burnt"
    return {**pending, "attempts": attempts}, "wrong"


def _cleared(pending: dict) -> dict:
    """No code is valid any more, but the ticket remembers how many codes it used."""
    return {"codes_sent": pending.get("codes_sent", 0)}


def is_pending(pending: Optional[dict]) -> bool:
    return bool(pending and pending.get("code_hash"))
