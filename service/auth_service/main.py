"""Auth service: customer sign-up and login. Owns the auth database (credentials, verification codes).

    POST /register/start     {email}                   -> 202, emails a 6-digit code to a CultPass customer
    POST /register/complete  {email, code, password}   -> 201, creates the login and returns a token
    POST /login              {email, password}         -> 200, returns a token

Only CultPass customers can sign up, and only by proving they own the email: the code goes to that
address. In this demo the code is written to the service log instead of emailed (`deliver_code`). The
responses never reveal whether an email belongs to a customer or already has a login.

Passwords are stored as bcrypt hashes. After AUTH_MAX_FAILED_LOGINS wrong passwords in a row, the login is
locked for AUTH_LOCKOUT_MINUTES. Tokens are JWTs (service/common/tokens.py) that the gateway verifies.
Like every internal service, this one is reached only through the gateway, with the internal token.
"""
import hashlib
import hmac
import logging
import os
import secrets
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import bcrypt
from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic.tools import cultpass_tools
from service.auth_service.models import Credential, VerificationCode
from service.common.app import create_app, ping_database
from service.common.tokens import issue_access_token, jwt_secret

logger = logging.getLogger("auth")

CODE_TTL = timedelta(minutes=int(os.getenv("AUTH_CODE_TTL_MINUTES", "10")))
CODE_RESEND_SECONDS = int(os.getenv("AUTH_CODE_RESEND_SECONDS", "60"))
CODE_MAX_ATTEMPTS = int(os.getenv("AUTH_CODE_MAX_ATTEMPTS", "5"))
MAX_FAILED_LOGINS = int(os.getenv("AUTH_MAX_FAILED_LOGINS", "5"))
LOCKOUT = timedelta(minutes=int(os.getenv("AUTH_LOCKOUT_MINUTES", "15")))
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_BYTES = 72  # bcrypt's limit

CODE_SENT = "If this email belongs to a CultPass customer without a login, a verification code has been sent to it."
INVALID_CODE = "Invalid or expired verification code"
BAD_LOGIN = "Incorrect email or password, or too many failed attempts: try again later"


@lru_cache(maxsize=1)
def engine():
    url = os.getenv("AUTH_DB_URL")
    if not url:
        raise RuntimeError("Set AUTH_DB_URL")
    return create_engine(url, pool_pre_ping=True)


@contextmanager
def session_scope():
    with Session(engine()) as session, session.begin():
        yield session


def now() -> datetime:
    """UTC, without a time zone, as the columns store it."""
    return datetime.now(UTC).replace(tzinfo=None)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def code_hash(email: str, code: str) -> str:
    return hmac.new(jwt_secret().encode(), f"{email}:{code}".encode(), hashlib.sha256).hexdigest()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


# Checked against when the email has no login, so a login takes as long whether or not the email exists
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def deliver_code(email: str, code: str) -> None:
    """Demo delivery: log the code. Replace with an email provider in production, and stop logging codes."""
    logger.warning("Verification code for %s: %s (demo delivery: logged, not emailed)", email, code)


@asynccontextmanager
async def lifespan(app):
    jwt_secret()  # refuse to start without a usable signing key
    engine()
    yield


app = create_app("auth", "UDA-Hub auth service", health_check=lambda: ping_database(engine()), lifespan=lifespan)


class EmailIn(BaseModel):
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterIn(EmailIn):
    code: str = Field(pattern=r"^\d{6}$")
    password: str = Field(min_length=MIN_PASSWORD_LENGTH)

    @field_validator("password")
    @classmethod
    def strong_enough(cls, password: str) -> str:
        if len(password.encode()) > MAX_PASSWORD_BYTES:
            raise ValueError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes")
        if not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
            raise ValueError("Password must contain at least one letter and one digit")
        return password


class LoginIn(EmailIn):
    password: str = Field(min_length=1, max_length=200)


@app.post("/register/start", status_code=202)
def register_start(body: EmailIn):
    email = normalize_email(body.email)
    with session_scope() as session:
        if session.get(Credential, email) is not None:
            return {"detail": CODE_SENT}
        pending = session.get(VerificationCode, email)
        if pending is not None and (now() - pending.created_at).total_seconds() < CODE_RESEND_SECONDS:
            return {"detail": CODE_SENT}

    if cultpass_tools.find_customer_by_email(email) is None:
        logger.info("Sign-up requested for an email that is not a CultPass customer")
        return {"detail": CODE_SENT}

    code = f"{secrets.randbelow(10**6):06d}"
    with session_scope() as session:
        session.merge(VerificationCode(email=email, code_hash=code_hash(email, code), attempts=0,
                                       expires_at=now() + CODE_TTL, created_at=now()))
    deliver_code(email, code)
    return {"detail": CODE_SENT}


@app.post("/register/complete", status_code=201)
def register_complete(body: RegisterIn):
    email = normalize_email(body.email)
    error = None
    with session_scope() as session:
        pending = session.get(VerificationCode, email)
        if pending is None or pending.expires_at < now():
            error = (400, INVALID_CODE)
        elif not hmac.compare_digest(pending.code_hash, code_hash(email, body.code)):
            pending.attempts += 1
            if pending.attempts >= CODE_MAX_ATTEMPTS:
                session.delete(pending)  # too many guesses: the customer has to ask for a new code
            error = (400, INVALID_CODE)
        else:
            session.delete(pending)  # a code works once
            profile = cultpass_tools.find_customer_by_email(email)
            if profile is None:
                error = (400, INVALID_CODE)
            elif session.get(Credential, email) is not None or \
                    session.query(Credential).filter_by(external_user_id=profile["user_id"]).first() is not None:
                error = (409, "This customer already has a login")
            else:
                session.add(Credential(email=email, external_user_id=profile["user_id"],
                                       password_hash=hash_password(body.password), failed_logins=0,
                                       created_at=now()))
    if error:
        raise HTTPException(*error)
    logger.info("Login created", extra={"fields": {"customer": profile["user_id"]}})
    return issue_access_token(profile["user_id"], profile["name"])


@app.post("/login")
def login(body: LoginIn):
    email = normalize_email(body.email)
    with session_scope() as session:
        credential = session.get(Credential, email)
        locked = credential is not None and credential.locked_until is not None and credential.locked_until > now()
        valid = bcrypt.checkpw(body.password.encode()[:MAX_PASSWORD_BYTES],
                               (credential.password_hash if credential else _DUMMY_HASH).encode())
        ok = credential is not None and valid and not locked
        if credential is not None and not locked:
            if ok:
                credential.failed_logins = 0
                credential.locked_until = None
            else:
                credential.failed_logins += 1
                if credential.failed_logins >= MAX_FAILED_LOGINS:
                    credential.locked_until = now() + LOCKOUT
                    credential.failed_logins = 0
                    logger.warning("Login locked after repeated failures",
                                   extra={"fields": {"customer": credential.external_user_id}})
        customer_id = credential.external_user_id if ok else None
    if not ok:
        raise HTTPException(401, BAD_LOGIN)

    profile = cultpass_tools.get_customer_profile(customer_id)
    if profile is None:  # the CultPass account is gone
        raise HTTPException(401, BAD_LOGIN)
    return issue_access_token(customer_id, profile["name"])
