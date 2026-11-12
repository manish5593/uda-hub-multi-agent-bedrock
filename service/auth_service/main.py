"""Auth service: customer sign-up and login. Owns the auth database (credentials, sign-up and login codes).

    POST /register/start     {email}                   -> 202, emails a 6-digit code to a CultPass customer
    POST /register/complete  {email, code, password}   -> 201, creates the login and returns a token
    POST /login              {email, password}         -> 202, emails a 6-digit login code
    POST /login/verify       {email, code}             -> 200, returns a token

Login takes two steps (AUTH_LOGIN_CODES, on by default): the password, then a code sent to the customer's
email, so a stolen password alone is not enough. With AUTH_LOGIN_CODES=false, /login returns the token directly.

Only CultPass customers can sign up, and only by proving they own the email: the code goes to that
address, by Amazon SES or, by default, in the service log instead (`deliver_code`, agentic/mailer.py). The
responses never reveal whether an email belongs to a customer or already has a login.

Passwords are stored as bcrypt hashes. After AUTH_MAX_FAILED_LOGINS wrong passwords or login codes in a row,
the login is locked for AUTH_LOCKOUT_MINUTES. The count only resets after a complete login, so knowing the
password doesn't buy more guesses at the code. Tokens are JWTs (service/common/tokens.py) that the gateway verifies.
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
from fastapi import HTTPException, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agentic import mailer
from agentic.tools import cultpass_tools
from service.auth_service.models import Credential, LoginCode, VerificationCode
from service.common.app import create_app, ping_database
from service.common.tokens import issue_access_token, jwt_secret

logger = logging.getLogger("auth")

CODE_TTL = timedelta(minutes=int(os.getenv("AUTH_CODE_TTL_MINUTES", "10")))
CODE_RESEND_SECONDS = int(os.getenv("AUTH_CODE_RESEND_SECONDS", "60"))
CODE_MAX_ATTEMPTS = int(os.getenv("AUTH_CODE_MAX_ATTEMPTS", "5"))
MAX_FAILED_LOGINS = int(os.getenv("AUTH_MAX_FAILED_LOGINS", "5"))
LOCKOUT = timedelta(minutes=int(os.getenv("AUTH_LOCKOUT_MINUTES", "15")))
LOGIN_CODES = os.getenv("AUTH_LOGIN_CODES", "true").strip().lower() not in ("0", "false", "no", "off")
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_BYTES = 72  # bcrypt's limit

CODE_SENT = "If this email belongs to a CultPass customer without a login, a verification code has been sent to it."
INVALID_CODE = "Invalid or expired verification code"
BAD_LOGIN = "Incorrect email or password, or too many failed attempts: try again later"
# Clients only reach this service through the gateway, so the message names the gateway's path
LOGIN_CODE_SENT = "A login code has been sent to your email. Send it to /v1/auth/login/verify within {minutes} minutes."
BAD_LOGIN_CODE = "Invalid or expired login code, or too many failed attempts: log in again"


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


def code_hash(email: str, code: str, purpose: str = "sign-up") -> str:
    """Keyed hash of a code. The purpose is part of it, so a sign-up code can never pass as a login code."""
    message = f"{email}:{code}" if purpose == "sign-up" else f"{purpose}:{email}:{code}"
    return hmac.new(jwt_secret().encode(), message.encode(), hashlib.sha256).hexdigest()


def new_code() -> str:
    return f"{secrets.randbelow(10**6):06d}"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


# Checked against when the email has no login, so a login takes as long whether or not the email exists
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def deliver_code(email: str, code: str, purpose: str = "sign-up") -> None:
    """Email the code with Amazon SES, or log it for demos (agentic/mailer.py, UDAHUB_EMAIL_PROVIDER)."""
    mailer.send_code(email, code, purpose)


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


class LoginCodeIn(EmailIn):
    code: str = Field(pattern=r"^\d{6}$")


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

    code = new_code()
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


def is_locked(credential: Credential) -> bool:
    return credential.locked_until is not None and credential.locked_until > now()


def record_failure(credential: Credential) -> bool:
    """Count a wrong password or login code; returns True when it locks the login."""
    credential.failed_logins += 1
    if credential.failed_logins < MAX_FAILED_LOGINS:
        return False
    credential.locked_until = now() + LOCKOUT
    credential.failed_logins = 0
    logger.warning("Login locked after repeated failures", extra={"fields": {"customer": credential.external_user_id}})
    return True


def send_login_code(email: str) -> None:
    """Email a login code, unless one was sent within the resend cooldown (that one still works)."""
    code = new_code()
    with session_scope() as session:
        pending = session.get(LoginCode, email)
        if pending is not None and (now() - pending.created_at).total_seconds() < CODE_RESEND_SECONDS:
            return
        session.merge(LoginCode(email=email, code_hash=code_hash(email, code, "login"), attempts=0,
                                expires_at=now() + CODE_TTL, created_at=now()))
    deliver_code(email, code, "login")


@app.post("/login")
def login(body: LoginIn, response: Response):
    email = normalize_email(body.email)
    with session_scope() as session:
        credential = session.get(Credential, email)
        locked = credential is not None and is_locked(credential)
        valid = bcrypt.checkpw(body.password.encode()[:MAX_PASSWORD_BYTES],
                               (credential.password_hash if credential else _DUMMY_HASH).encode())
        ok = credential is not None and valid and not locked
        if credential is not None and not locked:
            if not ok:
                record_failure(credential)
            elif not LOGIN_CODES:
                credential.failed_logins = 0
                credential.locked_until = None
        customer_id = credential.external_user_id if ok else None
    if not ok:
        raise HTTPException(401, BAD_LOGIN)

    profile = cultpass_tools.get_customer_profile(customer_id)
    if profile is None:  # the CultPass account is gone
        raise HTTPException(401, BAD_LOGIN)
    if not LOGIN_CODES:
        return issue_access_token(customer_id, profile["name"])
    send_login_code(email)
    response.status_code = 202
    return {"detail": LOGIN_CODE_SENT.format(minutes=int(CODE_TTL.total_seconds() // 60)), "code_required": True}


@app.post("/login/verify")
def login_verify(body: LoginCodeIn):
    email = normalize_email(body.email)
    customer_id = None
    with session_scope() as session:
        credential = session.get(Credential, email)
        pending = session.get(LoginCode, email)
        if credential is None or pending is None:
            pass
        elif is_locked(credential) or pending.expires_at < now():
            session.delete(pending)
        elif not hmac.compare_digest(pending.code_hash, code_hash(email, body.code, "login")):
            pending.attempts += 1
            # Too many guesses at this code, or at the login as a whole: start again from the password
            if record_failure(credential) or pending.attempts >= CODE_MAX_ATTEMPTS:
                session.delete(pending)
        else:
            session.delete(pending)  # a code works once
            credential.failed_logins = 0
            credential.locked_until = None
            customer_id = credential.external_user_id
    if customer_id is None:
        raise HTTPException(401, BAD_LOGIN_CODE)

    profile = cultpass_tools.get_customer_profile(customer_id)
    if profile is None:
        raise HTTPException(401, BAD_LOGIN_CODE)
    logger.info("Logged in with a login code", extra={"fields": {"customer": customer_id}})
    return issue_access_token(customer_id, profile["name"])
