"""The auth service: verified sign-up, login, lockout, and the tokens it issues."""
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from agentic.tools import cultpass_tools
from service.auth_service import main
from service.auth_service.models import Credential
from service.common import tokens

EMAIL = "bob.stone@granite.com"
PASSWORD = "granite-rocks-42"


def sign_up(client, codes, email=EMAIL, password=PASSWORD):
    assert client.post("/register/start", json={"email": email}).status_code == 202
    return client.post("/register/complete", json={"email": email, "code": codes[email.lower()],
                                                   "password": password})


def test_sign_up_then_login_issue_tokens_for_that_customer(auth_client, codes):
    bob = cultpass_tools.find_customer_by_email(EMAIL)

    created = sign_up(auth_client, codes)
    assert created.status_code == 201
    claims = tokens.verify_access_token(created.json()["access_token"])
    assert claims["sub"] == bob["user_id"] and claims["name"] == bob["name"]
    assert created.json()["token_type"] == "bearer"

    # Emails are case-insensitive
    login = auth_client.post("/login", json={"email": EMAIL.upper(), "password": PASSWORD})
    assert login.status_code == 200
    assert tokens.verify_access_token(login.json()["access_token"])["sub"] == bob["user_id"]


def test_password_is_stored_as_a_bcrypt_hash(auth_client, codes):
    sign_up(auth_client, codes)
    with main.session_scope() as session:
        stored = session.get(Credential, EMAIL).password_hash
    assert stored.startswith("$2b$") and PASSWORD not in stored


def test_sign_up_never_reveals_whether_an_email_is_a_customer(auth_client, codes):
    stranger = auth_client.post("/register/start", json={"email": "nobody@example.com"})
    sign_up(auth_client, codes)
    codes.clear()
    registered = auth_client.post("/register/start", json={"email": EMAIL})

    # Same answer for a non-customer, a customer and a customer who already has a login; only a
    # customer without a login gets a code
    assert stranger.status_code == registered.status_code == 202
    assert stranger.json() == registered.json()
    assert codes == {}


def test_a_code_is_needed_and_works_once(auth_client, codes):
    auth_client.post("/register/start", json={"email": EMAIL})
    wrong = "000000" if codes[EMAIL] != "000000" else "111111"
    assert auth_client.post("/register/complete", json={"email": EMAIL, "code": wrong,
                                                         "password": PASSWORD}).status_code == 400
    # Never requested: no code to match
    assert auth_client.post("/register/complete", json={"email": "cathy.bloom@florals.org", "code": "123456",
                                                         "password": PASSWORD}).status_code == 400
    ok = {"email": EMAIL, "code": codes[EMAIL], "password": PASSWORD}
    assert auth_client.post("/register/complete", json=ok).status_code == 201
    assert auth_client.post("/register/complete", json=ok).status_code == 400


def test_too_many_wrong_codes_burn_the_code(auth_client, codes, monkeypatch):
    monkeypatch.setattr(main, "CODE_MAX_ATTEMPTS", 3)
    auth_client.post("/register/start", json={"email": EMAIL})
    code = codes[EMAIL]
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(3):
        auth_client.post("/register/complete", json={"email": EMAIL, "code": wrong, "password": PASSWORD})
    # The right code no longer works: the customer has to ask for a new one
    assert auth_client.post("/register/complete", json={"email": EMAIL, "code": code,
                                                         "password": PASSWORD}).status_code == 400


def test_expired_code_is_refused(auth_client, codes, monkeypatch):
    monkeypatch.setattr(main, "CODE_TTL", timedelta(seconds=-1))
    auth_client.post("/register/start", json={"email": EMAIL})
    assert auth_client.post("/register/complete", json={"email": EMAIL, "code": codes[EMAIL],
                                                         "password": PASSWORD}).status_code == 400


def test_codes_are_not_resent_within_the_cooldown(auth_client, codes):
    auth_client.post("/register/start", json={"email": EMAIL})
    first = codes.pop(EMAIL)
    auth_client.post("/register/start", json={"email": EMAIL})
    assert EMAIL not in codes
    # The first code still works
    assert auth_client.post("/register/complete", json={"email": EMAIL, "code": first,
                                                         "password": PASSWORD}).status_code == 201


@pytest.mark.parametrize("password", ["short1", "no-digits-at-all", "1234567890123", "a1" * 40])
def test_weak_passwords_are_refused(auth_client, codes, password):
    auth_client.post("/register/start", json={"email": EMAIL})
    response = auth_client.post("/register/complete", json={"email": EMAIL, "code": codes[EMAIL],
                                                             "password": password})
    assert response.status_code == 422


def test_wrong_password_and_unknown_email_get_the_same_answer(auth_client, codes):
    sign_up(auth_client, codes)
    wrong = auth_client.post("/login", json={"email": EMAIL, "password": "not-the-password-1"})
    unknown = auth_client.post("/login", json={"email": "nobody@example.com", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_repeated_failures_lock_the_login(auth_client, codes, monkeypatch):
    monkeypatch.setattr(main, "MAX_FAILED_LOGINS", 3)
    sign_up(auth_client, codes)
    for _ in range(3):
        assert auth_client.post("/login", json={"email": EMAIL, "password": "wrong-password-1"}).status_code == 401
    # Locked: even the right password is refused, with the same answer
    locked = auth_client.post("/login", json={"email": EMAIL, "password": PASSWORD})
    assert locked.status_code == 401

    with main.session_scope() as session:
        session.get(Credential, EMAIL).locked_until = main.now() - timedelta(seconds=1)
    assert auth_client.post("/login", json={"email": EMAIL, "password": PASSWORD}).status_code == 200


def test_a_success_resets_the_failure_count(auth_client, codes, monkeypatch):
    monkeypatch.setattr(main, "MAX_FAILED_LOGINS", 3)
    sign_up(auth_client, codes)
    for _ in range(2):
        auth_client.post("/login", json={"email": EMAIL, "password": "wrong-password-1"})
    assert auth_client.post("/login", json={"email": EMAIL, "password": PASSWORD}).status_code == 200
    for _ in range(2):
        auth_client.post("/login", json={"email": EMAIL, "password": "wrong-password-1"})
    assert auth_client.post("/login", json={"email": EMAIL, "password": PASSWORD}).status_code == 200


def test_the_service_needs_the_internal_token(auth_client):
    auth_client.headers.pop("X-Internal-Token")
    assert auth_client.post("/login", json={"email": EMAIL, "password": PASSWORD}).status_code == 401


def test_the_service_refuses_to_start_without_a_signing_key(auth_client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "too-short")
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        tokens.jwt_secret()


# --- Tokens -------------------------------------------------------------------------------------------

def forge(**overrides):
    now = datetime.now(UTC)
    claims = {"iss": tokens.ISSUER, "aud": tokens.AUDIENCE, "sub": "a4ab87", "typ": "access",
              "iat": now, "exp": now + timedelta(minutes=5), **overrides}
    secret = claims.pop("secret", tokens.jwt_secret())
    algorithm = claims.pop("alg", tokens.ALGORITHM)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, secret, algorithm=algorithm)


def test_a_valid_token_is_accepted():
    assert tokens.verify_access_token(forge())["sub"] == "a4ab87"


@pytest.mark.parametrize("overrides", [
    {"exp": datetime.now(UTC) - timedelta(seconds=1)},
    {"secret": "another-secret-that-is-also-long-enough-to-use"},
    {"aud": "someone-else"},
    {"iss": "someone-else"},
    {"typ": "refresh"},
    {"sub": None},
    {"exp": None},
    {"alg": "HS512"},
], ids=["expired", "wrong-secret", "wrong-audience", "wrong-issuer", "wrong-type", "no-subject", "no-expiry",
        "other-algorithm"])
def test_bad_tokens_are_refused(overrides):
    with pytest.raises(jwt.InvalidTokenError):
        tokens.verify_access_token(forge(**overrides))


def test_unsigned_tokens_are_refused():
    unsigned = jwt.encode({"iss": tokens.ISSUER, "aud": tokens.AUDIENCE, "sub": "a4ab87", "typ": "access",
                           "iat": datetime.now(UTC), "exp": datetime.now(UTC) + timedelta(minutes=5)},
                          key=None, algorithm="none")
    with pytest.raises(jwt.InvalidTokenError):
        tokens.verify_access_token(unsigned)
