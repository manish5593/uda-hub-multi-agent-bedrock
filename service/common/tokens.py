"""Customer access tokens: JWTs issued by the auth service and verified by the gateway.

Both sides share JWT_SECRET_KEY (HS256). A token says which CultPass customer is calling (`sub`, their
CultPass user id) and is only accepted with this issuer, this audience, the `access` type and an unexpired
`exp`. The gateway verifies tokens locally, without a call to the auth service per request.
"""
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Optional

import jwt

ALGORITHM = "HS256"
ISSUER = "udahub-auth"
AUDIENCE = "udahub-api"
TOKEN_TYPE = "access"
MIN_SECRET_LENGTH = 32


def jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "").strip()
    if len(secret) < MIN_SECRET_LENGTH:
        raise RuntimeError(f"JWT_SECRET_KEY must be at least {MIN_SECRET_LENGTH} characters")
    return secret


def jwt_configured() -> bool:
    try:
        jwt_secret()
    except RuntimeError:
        return False
    return True


def token_ttl() -> timedelta:
    return timedelta(minutes=int(os.getenv("JWT_TTL_MINUTES", "60")))


def issue_access_token(customer_id: str, name: Optional[str] = None) -> dict:
    """A signed access token for one CultPass customer, with the response body the API returns."""
    now = datetime.now(UTC)
    expires = now + token_ttl()
    claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": customer_id, "typ": TOKEN_TYPE,
              "iat": now, "exp": expires, "jti": uuid.uuid4().hex}
    if name:
        claims["name"] = name
    token = jwt.encode(claims, jwt_secret(), algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer", "expires_at": expires.isoformat()}


def verify_access_token(token: str) -> dict:
    """The token's claims. Raises jwt.InvalidTokenError for a bad signature, an expired token, the wrong
    issuer, audience or type, a missing claim, or any algorithm other than HS256."""
    claims = jwt.decode(token, jwt_secret(), algorithms=[ALGORITHM], audience=AUDIENCE, issuer=ISSUER,
                        options={"require": ["exp", "iat", "sub", "iss", "aud"]})
    if claims.get("typ") != TOKEN_TYPE:
        raise jwt.InvalidTokenError("not an access token")
    return claims
