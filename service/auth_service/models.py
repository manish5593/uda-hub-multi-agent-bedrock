"""Tables of the auth database: customer logins and pending email verification codes.

Only the auth service uses them. The customer's profile stays in CultPass: a login just links an email
and a password hash to a CultPass user id.
"""
from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Credential(Base):
    __tablename__ = "credentials"

    # Lower-cased
    email: Mapped[str] = mapped_column(String(254), primary_key=True)
    # The CultPass user id, which is also the token's subject
    external_user_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(100), nullable=False)
    failed_logins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until = mapped_column(DateTime, nullable=True)
    created_at = mapped_column(DateTime, nullable=False)


class VerificationCode(Base):
    """One pending sign-up per email. A new code replaces the previous one."""
    __tablename__ = "verification_codes"

    email: Mapped[str] = mapped_column(String(254), primary_key=True)
    # HMAC-SHA256 of the email and the code: the code itself is never stored
    code_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    expires_at = mapped_column(DateTime, nullable=False)
    created_at = mapped_column(DateTime, nullable=False)
