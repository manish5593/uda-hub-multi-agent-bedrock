"""Sending one-time codes: by email through Amazon SES, or written to the log for demos.

Three kinds of code use it: sign-up and login codes (the auth service) and the code that proves an email a
guest typed in the chat (agentic/verification.py).

    UDAHUB_EMAIL_PROVIDER   "log" (default): the code is written to the service log, not emailed
                            "ses": sent with Amazon SES (AWS credentials and AWS_REGION as for Bedrock)
    UDAHUB_EMAIL_FROM       the sender, an address or domain verified in SES
    UDAHUB_EMAIL_LOG_FALLBACK  "true" (default): if SES fails, log the code so a demo can go on; set
                            "false" in production, where a code must never reach the logs

With SES, the code is never logged. While the SES account is in the sandbox, it only delivers to addresses
verified in SES as well.
"""
import logging
import os
from functools import lru_cache

logger = logging.getLogger("udahub.mailer")

SUBJECTS = {
    "sign-up": "Your CultPass sign-up code",
    "login": "Your CultPass login code",
    "chat": "Your CultPass verification code",
}
BODIES = {
    "sign-up": "Your code to create your CultPass login is {code}.",
    "login": "Your code to log in to CultPass is {code}.",
    "chat": "Your code to confirm this email address in your CultPass support chat is {code}.",
}
FOOTER = ("\n\nIt expires in 10 minutes and works once. If you didn't ask for it, ignore this email: "
          "nobody can use your account without the code.\n\nCultPass Support")
LABELS = {"sign-up": "Verification code", "login": "Login code", "chat": "Verification code"}


def provider() -> str:
    return (os.getenv("UDAHUB_EMAIL_PROVIDER") or "log").strip().lower()


@lru_cache(maxsize=1)
def _ses():
    import boto3
    return boto3.client("sesv2", region_name=os.getenv("AWS_REGION") or "us-east-1")


def send_code(email: str, code: str, purpose: str = "sign-up") -> bool:
    """Deliver a code. Returns True when it was emailed, False when it was only logged."""
    sender = os.getenv("UDAHUB_EMAIL_FROM", "").strip()
    if provider() == "ses" and sender:
        try:
            _ses().send_email(
                FromEmailAddress=sender,
                Destination={"ToAddresses": [email]},
                Content={"Simple": {
                    "Subject": {"Data": SUBJECTS.get(purpose, SUBJECTS["sign-up"])},
                    "Body": {"Text": {"Data": BODIES.get(purpose, BODIES["sign-up"]).format(code=code) + FOOTER}},
                }},
            )
            logger.info("%s emailed with SES", LABELS.get(purpose, "Code"))
            return True
        except Exception as exc:  # noqa: BLE001 - any SES failure falls back or is reported, never crashes a login
            logger.error("Could not email a %s with SES: %s: %s", purpose, type(exc).__name__, str(exc)[:200])
            if (os.getenv("UDAHUB_EMAIL_LOG_FALLBACK") or "true").strip().lower() in ("0", "false", "no", "off"):
                return False
    elif provider() == "ses":
        logger.error("UDAHUB_EMAIL_PROVIDER=ses but UDAHUB_EMAIL_FROM is not set: logging the code instead")
    logger.warning("%s for %s: %s (demo delivery: logged, not emailed)", LABELS.get(purpose, "Code"), email, code)
    return False
