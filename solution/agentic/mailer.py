"""Sending one-time codes: by email through Amazon SES, or written to the log for demos.

Three kinds of code use it: sign-up and login codes (the auth service) and the code that proves an email a
guest typed in the chat (agentic/verification.py).

    UDAHUB_EMAIL_PROVIDER   "log" (default): the code is written to the service log, not emailed
                            "ses": sent with Amazon SES (AWS credentials and AWS_REGION as for Bedrock)
                            "smtp": sent through an SMTP server, e.g. Gmail with an app password
    UDAHUB_EMAIL_FROM       the sender: an address or domain verified in SES, or for SMTP the account's address
                            (defaults to UDAHUB_SMTP_USER)
    UDAHUB_SMTP_HOST, UDAHUB_SMTP_PORT   default smtp.gmail.com, 587 (STARTTLS)
    UDAHUB_SMTP_USER, UDAHUB_SMTP_PASSWORD   for Gmail: the address and an app password
    UDAHUB_EMAIL_LOG_FALLBACK  "true" (default): if SES fails, log the code so a demo can go on; set
                            "false" in production, where a code must never reach the logs

Once emailed, a code is never logged. While the SES account is in the sandbox, it only delivers to addresses
verified in SES as well. Gmail rejects mail sent through SES "from" a gmail.com address (it isn't signed by
Google), so with a Gmail sender use SMTP; SES needs a sender on a domain with DKIM set up in SES.
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


def _send_smtp(sender: str, to: str, subject: str, body: str) -> None:
    import smtplib
    from email.message import EmailMessage
    message = EmailMessage()
    message["From"], message["To"], message["Subject"] = f"CultPass Support <{sender}>", to, subject
    message.set_content(body)
    host = os.getenv("UDAHUB_SMTP_HOST") or "smtp.gmail.com"
    port = int(os.getenv("UDAHUB_SMTP_PORT") or 587)
    with smtplib.SMTP(host, port, timeout=15) as smtp:
        smtp.starttls()
        smtp.login(os.getenv("UDAHUB_SMTP_USER", ""), os.getenv("UDAHUB_SMTP_PASSWORD", ""))
        smtp.send_message(message)


def send_code(email: str, code: str, purpose: str = "sign-up") -> bool:
    """Deliver a code. Returns True when it was emailed, False when it was only logged."""
    kind = provider()
    sender = (os.getenv("UDAHUB_EMAIL_FROM") or (os.getenv("UDAHUB_SMTP_USER") if kind == "smtp" else "") or "").strip()
    subject = SUBJECTS.get(purpose, SUBJECTS["sign-up"])
    body = BODIES.get(purpose, BODIES["sign-up"]).format(code=code) + FOOTER
    if kind in ("ses", "smtp") and sender:
        try:
            if kind == "ses":
                _ses().send_email(FromEmailAddress=sender, Destination={"ToAddresses": [email]},
                                  Content={"Simple": {"Subject": {"Data": subject}, "Body": {"Text": {"Data": body}}}})
            else:
                _send_smtp(sender, email, subject, body)
            logger.info("%s emailed with %s", LABELS.get(purpose, "Code"), kind.upper())
            return True
        except Exception as exc:  # noqa: BLE001 - any email failure falls back or is reported, never crashes a login
            logger.error("Could not email a %s with %s: %s: %s", purpose, kind.upper(), type(exc).__name__,
                         str(exc)[:200])
            if (os.getenv("UDAHUB_EMAIL_LOG_FALLBACK") or "true").strip().lower() in ("0", "false", "no", "off"):
                return False
    elif kind in ("ses", "smtp"):
        logger.error("UDAHUB_EMAIL_PROVIDER=%s but no sender is set (UDAHUB_EMAIL_FROM%s): logging the code instead",
                     kind, " or UDAHUB_SMTP_USER" if kind == "smtp" else "")
    logger.warning("%s for %s: %s (demo delivery: logged, not emailed)", LABELS.get(purpose, "Code"), email, code)
    return False
