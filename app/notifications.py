import smtplib
from email.mime.text import MIMEText

from app.config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFY_EMAIL


def send_notification_email(subject: str, body: str):
    """
    Sends a plain-text email to NOTIFY_EMAIL. Silently no-ops (with a
    printed warning) if SMTP isn't configured, rather than crashing the
    request — a request being logged to the database matters more than
    the email succeeding.
    """
    if not (SMTP_USER and SMTP_PASSWORD and NOTIFY_EMAIL):
        print("[WARN] SMTP not configured — skipping email, request was still saved to the database.")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = NOTIFY_EMAIL

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, [NOTIFY_EMAIL], msg.as_string())
