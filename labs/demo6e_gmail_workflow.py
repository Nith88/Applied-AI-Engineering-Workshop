"""Optional read-only Gmail IMAP input for the LU08 workflow."""

import email
import imaplib
import os
from email.header import decode_header
from dotenv import load_dotenv

from workflow_core import build_workflow, run_one

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")


def decode_text(value):
    if not value:
        return ""
    parts = []
    for part, encoding in decode_header(value):
        if isinstance(part, bytes):
            parts.append(part.decode(encoding or "utf-8", errors="replace"))
        else:
            parts.append(part)
    return "".join(parts)


def get_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        return ""
    payload = msg.get_payload(decode=True)
    if payload:
        charset = msg.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
    return ""


def get_latest_email():
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        raise RuntimeError("Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env")

    mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
    try:
        mail.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        status, _ = mail.select("INBOX", readonly=True)
        if status != "OK":
            raise RuntimeError("Could not select Gmail INBOX")

        status, messages = mail.search(None, "ALL")
        if status != "OK" or not messages or not messages[0].split():
            raise RuntimeError("Inbox is empty or could not be searched")

        latest_id = messages[0].split()[-1]
        status, data = mail.fetch(latest_id, "(RFC822)")
        if status != "OK":
            raise RuntimeError("Could not fetch latest email")

        raw_message = next(
            item[1] for item in data
            if isinstance(item, tuple) and isinstance(item[1], bytes)
        )
        msg = email.message_from_bytes(raw_message)
        return {
            "sender": decode_text(msg.get("From")),
            "subject": decode_text(msg.get("Subject")),
            "body": get_body(msg),
        }
    finally:
        try:
            mail.logout()
        except Exception:
            pass


if __name__ == "__main__":
    print("Reading newest message from a dedicated test inbox (read-only)...")
    email_data = get_latest_email()
    graph = build_workflow()
    run_one(graph, email_data)
