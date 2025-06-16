import email
from io import StringIO
from typing import Dict

def parse_eml_content(content: str) -> Dict[str, str]:
    msg = email.message_from_file(StringIO(content))
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type in ["text/plain", "text/html"]:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body += payload.decode(charset, errors="ignore")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="ignore")

    return {
        "subject": msg["Subject"] or "No Subject",
        "from": msg["From"] or "Unknown",
        "to": msg["To"] or "Unknown",
        "date": msg["Date"] or "Unknown",
        "body": body,
    }