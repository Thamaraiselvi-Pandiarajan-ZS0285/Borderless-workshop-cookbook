from datetime import datetime
from uuid import uuid4

from bs4 import BeautifulSoup


def clean_html(body_content):
    soup = BeautifulSoup(body_content, "html.parser")
    return soup.get_text().strip()

class EmailParser:
    @staticmethod
    def parse_email(email_data):
        """
         Parses email metadata and body
         """
        email_id = str(uuid4())
        message_id = email_data.get("id", "")
        subject = email_data.get("subject", "")
        sender = email_data.get("from", {}).get("emailAddress", {}).get("address", "")
        recipients = [r.get("emailAddress", {}).get("address", "") for r in email_data.get("toRecipients", [])]
        received_time = email_data.get("receivedDateTime", datetime.utcnow().isoformat())
        body_content = email_data.get("body", {}).get("content", "")
        body_type = email_data.get("body", {}).get("contentType", "html")

        # Step 1: Clean body
        plain_text = clean_html(body_content) if body_type == "html" else body_content

        return {
            "id": email_id,
            "message_id": message_id,
            "subject": subject,
            "from": sender,
            "to": recipients,
            "received_time": received_time,
            "body_content": body_content,
            "body_type": body_type,
            "body": plain_text,
        }
