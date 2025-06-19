from bs4 import BeautifulSoup

class EmailParser:
    @staticmethod
    def parse_email(email_data):
        body = email_data.get("body", {}).get("content", "")
        clean_body = BeautifulSoup(body, "html.parser").get_text()
        return {
            "id": email_data["id"],
            "subject": email_data.get("subject", ""),
            "from": email_data.get("from", {}).get("emailAddress", {}).get("address", ""),
            "to": [r["emailAddress"]["address"] for r in email_data.get("toRecipients", [])],
            "body": clean_body,
        }
