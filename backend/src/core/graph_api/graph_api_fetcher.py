from typing import List
import requests
from msal import ConfidentialClientApplication
from backend.src.config.dev_config import *
from backend.src.controller.request_handler.email_request import EmailClassificationRequest, Attachment


class GraphEmailFetcher:
    def __init__(self):
        self.authority = f"https://login.microsoftonline.com/{TENANT_ID}"
        self.token = self._authenticate()
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def _authenticate(self) -> str:
        app = ConfidentialClientApplication(
            client_id=CLIENT_ID,
            client_credential=CLIENT_SECRET,
            authority=self.authority
        )
        token_response = app.acquire_token_for_client(scopes=SCOPE)
        if "access_token" not in token_response:
            raise Exception("Authentication failed: " + token_response.get("error_description"))
        return token_response["access_token"]

    def get_messages_with_attachments(self, top: int = 10) -> List[EmailClassificationRequest]:
        url = f"{GRAPH_API_ENDPOINT}/users/{SHARED_MAILBOX}/mailFolders/inbox/messages?$top={top}"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching messages: {response.status_code} - {response.text}")

        messages = response.json().get("value", [])
        result = []

        for msg in messages:
            attachments_list = []

            if msg.get("hasAttachments", False):
                message_id = msg["id"]
                attachment_url = f"{GRAPH_API_ENDPOINT}/users/{SHARED_MAILBOX}/messages/{message_id}/attachments"
                att_response = requests.get(attachment_url, headers=self.headers)
                if att_response.status_code == 200:
                    attachments = att_response.json().get("value", [])
                    for att in attachments:
                        if "contentBytes" in att:
                            attachments_list.append(Attachment(
                                name=att.get("name"),
                                contentType=att.get("contentType"),
                                contentBytes=att.get("contentBytes")
                            ))

            # email = EmailClassificationRequest(
            #     sender=msg.get("from", {}).get("emailAddress", {}).get("address", ""),
            #     received_at=msg.get("sentDateTime"),
            #     subject=msg.get("subject", ""),
            #     body=msg.get("body", {}).get("content", ""),
            #     hasAttachments=msg.get("hasAttachments", False),
            #     attachments=attachments_list
            # )
            email_with_attachment={
                "email":msg,
                "attachment":attachments_list
            }
            result.append(email_with_attachment)

        return result