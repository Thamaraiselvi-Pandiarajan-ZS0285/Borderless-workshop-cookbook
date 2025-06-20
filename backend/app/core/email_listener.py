import time
import logging
from datetime import datetime, timedelta

import requests

from backend.app.core.email_processor import EmailProcessor
from backend.app.core.graph_client import GraphClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EmailListener:
    def __init__(self, user_email, poll_interval=10):
        self.graph_client = GraphClient()
        self.access_token = self.graph_client._get_token()
        self.user_email = user_email
        self.poll_interval = poll_interval
        self.last_checked_time = datetime.utcnow()
        self.email_processor = EmailProcessor()

    def _fetch_new_emails(self):
        logger.info("Checking for new emails...")
        # Filter emails received after last check
        iso_time = self.last_checked_time.isoformat() + "Z"
        url = (
            f"https://graph.microsoft.com/v1.0/users/{self.user_email}/messages"
            f"?$filter=receivedDateTime ge {iso_time}&$orderby=receivedDateTime desc&$top=10"
        )
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        messages = response.json().get("value", [])
        return messages

    def start(self):
        logger.info(f"Started listening to mailbox: {self.user_email}")
        while True:
            try:
                messages = self._fetch_new_emails()
                self.email_processor.process_emails(messages)
                self.last_checked_time = datetime.utcnow()
            except Exception as e:
                logger.error(f"Error fetching emails: {e}")
            time.sleep(self.poll_interval)
