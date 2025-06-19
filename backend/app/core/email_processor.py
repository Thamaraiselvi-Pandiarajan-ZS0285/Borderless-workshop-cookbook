from backend.app.core.email_listener import EmailListener
from backend.app.core.email_parser import EmailParser
from backend.app.core.graph_client import GraphClient
from backend.utils.email_move_utils import EmailMover
from backend.config.dev_config import USER_ID

class EmailProcessor:
    def __init__(self):
        self.listener = EmailListener()
        self.parser = EmailParser()
        self.graph = GraphClient()
        self.access_token = self.graph._get_token()
        self.email_mover = EmailMover(self.access_token, USER_ID)

    def process_emails(self, limit=10):
        emails = self.listener.fetch_emails(top=limit)
        for email in emails:
            message_id = email["id"]
            try:
                parsed = self.parser.parse_email(email)
                # TODO: Store to DB or queue (e.g., Cosmos, PostgreSQL, Kafka)
                print(f"Storing email: {parsed['subject']}")
                #self.graph.move_message(email["id"], destination_folder="Processed")

                # 2. Move to Processed folder
                self.email_mover.move_email_to_folder(email["id"], "Processed")
                print("Email moved to 'Processed' folder.")

            except Exception as e:
                print(f"Failed to process: {email['subject']} -> {str(e)}")
                #self.graph.move_message(email["id"], destination_folder="Failed")

                # Try moving to Failed folder if processing fails
                self.email_mover.move_email_to_folder(message_id, "Failed")
                print("Email moved to 'Failed' folder.")

