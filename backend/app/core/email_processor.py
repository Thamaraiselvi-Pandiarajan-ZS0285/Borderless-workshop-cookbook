from backend.app.core.email_parser import EmailParser
from backend.app.core.graph_client import GraphClient
from backend.utils.email_move_utils import EmailMover
from backend.config.dev_config import USER_ID

class EmailProcessor:
    def __init__(self):
        self.parser = EmailParser()
        self.graph = GraphClient()
        self.access_token = self.graph._get_token()
        self.email_mover = EmailMover(self.access_token, USER_ID)

    def process_emails(self, emails):
        for email in emails:
            print(email)
            message_id = email["id"]
            try:
                # Move to Processed folder
                self.email_mover.move_email_to_folder(email["id"], "Processed")
                print("Email moved to 'Processed' folder.")

            except Exception as e:
                print(f"Failed to process: {email['subject']} -> {str(e)}")

                # Try moving to Failed folder if processing fails
                self.email_mover.move_email_to_folder(message_id, "Failed")
                print("Email moved to 'Failed' folder.")
