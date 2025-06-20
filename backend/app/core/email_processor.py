from sqlalchemy.orm import sessionmaker

from backend.app.core.email_listener import EmailListener
from backend.app.core.email_parser import EmailParser
from backend.app.core.graph_client import GraphClient
from backend.utils.email_move_utils import EmailMover
from backend.config.dev_config import USER_ID
from dao.EmailDAO import EmailDAO


class EmailProcessor:
    def __init__(self):
        self.listener = EmailListener()
        self.parser = EmailParser()
        self.graph = GraphClient()
        self.access_token = self.graph._get_token()
        #self.email_dao = EmailDAO(db_session)
        self.email_mover = EmailMover(self.access_token, USER_ID)

    def process_emails(self, limit=10):
        emails = self.listener.fetch_emails(top=limit)
        for email in emails:
            message_id = email["id"]
            try:
                print(email)
                #parsed = self.parser.parse_email(email)
                #print(parsed)

                # Store email metadata into postgres
                #self.store_email(parsed)

                # Move to Processed folder
                self.email_mover.move_email_to_folder(email["id"], "Processed")
                print("Email moved to 'Processed' folder.")

            except Exception as e:
                print(f"Failed to process: {email['subject']} -> {str(e)}")

                # Try moving to Failed folder if processing fails
                self.email_mover.move_email_to_folder(message_id, "Failed")
                print("Email moved to 'Failed' folder.")

    def process_email_by_id(self, message_id: str):
        email_data = self.graph.get_message_by_id(message_id)
        if email_data:
            try:
                parsed = self.parser.parse_email(email_data)
                # TODO: Store to DB or queue (e.g., Cosmos, PostgreSQL, Kafka)
                print(f"Storing email: {parsed['subject']}")

                # Store email metadata into postgres
                #self.store_email(parsed)

                # 2. Move to Processed folder
                self.email_mover.move_email_to_folder(message_id, "Processed")
                print("Email moved to 'Processed' folder.")

            except Exception as e:
                print(f"Failed to process: {email_data['subject']} -> {str(e)}")

                # Try moving to Failed folder if processing fails
                self.email_mover.move_email_to_folder(message_id, "Failed")
                print("Email moved to 'Failed' folder.")

    def store_email(self, parsed_email):
        print(f"Storing email: {parsed_email['subject']}")

        self.email_dao.insert_email({
            "email_id": parsed_email['email_id'],
            "subject": parsed_email['subject'],
            "sender": parsed_email['sender'],
            "recipients": parsed_email['recipients'],
            "received_time": parsed_email['received_time'],
            "body": parsed_email['body']
        })