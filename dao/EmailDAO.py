from sqlalchemy.orm import sessionmaker

from backend.models.all_db_models import Email

class EmailDAO:
    def __init__(self, db_session:sessionmaker):
        self.db = db_session

    def insert_email(self, email: dict):
        with self.db() as session:
            session.add(Email(
                email_id=email["email_id"],
                subject=email["subject"],
                sender=email["sender"],
                recipients=email["recipients"],
                received_time=email["received_time"],
                body=email["body"]
            ))
            session.commit()
