from pydantic import BaseModel
from datetime import datetime

from backend.app.request_handler.email_request import EmailClassificationRequest


class EmailClassifierResponse(BaseModel):
    subject: str
    body: str
    summary: str
    sender: str
    receivedAt: datetime
    classification: str


def build_email_classifier_response( email:EmailClassificationRequest,
    classification: str,summary:str ) -> EmailClassifierResponse:
    return EmailClassifierResponse(
        subject=email.subject,
        body=email.body,
        summary=summary,
        sender=email.sender,
        receivedAt=email.received_at,
        classification=classification
    )
