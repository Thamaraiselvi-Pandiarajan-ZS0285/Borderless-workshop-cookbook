from pydantic import BaseModel
from datetime import datetime


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

class EmailImageClassifierResponse(BaseModel):
    subject: str
    body: str
    summary: str
    sender: str
    receivedAt: datetime
    classification: list[str]

def email_classify_response_via_vlm(request:EmailClassifyImageRequest, classification:list[str], summary:str) ->EmailImageClassifierResponse:
    return EmailImageClassifierResponse(
        subject=request.json_data.subject,
        body=request.json_data.body,
        summary=summary,
        sender=request.json_data.sender,
        receivedAt=request.json_data.received_at,
        classification=classification
    )
