from pydantic import BaseModel, ValidationError
from datetime import datetime
from typing import List

from backend.src.controller.request_handler.email_request import (
    EmailClassificationRequest,
    EmailClassifyImageRequest
)


class EmailClassifierResponse(BaseModel):
    """
    Response model for classified emails.
    """
    subject: str
    body: str
    summary: str
    sender: str
    receivedAt: datetime
    classification: str


def build_email_classifier_response(
    email: EmailClassificationRequest,
    classification: str,
    summary: str
) -> EmailClassifierResponse:
    """
    Builds the response object for a classified email.

    Args:
        email (EmailClassificationRequest): The incoming email request object.
        classification (str): The classification result (e.g., 'spam', 'important').
        summary (str): A summary of the email body.

    Returns:
        EmailClassifierResponse: The structured response object.
    """
    try:
        return EmailClassifierResponse(
            subject=email.subject,
            body=email.body,
            summary=summary,
            sender=email.sender,
            receivedAt=email.received_at,
            classification=classification
        )
    except ValidationError as e:
        raise ValueError(f"Invalid data for email classification response: {e}")


class EmailImageClassifierResponse(BaseModel):
    """
    Response model for classified emails with images.
    """
    subject: str
    body: str
    summary: str
    sender: str
    receivedAt: datetime
    classification: List[str]


def email_classify_response_via_vlm(
    request: EmailClassifyImageRequest,
    classification: List[str],
    summary: str
) -> EmailImageClassifierResponse:
    """
    Builds the response object for an email classified using a VLM (vision-language model).

    Args:
        request (EmailClassifyImageRequest): The incoming request containing email content and images.
        classification (List[str]): List of classification results.
        summary (str): Summary of the email content.

    Returns:
        EmailImageClassifierResponse: The structured response including classification results.
    """
    try:
        return EmailImageClassifierResponse(
            subject=request.json_data.subject,
            body=request.json_data.body,
            summary=summary,
            sender=request.json_data.sender,
            receivedAt=request.json_data.received_at,
            classification=classification
        )
    except ValidationError as e:
        raise ValueError(f"Invalid data for image-based email classification response: {e}")
