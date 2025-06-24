from dataclasses import dataclass
from enum import Enum
from typing import Optional,Dict
from backend.app.request_handler.email_request import EmailClassificationRequest
from pydantic import BaseModel

class OrchestrateRequest(BaseModel):
    message: str
    conversation_id: str

class WorkflowStage(Enum):
    """Defines workflow stages"""
    EMAIL_CLASSIFICATION = "email_classification"
    # add the upcoming workflow stages
    HUMAN_REVIEW = "human_review"
    COMPLETED = "completed"


@dataclass
class WorkflowState:
    """Tracks the current state of the workflow"""
    email_classification: Optional[EmailClassificationRequest] = None
    human_feedback: Optional[Dict] = None
