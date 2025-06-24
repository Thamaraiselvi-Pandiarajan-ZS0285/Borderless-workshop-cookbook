from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class Attachment(BaseModel):
    name: str
    contentType: str
    contentBytes: str

class EmailClassificationRequest(BaseModel):
    session_id: Optional[str] = None
    sender: str
    received_at:datetime
    subject: str
    body: str
    hasAttachments:bool
    attachments: Optional[List[Attachment]] = []
