from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class Attachment(BaseModel):
    name: str
    contentType: str
    contentBytes: str

class EmailClassificationRequest(BaseModel):
    sender: str
    received_at:datetime
    subject: str
    body: str
    hasAttachments:bool
    attachments: Optional[List[Attachment]] = []

class EmailClassifyImageInput(BaseModel):
    input_path: str
    file_name: str
    file_extension: str

class EmailClassifyImageRequest(BaseModel):
    imagedata: List[EmailClassifyImageInput]
    json_data : EmailClassificationRequest