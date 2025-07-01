from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
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

@dataclass
class RFPContext:
    """Context structure for RFP analysis"""
    raw_content: str
    attachments_content: str
    metadata: Dict[str, Any]

@dataclass
class DimensionalSummary:
    """Structure for dimensional summary results"""
    dimension: str
    summary: str
    confidence_score: float