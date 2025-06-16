from pydantic import BaseModel
from enum import Enum

class EmailFormat(str, Enum):
    eml = "eml"
    html = "html"
    xml = "xml"

class EmailInput(BaseModel):
    content: str            # Raw email content
    format: EmailFormat

