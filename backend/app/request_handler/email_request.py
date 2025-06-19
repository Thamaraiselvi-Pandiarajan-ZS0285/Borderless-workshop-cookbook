from pydantic import BaseModel

class EmailClassificationRequest(BaseModel):
    subject: str
    body: str
    sender:str
    received_at:str
    attachments:list