from pydantic import BaseModel

class EmailClassificationRequest(BaseModel):
    subject: str
    body: str