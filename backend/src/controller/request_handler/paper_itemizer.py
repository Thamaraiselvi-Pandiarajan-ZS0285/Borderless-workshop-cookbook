import uuid

from pydantic import BaseModel, Field

class PaperItemizerRequest(BaseModel):
    input: str
    file_name: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_extension: str = ".jpg"