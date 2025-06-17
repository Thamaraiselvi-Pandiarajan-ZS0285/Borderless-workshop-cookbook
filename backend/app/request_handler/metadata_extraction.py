from pydantic import BaseModel
from typing import List

class EmailImageInput(BaseModel):
    input: str
    filename: str
    fileextension: str


class EmailImageRequest(BaseModel):
    data: List[EmailImageInput]