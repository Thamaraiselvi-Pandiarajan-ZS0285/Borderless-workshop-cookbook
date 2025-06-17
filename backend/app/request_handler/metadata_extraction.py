from pydantic import BaseModel
from typing import List

class EmailImageInput(BaseModel):
    input: str
    file_name: str
    file_extension: str


class EmailImageRequest(BaseModel):
    data: List[EmailImageInput]