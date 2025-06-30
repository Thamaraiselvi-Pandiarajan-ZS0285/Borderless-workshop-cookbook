from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_name: str
    user_input: str