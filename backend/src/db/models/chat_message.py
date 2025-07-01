from sqlalchemy import String,Integer,Text,DateTime,func
from datetime import datetime

from backend.src.db.models.base_class import Base
from sqlalchemy.orm import Mapped, mapped_column


class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    __table_args__ = {"schema": "ChatHistory"}
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    role: Mapped[str] = mapped_column(String) #user, assistant, system
    sender: Mapped[str] = mapped_column(String, nullable=True) # agent name
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_index:Mapped[int] = mapped_column(Integer, nullable = False, default=0)
    metadata_in:Mapped[str] = mapped_column(String, nullable = True)
    created_on: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), server_default=func.now())
