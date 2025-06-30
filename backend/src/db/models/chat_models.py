from typing import List

from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, func
from sqlalchemy.orm import relationship, Mapped, mapped_column
from backend.src.db.models.base_class import Base
from datetime import datetime

CHAT_SCHEMA = "ChatHistoryAndSessionManagement"


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = {"schema": CHAT_SCHEMA}
    id: Mapped[int] = mapped_column(primary_key=True)
    session_name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    messages: Mapped[List["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = {"schema": CHAT_SCHEMA}
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey(f"{CHAT_SCHEMA}.chat_sessions.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    session: Mapped["ChatSession"] = relationship(back_populates="messages")