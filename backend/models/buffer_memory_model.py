from abc import ABC,abstractmethod
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.memory import ListMemory, MemoryContent
from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import  DeclarativeBase, Mapped, mapped_column
import uuid
import datetime
from sqlalchemy import DateTime
from sqlalchemy.orm.session import Session
from sqlalchemy.sql.functions import func
from backend.db.db_helper.db_Initializer import DbInitializer
from backend.config.db_config import *
import asyncio

db_init = DbInitializer(
            POSTGRESQL_DRIVER_NAME, POSTGRESQL_HOST, POSTGRESQL_DB_NAME,
            POSTGRESQL_USER_NAME, POSTGRESQL_PASSWORD, POSTGRESQL_PORT_NO
        )

class Base(DeclarativeBase):
    pass


# Define ORM model for message persistence
class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    __table_args__ = {"schema": "public"}
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    created_on: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=func.now(), server_default=func.now())


