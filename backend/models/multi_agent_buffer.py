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

Base = DeclarativeBase()
db_init = DbInitializer(
            POSTGRESQL_DRIVER_NAME, POSTGRESQL_HOST, POSTGRESQL_DB_NAME,
            POSTGRESQL_USER_NAME, POSTGRESQL_PASSWORD, POSTGRESQL_PORT_NO
        )

# Define ORM model for message persistence
class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    __table_args__ = {"schema": "public"}
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    created_on: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=func.now(), server_default=func.now())

class MultiAgentBuffer(ABC):
    def __init__(self, conversation_id:str=None, buffer_size:int=None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.db_engine = db_init.db_create_engin()
        self.session = db_init.db_create_session()

        # Initialize memory and model context
        self.memory = ListMemory(name=f"memory_{self.conversation_id}")
        self.context = BufferedChatCompletionContext(buffer_size=buffer_size)

        # Load history and sync with context
        asyncio.run(self.load_and_sync_memory())

    async def load_and_sync_memory(self):
        """Load history from DB and update model context"""
        history = self._load_message_history()
        for msg in history:
            await self.memory.add(MemoryContent(content=str(msg["content"]), mime_type="text/plain"))
        await self.memory.update_context(self.context)

    def _load_message_history(self):
        session = Session()
        rows = session.query(ChatMessage).filter_by(conversation_id=self.conversation_id).all()
        return [{"role": msg.role, "content": msg.content} for msg in rows]

    def _save_message_to_db(self, role: str, content: str):
        db_msg = ChatMessage(conversation_id=self.conversation_id,
            role=role,
            content=content)
        session = Session()
        session.add(db_msg)
        session.commit()

    @abstractmethod
    async def on_message(self, message: str, sender: str):
        pass

    async def persist_message(self, role: str, content: str):
        """Add message to both buffer (context) and database"""
        await self.memory.add(MemoryContent(content=content, mime_type="text/plain"))
        await self.memory.update_context(self.context)
        self._save_message_to_db(role, content)

    def get_context_messages(self):
        """Get current buffered messages from model context"""
        return asyncio.run(self.context.get_messages())

    def close(self):
        session = Session()
        session.close()

