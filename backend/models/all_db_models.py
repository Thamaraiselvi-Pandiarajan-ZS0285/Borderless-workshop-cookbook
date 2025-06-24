import uuid
from datetime import datetime
from sqlalchemy import String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from sqlalchemy.sql.sqltypes import Integer, DateTime

from backend.config.db_config import *
from backend.db.db_helper.db_Initializer import DbInitializer


class Base(DeclarativeBase):
    pass

db_init = DbInitializer(
            POSTGRESQL_DRIVER_NAME, POSTGRESQL_HOST, POSTGRESQL_DB_NAME,
            POSTGRESQL_USER_NAME, POSTGRESQL_PASSWORD, POSTGRESQL_PORT_NO
        )

# class EmailChunk(Base):
#     __tablename__ = "email_chunks"
#     __table_args__ = {"schema": "EmailChunks"}
#
#     email_chunk_id: Mapped[uuid.UUID] = mapped_column(
#         UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
#     )
#     email_id: Mapped[str] = mapped_column(String, index=True)
#     chunk_text: Mapped[str] = mapped_column(Text)
#     embedding: Mapped[list[float]] = mapped_column(Vector(1536))
#     created_on: Mapped[datetime] = mapped_column(
#         server_default=func.now()
#     )

class EmailContentEmbedding(Base):
    __tablename__ = "email_content_embeddings"
    __table_args__ = {"schema": "Embeddings"}

    content_embedding_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email_id: Mapped[str] = mapped_column(String, index=True)
    email_content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    created_on: Mapped[datetime] = mapped_column(
        server_default=func.now()
    )

class MetadataExtractionJsonEmbedding(Base):
    __tablename__ = "metadata_extraction_json_embeddings"
    __table_args__ = {"schema": "Embeddings"}

    json_embedding_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email_id: Mapped[str] = mapped_column(String, index=True)
    json_content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    created_on: Mapped[datetime] = mapped_column(
        server_default=func.now()
    )

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
