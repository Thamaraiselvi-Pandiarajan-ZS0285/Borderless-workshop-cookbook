import uuid
from datetime import datetime
from sqlalchemy import String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

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
