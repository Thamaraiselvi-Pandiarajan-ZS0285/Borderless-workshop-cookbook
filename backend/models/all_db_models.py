import uuid
from datetime import datetime
from sqlalchemy import String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass

class EmailChunk(Base):
    __tablename__ = "email_chunks"
    __table_args__ = {"schema": "public"}

    email_chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email_id: Mapped[str] = mapped_column(String, index=True)
    content_id: Mapped[str] = mapped_column(Text)
    chunk_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    created_on: Mapped[datetime] = mapped_column(
        server_default=func.now()
    )

