import uuid
from datetime import datetime

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class EmailChunk(Base):
    __tablename__ = "email_chunks"
    __table_args__ = {"schema": "EmailChunks"}

    email_chunk_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email_id: Mapped[str] = mapped_column(String, index=True)
    chunk_text: Mapped[str] = mapped_column(Text(1024))
    embedding: Mapped[list[float]] = mapped_column(Text(1536))
    created_on: Mapped[datetime] = mapped_column(
        default=func.now(), server_default=func.now()
    )


