import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, func, DateTime, Boolean, JSON, Integer
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


class Email(Base):
    __tablename__ = 'emails'
    __table_args__ = {"schema": "public"}

    email_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    subject: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    sender: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    to_recipients: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # comma-separated list
    cc_recipients: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bcc_recipients: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    body_plain: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    body_html: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    attachments_info: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string of attachment metadata
    has_attachments: Mapped[bool] = mapped_column(Boolean, default=False)

    message_id: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    folder: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)  # e.g., Inbox, Processed, Failed
    status: Mapped[Optional[str]] = mapped_column(String(64), default="unprocessed")  # processed / failed / unprocessed

    received_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Email(email_id={self.email_id}, subject={self.subject}, sender={self.sender})>"
