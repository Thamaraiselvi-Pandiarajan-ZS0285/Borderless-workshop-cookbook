import uuid
from datetime import datetime
from sqlalchemy import String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from backend.src.db.models.base_class import Base


"//NOTE: All the base class import should be from the base_class.py"

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