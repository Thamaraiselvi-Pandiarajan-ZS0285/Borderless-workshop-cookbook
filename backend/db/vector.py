from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base
from db_helper.db_Initializer import create_engine, sessionmaker

Base = declarative_base()

engine = create_engine(Base)
SessionLocal = sessionmaker(bind=engine)

class EmailChunk(Base):
    __tablename__ = "email_chunks"
    id = Column(Integer, primary_key=True)
    email_id = Column(String, index=True)
    chunk_text = Column(Text)
    embedding = Column(VECTOR(1536))  # match Azure embedding dims

Base.metadata.create_all(bind=engine)
