from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass

class MemoryLog(Base):
    __tablename__ = 'memory_logs'

    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    agent_name = Column(String)
    step = Column(Integer)
    input_text = Column(Text)
    output_text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    context = Column(Text)
