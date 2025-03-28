# models.py
import datetime
import uuid
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID  # Import UUID type
from config import config # Import th config
from pydantic import BaseModel, Field  # For data validation (optional)

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=False, nullable=False)  # Removed unique=True
    sessions = relationship('Session', back_populates='user')


class Session(Base):
    __tablename__ = 'sessions'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  # Generate UUID
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    messages = relationship('Message', back_populates='session', cascade="all, delete-orphan") # Cascade delete
    user = relationship('User', back_populates='sessions')


class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(36), ForeignKey('sessions.id'))
    role = Column(String(20))  # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship('Session', back_populates='messages')

# --- Pydantic Models (Optional, but recommended for validation) ---

class MessageCreate(BaseModel):
    session_id: str
    role: str = Field(..., max_length=20)  # Enforce max_length
    content: str

class MessageResponse(MessageCreate):
    id: int
    timestamp: datetime.datetime

    class Config:
        orm_mode = True  # Enable ORM mode for automatic conversion


# --- Database Setup ---
engine = create_engine(
    config.DATABASE_URL,
    pool_size=10,  # Adjust as needed
    max_overflow=20,  # Adjust as needed
    pool_timeout=30,  # Adjust as needed
    pool_recycle=3600,  # Recycle connections after 1 hour (adjust)
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()