# models.py
import datetime
import uuid
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, create_engine, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, ARRAY  # Import UUID and ARRAY types
from config import config # Import the config
from pydantic import BaseModel, Field  # For data validation (optional)

Base = declarative_base()

# frp1 --------------------- start
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    sessions = relationship('Session', back_populates='user')

class Session(Base):
    __tablename__ = 'sessions'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  # Generate UUID
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    logged_out_at = Column(DateTime, nullable=True)
    messages = relationship('Message', back_populates='session', cascade="all, delete-orphan") # Cascade delete
    user = relationship('User', back_populates='sessions')


class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(36), ForeignKey('sessions.id'))
    role = Column(String(20))  # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    thumbs_up = Column(Boolean, nullable=True, default=None)
    session = relationship('Session', back_populates='messages')
# -------------------------- frp1 - end

class ExerciseTemplate(Base):
    __tablename__ = 'exercise_templates'
    id = Column(String(50), primary_key=True)
    title = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)
    equipment = Column(String(50), nullable=False)
    primary_muscle_group = Column(String(50), nullable=False)
    secondary_muscle_groups = Column(ARRAY(String(50)))
    is_custom = Column(Boolean, default=False)


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