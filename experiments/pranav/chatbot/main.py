# main.py
import logging
from fastapi import FastAPI, Depends, HTTPException  # Using FastAPI for a simple API
from sqlalchemy.orm import Session
from .models import get_db, Message, MessageCreate, MessageResponse, User, Session as DBSession # Renamed Session
from .redis_utils import store_chat_history, get_chat_history
from .config import config
import uuid

logging.basicConfig(level=config.LOGGING_LEVEL)
log = logging.getLogger(__name__)

app = FastAPI()


@app.post("/messages/", response_model=MessageResponse)
async def create_message(message_data: MessageCreate, db: Session = Depends(get_db)):
    """
    Creates a new message and stores it in both PostgreSQL and Redis.
    """

    # Check if the session exists
    db_session = db.query(DBSession).filter(DBSession.id == message_data.session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Create the new message in PostgreSQL
    db_message = Message(**message_data.dict())
    db.add(db_message)
    try:
        db.commit()
        db.refresh(db_message)
    except Exception as e:  # Catch database errors
        db.rollback()
        log.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    # Update Redis (get existing messages, append new one, store)
    existing_history = get_chat_history(message_data.session_id)
    new_history = existing_history + [f"{message_data.role}: {message_data.content}"]
    store_chat_history(message_data.session_id, new_history)

    return db_message

@app.get("/messages/{session_id}", response_model=list[MessageResponse])
async def get_messages(session_id: str, db: Session = Depends(get_db)):
    """
    Retrieves messages for a given session ID from PostgreSQL.
    """
    messages = db.query(Message).filter(Message.session_id == session_id).all()
    if not messages:
      #Try to get from redis cache
        redis_messages = get_chat_history(session_id=session_id)
        if not redis_messages:
            raise HTTPException(status_code=404, detail="Messages not found")
        return [] # Redis only stores strings. Can't use MessageResponse model
    return messages
    
@app.post("/users/", response_model=dict)
async def create_user(username: str, db: Session = Depends(get_db)):
    #Check if username is unique
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = User(username=username)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"user_id": new_user.id, "username": new_user.username}


@app.post("/sessions/", response_model=dict)
async def create_session(user_id: int, db: Session = Depends(get_db)):

    #Check if user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    new_session = DBSession(user_id = user_id)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return {"session_id": new_session.id, "user_id": new_session.user_id}
    

# --- Example Usage (Without FastAPI) ---
if __name__ == "__main__":
    from .models import Base, engine

    Base.metadata.create_all(bind=engine)  # Create tables (use Alembic in production!)
    # You can add some manual testing/interaction here if you don't run with FastAPI.
    # ... (Example: create a user, session, and messages)
    print("Database tables created.  Run 'uvicorn main:app --reload' to start the API.")