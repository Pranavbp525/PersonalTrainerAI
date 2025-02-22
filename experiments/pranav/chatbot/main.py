import logging
import os
import requests
import openai  # Import the OpenAI library
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from models import get_db, Message, MessageCreate, MessageResponse, User, Session as DBSession
from redis_utils import store_chat_history, get_chat_history
from config import config
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import Request
# Define a Pydantic model for user creation
class UserCreate(BaseModel):
    username: str

class SessionCreate(BaseModel):
    user_id: int



load_dotenv()

logging.basicConfig(level=config.LOGGING_LEVEL)
log = logging.getLogger(__name__)

app = FastAPI()

api_key = os.environ.get("OPENAI_API_KEY")
# --- OpenAI API Setup ---
client = openai.OpenAI(api_key=api_key)


if not client.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set!")
# --- (Optional) Vector Database Integration Placeholder ---
def retrieve_context_from_vector_db(user_input: str) -> list[str]:
    """
    (Placeholder) Retrieves relevant context from a vector database.

    Args:
        user_input: The user's input message.

    Returns:
        A list of relevant message IDs (from PostgreSQL) or an empty list.
    """
    # 1. Convert user_input to an embedding.
    # 2. Query the vector database for similar embeddings.
    # 3. Extract the associated message IDs.
    # 4. Return the list of message IDs.
    log.info("Retrieving Context")
    return []  # Replace with actual vector database query


async def generate_response(session_id: str, db: Session) -> str:
    try:
        messages_for_openai = []
        log.debug(f"Attempting to retrieve chat history from Redis for session: {session_id}") # Log attempt
        chat_history = get_chat_history(session_id)
        if not chat_history:
            log.info(f"Chat history not found in Redis for session: {session_id}.  Falling back to PostgreSQL.") # Log fallback
            db_messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.timestamp).all()
            chat_history = [f"{msg.role}: {msg.content}" for msg in db_messages]

        if chat_history:
            most_recent_user_message = ""
            for message in reversed(chat_history):
                if message.startswith("user:"):
                    most_recent_user_message = message.split(":", 1)[1].strip()
                    break
            relevant_message_ids = retrieve_context_from_vector_db(most_recent_user_message)
            if relevant_message_ids:
                context_messages = db.query(Message).filter(Message.id.in_(relevant_message_ids)).order_by(Message.timestamp).all()
                context_history = [f"{msg.role}: {msg.content}" for msg in context_messages]
                for message in reversed(context_history):
                    chat_history.insert(0, message)

            for message_text in chat_history:
                role, content = message_text.split(":", 1)
                messages_for_openai.append({"role": role.strip().lower(), "content": content.strip()})

        # Use the new OpenAI client
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_for_openai,
            max_tokens=150,
            temperature=0.7,
        )
        assistant_response = response.choices[0].message.content.strip()
        return assistant_response

    except Exception as e:
        log.error(f"Error in OpenAI API call: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in response generation: {str(e)}")

@app.post("/messages/", response_model=MessageResponse)
async def create_message(message_data: MessageCreate, db: Session = Depends(get_db)):
    """
    Handles a new user message, generates a response, and stores both.
    """

    # --- 1. Store User Message (PostgreSQL and Redis) ---
    db_message = Message(**message_data.dict())
    db.add(db_message)
    try:
        db.commit()
        db.refresh(db_message)
        log.info(f"User message stored in PostgreSQL for session: {message_data.session_id}") # Add log here
    except Exception as e:
        db.rollback()
        log.error(f"Database error (user message): {e}")
        raise HTTPException(status_code=500, detail="Database error")

    existing_history = get_chat_history(message_data.session_id)
    new_history = existing_history + [f"{message_data.role}: {message_data.content}"]
    store_chat_history(message_data.session_id, new_history)

    # --- 2. Generate Response (OpenAI) ---
    try:
        assistant_response = await generate_response(message_data.session_id, db)
    except HTTPException as e:  # Re-raise HTTP exceptions from generate_response
        raise e
    except Exception as e:
        log.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")

    # --- 3. Store Assistant Message (PostgreSQL and Redis) ---
    assistant_message_data = MessageCreate(session_id=message_data.session_id, role="assistant", content=assistant_response)
    db_assistant_message = Message(**assistant_message_data.dict())
    db.add(db_assistant_message)
    try:
        db.commit()
        db.refresh(db_assistant_message)
        log.info(f"Assistant message stored in PostgreSQL for session: {message_data.session_id}") # Add log here
    except Exception as e:
        db.rollback()
        log.error(f"Database error (assistant message): {e}")
        raise HTTPException(status_code=500, detail="Database error")

    new_history = new_history + [f"assistant: {assistant_response}"] #Append assistant response
    store_chat_history(message_data.session_id, new_history)

    return db_assistant_message  # Return the *assistant's* message (or user's, if you prefer)


@app.get("/messages/{session_id}", response_model=list[MessageResponse])
async def get_messages(session_id: str, db: Session = Depends(get_db)):
    """
    Retrieves messages for a given session ID from PostgreSQL.
    """
    messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.timestamp).all()
    if not messages:
      #Try to get from redis cache
        redis_messages = get_chat_history(session_id=session_id)
        if not redis_messages:
            raise HTTPException(status_code=404, detail="Messages not found")
        return [] # Redis only stores strings. Can't use MessageResponse model
    return messages
    
@app.post("/users/", response_model=dict)
async def create_user(user: UserCreate, request: Request, db: Session = Depends(get_db)):
    # Check if username is unique
    log.debug(f"Received request body: {await request.json()}")

    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = User(username=user.username)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"user_id": new_user.id, "username": new_user.username}


@app.post("/sessions/", response_model=dict)
async def create_session(session: SessionCreate, request: Request, db: Session = Depends(get_db)):
    # Check if user exists
    log.debug(f"Received request body: {await request.json()}")
    user = db.query(User).filter(User.id == session.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_session = DBSession(user_id=session.user_id)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return {"session_id": new_session.id, "user_id": new_session.user_id}

# --- Example Usage (Without FastAPI - for initialization) ---
if __name__ == "__main__":
    from .models import Base, engine
    # Base.metadata.create_all(bind=engine)  # Create tables (use Alembic!)
    print("Database tables created.  Run 'uvicorn main:app --reload' to start the API.")