import logging
# --- ELK Logging Import ---
from elk_logging import setup_elk_logging, get_agent_logger
# --- End ELK Logging Import ---
import time
import os
import requests
import openai
from fastapi import FastAPI, Depends, HTTPException, Request
from sqlalchemy.orm import Session
# Assuming models.py defines Session as DBSession to avoid conflict
from models import get_db, Message, MessageCreate, MessageResponse, User, Session as DBSession
from redis_utils import store_chat_history, get_chat_history
from config import config
from dotenv import load_dotenv
from pydantic import BaseModel
import json
from datetime import datetime
import re
from sqlalchemy import desc

# --- Agent Imports ---
from agent.agent_models import AgentState, UserModel
from agent.graph import build_fitness_trainer_graph, agent_checkpointer
from agent.prompts import (
    summarize_routine_prompt, get_memory_consolidation_prompt, get_coordinator_prompt
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.errors import GraphRecursionError


load_dotenv()

# --- Environment Variables & Logging ---
os.environ['LANGSMITH_API_KEY'] = os.environ.get('LANGSMITH_API')
os.environ['LANGSMITH_TRACING'] = os.environ.get('LANGSMITH_TRACING')
os.environ['LANGSMITH_PROJECT'] = os.environ.get('LANGSMITH_PROJECT')

# --- Setup ELK Logging ---
# Configure the main application logger. Send DEBUG level logs and above to Logstash.
# Adjust console_level if you want less verbose console output (e.g., logging.INFO)
log = setup_elk_logging(
    "fitness-chatbot.main",
    console_level=logging.INFO, # Or config.LOGGING_LEVEL
    logstash_level=logging.DEBUG # Send detailed logs to ELK
)
# Keep quieting down noisy libraries if needed
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# --- End Logging Setup ---


# --- FastAPI App and OpenAI Client ---
app = FastAPI()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    # Use the configured logger
    log.critical("OPENAI_API_KEY environment variable not set!")
    raise ValueError("OPENAI_API_KEY environment variable not set!")
client = openai.OpenAI(api_key=api_key)

# --- Build Agent and Get Checkpointer ---
log.info("Building fitness trainer graph...")
try:
    agent_app = build_fitness_trainer_graph()
    if agent_checkpointer is None:
        log.critical("Agent checkpointer instance is None after building graph!")
        raise RuntimeError("Agent checkpointer instance not available after building graph!")
    log.info(f"Graph built successfully. Checkpointer type: {type(agent_checkpointer)}")
except Exception as e:
    log.exception("Failed to build fitness trainer graph!") # Use log.exception to include traceback
    raise # Re-raise the exception to stop the app if graph build fails


# --- Pydantic Models ---
class UserCreate(BaseModel):
    username: str

class UserResponse(BaseModel): # Add a response model for user data
    id: int
    username: str

    class Config:
        orm_mode = True

class SessionCreate(BaseModel):
    user_id: int

class SessionResponse(BaseModel): # Add a response model for session data
    id: str
    user_id: int
    created_at: datetime

    class Config:
        orm_mode = True


async def generate_response(session_id: str, latest_human_message_content: str, db: Session) -> str:
    """
    Invokes the agent graph with persistence using the checkpointer.
    """
    start_time = time.time()
    # --- Use Agent-Specific Logger with Context ---
    agent_log = get_agent_logger("agent_graph", session_id)
    # ---

    agent_log.info(f"Starting generate_response", extra={
        "message_length": len(latest_human_message_content)
    })

    try:
        invoke_input = {"messages": [HumanMessage(content=latest_human_message_content)]}
        agent_log.debug(f"Input for agent invoke prepared", extra={"invoke_input": str(invoke_input)}) # Use agent_log

        config = {"configurable": {"thread_id": session_id, "checkpointer": agent_checkpointer}, "recursion_limit": 100}
        agent_log.debug(f"Invoke config prepared", extra={"config_thread_id": config["configurable"]["thread_id"]}) # Use agent_log, avoid logging sensitive details if any in config

        final_state: AgentState = None
        try:
            agent_log.info("Invoking agent graph...") # Use agent_log
            final_state = await agent_app.ainvoke(invoke_input, config=config)
            duration = time.time() - start_time
            agent_log.info(f"Agent invocation completed successfully.", extra={"duration_seconds": round(duration, 2)}) # Use agent_log

            if isinstance(final_state, dict):
                 agent_log.debug(f"Final agent state keys: {list(final_state.keys())}") # Use agent_log
            else:
                 # Log critical because this shouldn't happen with LangGraph typically
                 agent_log.critical(f"Agent returned non-dict state type: {type(final_state)}", extra={"final_state_type": str(type(final_state))})
                 raise ValueError("Agent did not return a valid state dictionary.")

        except GraphRecursionError as recursion_error:
             duration = time.time() - start_time
             # Log as error, include exception info
             agent_log.error(f"GraphRecursionError caught", exc_info=True, extra={"duration_seconds": round(duration, 2)}) # Use agent_log
             try:
                 checkpoint_data = agent_checkpointer.get(config)
                 if checkpoint_data and hasattr(checkpoint_data, 'channel_values'):
                     final_state = checkpoint_data.channel_values
                     agent_log.info("Retrieved state from checkpointer after GraphRecursionError.") # Use agent_log
                 elif checkpoint_data and isinstance(checkpoint_data, dict):
                      final_state = checkpoint_data
                      agent_log.info("Retrieved dictionary state from checkpointer after GraphRecursionError.") # Use agent_log
                 else:
                      final_state = None
                      agent_log.warning("Could not retrieve valid state from checkpointer after GraphRecursionError.") # Use agent_log
             except Exception as get_err:
                  agent_log.error(f"Error trying to get state after GraphRecursionError", exc_info=True) # Use agent_log
                  final_state = None
             return "I seem to be stuck in a loop. Could you please rephrase your request or try something slightly different?" # User-facing message

        except Exception as invoke_err:
            duration = time.time() - start_time
            # Log as error, include exception info
            agent_log.error(f"Unhandled error during agent invocation", exc_info=True, extra={"duration_seconds": round(duration, 2), "error_type": type(invoke_err).__name__}) # Use agent_log
            # Try to get last known state (optional, might fail)
            try:
                checkpoint_data = agent_checkpointer.get(config)
                if checkpoint_data and hasattr(checkpoint_data, 'channel_values'):
                    final_state = checkpoint_data.channel_values
                elif checkpoint_data and isinstance(checkpoint_data, dict):
                    final_state = checkpoint_data
                else:
                    final_state = None
                if final_state:
                    agent_log.info("Retrieved state from checkpointer after unhandled invocation error.") # Use agent_log
                else:
                    agent_log.warning("Could not retrieve state from checkpointer after unhandled invocation error.") # Use agent_log
            except Exception as get_err:
                agent_log.error(f"Error trying to get state after unhandled invocation error", exc_info=True) # Use agent_log

            return f"Sorry, I encountered an internal error ({type(invoke_err).__name__}). Please try again later." # User-facing message


        # --- Extract the response ---
        all_messages = final_state.get("messages", []) if final_state else [] # Handle potential None final_state
        assistant_response = "I'm sorry, I couldn't generate a response for that." # Default

        if all_messages and isinstance(all_messages[-1], AIMessage):
             assistant_response = all_messages[-1].content
             agent_log.debug("Extracted AIMessage content as response.") # Use agent_log
             # Extract content within <user> tags if present
             # Ensure 're' is imported and available
             if isinstance(re.search, type(lambda: 0)): # Check if re.search is valid (a bit defensive)
                 user_match = re.search(r'<user>(.*?)</user>', assistant_response, re.DOTALL)
                 if user_match:
                     assistant_response = user_match.group(1).strip()
                     agent_log.debug("Extracted content from <user> tags.") # Use agent_log
             else:
                 agent_log.error("'re' module or 're.search' seems unavailable for tag extraction.") # Use agent_log

        elif final_state: # Only check further if we have a final_state
             agent_log.warning(f"No AIMessage found at the end of the state messages. Last msg type: {type(all_messages[-1]).__name__ if all_messages else 'None'}", extra={"last_message": str(all_messages[-1]) if all_messages else None}) # Use agent_log
             if final_state.get("agent_state", {}).get("needs_human_input"):
                 assistant_response = "I need more information to proceed. Could you please provide details?"
                 agent_log.info("Setting response asking for more info as agent needs human input.") # Use agent_log
        else:
            agent_log.error("Final state was None, cannot extract response.") # Use agent_log


        agent_log.info(f"Final assistant response prepared.", extra={"response_length": len(assistant_response)}) # Use agent_log

        # State Saving is handled by the checkpointer

        duration = time.time() - start_time
        agent_log.info("generate_response finished.", extra={"total_duration_seconds": round(duration, 2)}) # Use agent_log
        return assistant_response

    except Exception as e:
        duration = time.time() - start_time
        # Log critical error with traceback using the agent_log which has session_id context
        agent_log.critical(f"Critical error in generate_response", exc_info=True, extra={"total_duration_seconds": round(duration, 2), "error_type": type(e).__name__})
        # Re-raise as HTTPException for FastAPI
        raise HTTPException(status_code=500, detail=f"Internal server error: {type(e).__name__}")


@app.post("/messages/", response_model=MessageResponse)
async def create_message(message_data: MessageCreate, request: Request, db: Session = Depends(get_db)):
    """
    Handles a new user message, generates a response, and stores both.
    """
    # --- Create a logger with request context ---
    # Use the main 'log' instance and add context for this specific request
    request_log = log.add_context(
        session_id=message_data.session_id,
        request_path=request.url.path,
        request_method=request.method
        # Add user_id here if easily obtainable from session_id or request headers/token
    )
    # ---

    request_log.info(f"Received message, role: {message_data.role}") # Use request_log
    if message_data.role.lower() != "user":
         request_log.warning("Received message with non-user role.", extra={"role": message_data.role}) # Use request_log
         raise HTTPException(status_code=400, detail="Only user messages can be posted.")

    # --- 1. Store User Message ---
    db_message = Message(**message_data.dict())
    db.add(db_message)
    try:
        db.commit()
        db.refresh(db_message)
        request_log.info(f"User message stored in PostgreSQL.", extra={"message_id": db_message.id}) # Use request_log
    except Exception as e:
        db.rollback()
        request_log.error(f"Database error storing user message", exc_info=True) # Use request_log
        raise HTTPException(status_code=500, detail="Database error storing user message.")

    # Store in Redis
    try:
        existing_history = get_chat_history(message_data.session_id)
        new_history = existing_history + [f"{message_data.role}: {message_data.content}"]
        store_chat_history(message_data.session_id, new_history)
        request_log.info(f"User message added to Redis history.", extra={"history_length": len(new_history)}) # Use request_log
    except Exception as e:
        request_log.error(f"Redis error storing user message", exc_info=True) # Use request_log
        # Continue even if Redis fails, but log it

    # --- 2. Generate Response ---
    try:
        # generate_response now uses its own agent_logger with session_id context
        assistant_response_content = await generate_response(message_data.session_id, message_data.content, db)
    except HTTPException as e:
        # The error should have been logged within generate_response
        raise e # Re-raise FastAPI exceptions
    except Exception as e:
        # This catches errors *calling* generate_response, not *inside* it (those are caught within)
        request_log.error(f"Unexpected error calling generate_response", exc_info=True) # Use request_log
        raise HTTPException(status_code=500, detail="Error generating response.")

    # --- 3. Store Assistant Message ---
    assistant_message_data = MessageCreate(session_id=message_data.session_id, role="assistant", content=assistant_response_content)
    db_assistant_message = Message(**assistant_message_data.dict())
    db.add(db_assistant_message)
    try:
        db.commit()
        db.refresh(db_assistant_message)
        request_log.info(f"Assistant message stored in PostgreSQL.", extra={"message_id": db_assistant_message.id}) # Use request_log
    except Exception as e:
        db.rollback()
        # Log the error, but don't fail the request since we have the response
        request_log.error(f"Database error storing assistant message", exc_info=True) # Use request_log

    # Store in Redis
    try:
        # new_history should still hold user message + previous history
        final_history = new_history + [f"assistant: {assistant_response_content}"]
        store_chat_history(message_data.session_id, final_history)
        request_log.info(f"Assistant message added to Redis history.", extra={"history_length": len(final_history)}) # Use request_log
    except Exception as e:
        request_log.error(f"Redis error storing assistant message", exc_info=True) # Use request_log

    request_log.info("Message request processed successfully.") # Use request_log
    return db_assistant_message


@app.get("/messages/{session_id}", response_model=list[MessageResponse])
async def get_messages(session_id: str, request: Request, db: Session = Depends(get_db)):
    """
    Retrieves messages for a given session ID from PostgreSQL.
    """
    # --- Create a logger with request context ---
    request_log = log.add_context(
        session_id=session_id,
        request_path=request.url.path,
        request_method=request.method
    )
    # ---
    request_log.info("Attempting to retrieve messages from PostgreSQL.") # Use request_log
    messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.timestamp).all()

    if not messages:
        request_log.info("No messages found in PostgreSQL, checking Redis.") # Use request_log
        # try:
        #     redis_messages = get_chat_history(session_id=session_id)
        #     if not redis_messages:
        #         request_log.warning("Messages not found in PostgreSQL or Redis.") # Use request_log
        #         raise HTTPException(status_code=404, detail="Messages not found for this session")
        #     else:
        #         # Can't return MessageResponse from Redis strings easily
        #         request_log.info("Messages found in Redis cache (returning empty list as format differs).") # Use request_log
        #         # Consider if you want to parse redis strings back into MessageResponse objects
        #         # For now, returning empty list as per original logic when only Redis hit
        #         return []
        # except Exception as e:
        #     request_log.error("Error checking Redis for messages.", exc_info=True) # Use request_log
        #     # Raise 404 as we couldn't confirm history exists
        raise HTTPException(status_code=404, detail="Messages not found for this session in the primary database")

    request_log.info(f"Retrieved {len(messages)} messages from PostgreSQL.") # Use request_log
    return messages

@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate, request: Request, db: Session = Depends(get_db)):
    """Creates a new user."""
    # --- Create a logger with request context ---
    request_log = log.add_context(
        request_path=request.url.path,
        request_method=request.method,
        username_provided=user.username # Add context specific to this endpoint
    )
    # ---
    try:
        # Log the request body safely (if not too large or sensitive)
        # request_body = await request.json() # Already read if using log.debug below
        # request_log.debug(f"Received request body: {request_body}")
        pass # Avoid logging raw body unless necessary and safe
    except Exception:
        request_log.warning("Could not read request body for logging.")

    request_log.info(f"Attempting to create user.") # Use request_log

    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        request_log.warning(f"Username already exists.") # Use request_log
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = User(username=user.username)
    db.add(new_user)
    try:
        db.commit()
        db.refresh(new_user)
        request_log.info(f"User created successfully.", extra={"user_id": new_user.id}) # Use request_log
        return {"user_id": new_user.id, "username": new_user.username}
    except Exception as e:
        db.rollback()
        request_log.error("Database error creating user.", exc_info=True) # Use request_log
        raise HTTPException(status_code=500, detail="Database error creating user")
    
# --- NEW: Endpoint to get user by username ---
@app.get("/users/username/{username}", response_model=UserResponse)
async def get_user_by_username(username: str, request: Request, db: Session = Depends(get_db)):
    """Gets a user by their username."""
    request_log = log.add_context(
        request_path=request.url.path,
        request_method=request.method,
        username_query=username
    )
    request_log.info("Attempting to find user by username.")
    user = db.query(User).filter(User.username == username).first()
    if user:
        request_log.info("User found.", extra={"user_id": user.id})
        return user
    else:
        request_log.info("User not found.")
        raise HTTPException(status_code=404, detail="User not found")


@app.post("/sessions/", response_model=SessionResponse)
async def create_session(session: SessionCreate, request: Request, db: Session = Depends(get_db)):
    """Creates a new session for a user."""
    # --- Create a logger with request context ---
    request_log = log.add_context(
        request_path=request.url.path,
        request_method=request.method,
        user_id_provided=session.user_id # Add context specific to this endpoint
    )
    # ---
    try:
        # request_body = await request.json() # Already read if using log.debug below
        # request_log.debug(f"Received request body: {request_body}")
        pass # Avoid logging raw body unless necessary and safe
    except Exception:
        request_log.warning("Could not read request body for logging.")

    request_log.info(f"Attempting to create session.") # Use request_log

    user = db.query(User).filter(User.id == session.user_id).first()
    if not user:
        request_log.warning(f"User not found.") # Use request_log
        raise HTTPException(status_code=404, detail="User not found")

    new_session = DBSession(user_id=session.user_id) # DBSession is your model
    db.add(new_session)
    try:
        db.commit()
        db.refresh(new_session)
        # Add the newly created session_id to the context for the final log message
        request_log.add_context(session_id=new_session.id)
        request_log.info(f"Session created successfully.") # Use request_log
        return new_session
    except Exception as e:
        db.rollback()
        request_log.error("Database error creating session.", exc_info=True) # Use request_log
        raise HTTPException(status_code=500, detail="Database error creating session")
    
# --- NEW: Endpoint to get the latest session for a user ---
@app.get("/users/{user_id}/sessions/latest", response_model=SessionResponse)
async def get_latest_session(user_id: int, request: Request, db: Session = Depends(get_db)):
    """Gets the most recent session for a given user ID."""
    request_log = log.add_context(
        request_path=request.url.path,
        request_method=request.method,
        user_id_query=user_id
    )
    request_log.info("Attempting to find latest session for user.")

    latest_session = db.query(DBSession).filter(DBSession.user_id == user_id)\
                       .order_by(desc(DBSession.created_at)).first()

    if latest_session:
        request_log.info(f"Latest session found.", extra={"session_id": latest_session.id})
        return latest_session
    else:
        request_log.info("No sessions found for this user.")
        raise HTTPException(status_code=404, detail="No sessions found for this user")


# --- Example Usage / Initialization ---
if __name__ == "__main__":
    # This part usually runs only when executing the file directly, not via uvicorn
    # It's often used for setup tasks like DB migrations (though Alembic is better)
    log.info("Running main.py directly (likely for setup checks or tests).")
    try:
        from models import Base, engine
        # Base.metadata.create_all(bind=engine) # Generally use Alembic for migrations
        log.info("Database connection seems okay (table creation skipped/handled by Alembic).")
        print("\n--- ELK Logging Configured ---")
        print(f"Attempting to send logs to: {os.environ.get('LOGSTASH_HOST', 'localhost')}:{os.environ.get('LOGSTASH_PORT', 5000)}")
        print("If running app in Docker, ensure LOGSTASH_HOST is the service name (e.g., 'logstash').")
        print("If running app locally, ensure Logstash port is exposed from Docker (e.g., docker-compose port 5000:5000).")
        print("\nRun 'docker-compose -f docker-compose.elk.yaml up -d' to start ELK.")
        print("Run 'uvicorn main:app --host 0.0.0.0 --port 8000 --reload' to start the API.") # Example command

        # Send a test log
        log.info("Test log message from __main__ block.", extra={"test_source": "__main__"})

    except ImportError as e:
        log.error(f"ImportError in __main__: {e}. Ensure models.py and DB components are correctly set up.")
        print(f"ImportError in __main__: {e}. Check your models.py and database setup.")
    except Exception as e:
        log.exception("Error during __main__ execution.") # Log full traceback
        print(f"An error occurred during __main__ execution: {e}")