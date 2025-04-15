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
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
# Assuming models.py defines Session as DBSession to avoid conflict
from models import get_db, Message, MessageCreate, MessageResponse, User, Session as DBSession
from redis_utils import store_chat_history, get_chat_history
from config import config
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
from datetime import datetime

import re
from sqlalchemy import desc
import asyncio # Import asyncio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# --- Agent Imports ---
from agent.agent_models import AgentState, UserModel
from agent.graph import build_fitness_trainer_graph
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
    console_level_str="INFO",     # Use console_level_str="DEBUG" for more console verbosity
    logstash_level_str="DEBUG"    # Send DEBUG and above to Logstash
)
# Keep quieting down noisy libraries if needed
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# --- End Logging Setup ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Application startup: Initializing resources...")
    app.state.db_pool = None
    app.state.agent_checkpointer = None
    app.state.agent_app = None

    try:
        # --- Create DB Connection Pool ---
        log.info("Creating DB connection pool...")
        app.state.db_pool = AsyncConnectionPool(
            conninfo=config.DATABASE_URL, min_size=1, max_size=10
        )
        log.info("DB connection pool created.")

        # --- Initialize Checkpointer passing pool as POSITIONAL argument ---
        log.info("Initializing AsyncPostgresSaver with pool...")
        # Pass the pool object directly as the first argument
        app.state.agent_checkpointer = AsyncPostgresSaver(app.state.db_pool) # <--- PASS POOL DIRECTLY
        log.info(f"AsyncPostgresSaver instance created: {type(app.state.agent_checkpointer)}")

        # --- Setup Checkpointer Database Schema ---
        log.info("Running checkpointer database setup...")
        await app.state.agent_checkpointer.setup() # Call setup on the instance
        log.info("AsyncPostgresSaver setup() executed successfully.")

        # --- Store Checkpointer Instance ---
        log.info(f"Checkpointer instance stored in app.state: {type(app.state.agent_checkpointer)}")

        # --- Build Agent Graph ---
        log.info("Building fitness trainer graph at startup...")
        app.state.agent_app = build_fitness_trainer_graph(checkpointer=app.state.agent_checkpointer)
        if app.state.agent_app is None:
            raise RuntimeError("build_fitness_trainer_graph returned None")
        log.info(f"Graph built successfully using checkpointer: {type(app.state.agent_checkpointer)}")

        log.info("Application startup complete.")

    except Exception as startup_err:
        log.exception("FATAL Error during application startup!")
        if hasattr(app.state, 'db_pool') and app.state.db_pool:
            log.info("Closing DB pool due to startup error...")
            await app.state.db_pool.close()
        raise RuntimeError("Application startup failed") from startup_err

    yield # Application runs here
    # --- Shutdown logic (remains the same) ---
    log.info("Application shutdown: Cleaning up resources...")
    if hasattr(app.state, 'db_pool') and app.state.db_pool:
        log.info("Closing DB connection pool...")
        await app.state.db_pool.close()
        log.info("DB connection pool closed.")
    app.state.db_pool = None
    app.state.agent_checkpointer = None
    app.state.agent_app = None
    log.info("Application shutdown complete.")

# --- Initialize FastAPI App WITH Lifespan ---
app = FastAPI(lifespan=lifespan)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    # Use the configured logger
    log.critical("OPENAI_API_KEY environment variable not set!")
    raise ValueError("OPENAI_API_KEY environment variable not set!")
client = openai.OpenAI(api_key=api_key)

# # --- Build Agent and Get Checkpointer ---
# log.info("Building fitness trainer graph...")
# try:
#     agent_app = build_fitness_trainer_graph()
#     if agent_checkpointer is None:
#         log.critical("Agent checkpointer instance is None after building graph!")
#         raise RuntimeError("Agent checkpointer instance not available after building graph!")
#     log.info(f"Graph built successfully. Checkpointer type: {type(agent_checkpointer)}")
# except Exception as e:
#     log.exception("Failed to build fitness trainer graph!") # Use log.exception to include traceback
#     raise # Re-raise the exception to stop the app if graph build fails


# --- Pydantic Models ---
class MessageCreate(BaseModel):
    session_id: str
    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str

class MessageResponse(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    timestamp: datetime

    class Config:
        # orm_mode = True # OLD
        from_attributes = True # NEW (Pydantic v2)

class UserCreate(BaseModel):
    username: str

class UserResponse(BaseModel):
    id: int
    username: str

    class Config:
        # orm_mode = True # OLD
        from_attributes = True # NEW (Pydantic v2)

class SessionCreate(BaseModel):
    user_id: int

class SessionResponse(BaseModel):
    id: str # Session ID is string (UUID)
    user_id: int
    created_at: datetime

    class Config:
        # orm_mode = True # OLD
        from_attributes = True # NEW (Pydantic v2)

async def generate_response(session_id: str, latest_human_message_content: str, db: Session, request: Request) -> str: # Add request
    """
    Invokes the agent graph with persistence using the checkpointer.
    Retrieves the compiled agent_app from app state.
    """
    start_time = time.time()
    agent_log = get_agent_logger("agent_graph", session_id)
    agent_log.info(f"Starting generate_response", extra={"message_length": len(latest_human_message_content)})

    # --- Get compiled agent_app from app state ---
    if not hasattr(request.app.state, 'agent_app') or request.app.state.agent_app is None:
        agent_log.critical("Agent graph (agent_app) not found in app state!")
        raise HTTPException(status_code=500, detail="Internal Server Error: Agent not initialized.")
    agent_app = request.app.state.agent_app

    # --- Get checkpointer INSTANCE from app state ---
    if not hasattr(request.app.state, 'agent_checkpointer') or request.app.state.agent_checkpointer is None:
        agent_log.critical("Agent checkpointer instance not found in app state!")
        raise HTTPException(status_code=500, detail="Internal Server Error: Checkpointer not initialized.")
    checkpointer_instance = request.app.state.agent_checkpointer
    # ---

    try:
        invoke_input = {"messages": [HumanMessage(content=latest_human_message_content)]}
        agent_log.debug(f"Input for agent invoke prepared", extra={"invoke_input": str(invoke_input)})

        # --- Use the checkpointer INSTANCE from app state in the config ---
        config = {"configurable": {"thread_id": session_id, "checkpointer": checkpointer_instance}, "recursion_limit": 100}
        agent_log.debug(f"Invoke config prepared", extra={"config_thread_id": config["configurable"]["thread_id"]})

        final_state: AgentState = None
        try:
            agent_log.info("Invoking agent graph...")
            final_state = await agent_app.ainvoke(invoke_input, config=config)
            # ---
            duration = time.time() - start_time
            agent_log.info(f"Agent invocation completed successfully.", extra={"duration_seconds": round(duration, 2)})

            if isinstance(final_state, dict):
                 agent_log.debug(f"Final agent state keys: {list(final_state.keys())}")
            else:
                 agent_log.critical(f"Agent returned non-dict state type: {type(final_state)}", extra={"final_state_type": str(type(final_state))})
                 raise ValueError("Agent did not return a valid state dictionary.")

        # --- Exception Handling (GraphRecursionError, other Exceptions) ---
        except GraphRecursionError as recursion_error:
             duration = time.time() - start_time
             agent_log.error(f"GraphRecursionError caught", exc_info=True, extra={"duration_seconds": round(duration, 2)})
             try:
                 # Use aget for async checkpointer
                 checkpoint_tuple = await checkpointer_instance.aget_tuple(config={"configurable": {"thread_id": session_id}})
                 if checkpoint_tuple:
                     final_state = checkpoint_tuple.checkpoint.get("channel_values", {}) # Get state dict
                     agent_log.info("Retrieved state from checkpointer after GraphRecursionError.")
                 else:
                     final_state = None
                     agent_log.warning("Could not retrieve valid state tuple from checkpointer after GraphRecursionError.")
             except Exception as get_err:
                  agent_log.error(f"Error trying to get state after GraphRecursionError", exc_info=True)
                  final_state = None
             # Provide a generic response, state might be inconsistent
             return "I seem to be stuck in a loop. Could you please rephrase your request or try something slightly different?"

        except Exception as invoke_err:
            duration = time.time() - start_time
            agent_log.error(f"Unhandled error during agent invocation", exc_info=True, extra={"duration_seconds": round(duration, 2), "error_type": type(invoke_err).__name__})
            try:
                 checkpoint_tuple = await checkpointer_instance.aget_tuple(config={"configurable": {"thread_id": session_id}})
                 if checkpoint_tuple:
                    final_state = checkpoint_tuple.checkpoint.get("channel_values", {}) # Get state dict
                    agent_log.info("Retrieved state from checkpointer after unhandled invocation error.")
                 else:
                     final_state = None
                     agent_log.warning("Could not retrieve state tuple from checkpointer after unhandled invocation error.")
            except Exception as get_err:
                agent_log.error(f"Error trying to get state after unhandled invocation error", exc_info=True)

            return f"Sorry, I encountered an internal error ({type(invoke_err).__name__}). Please try again later."


        # --- Extract the response (remains mostly the same, accesses final_state dict) ---
        all_messages = final_state.get("messages", []) if final_state else []
        assistant_response = "I'm sorry, I couldn't generate a response for that."

        if all_messages and isinstance(all_messages[-1], AIMessage):
             assistant_response = all_messages[-1].content
             agent_log.debug("Extracted AIMessage content as response.")
             # Extract content within <user> tags if present (assuming re is imported)
             user_match = re.search(r'<user>(.*?)</user>', assistant_response, re.DOTALL)
             if user_match:
                 assistant_response = user_match.group(1).strip()
                 agent_log.debug("Extracted content from <user> tags.")

        elif final_state:
             agent_log.warning(f"No AIMessage found at the end of the state messages. Last msg type: {type(all_messages[-1]).__name__ if all_messages else 'None'}", extra={"last_message": str(all_messages[-1]) if all_messages else None})
             # Check if agent requires input (adjust key based on your actual state structure if needed)
             # agent_needs_input = final_state.get("agent_state", {}).get("needs_human_input") # Example key
             # if agent_needs_input:
             #     assistant_response = "I need more information to proceed. Could you please provide details?"
             #     agent_log.info("Setting response asking for more info as agent needs human input.")
        else:
            agent_log.error("Final state was None, cannot extract response.")


        agent_log.info(f"Final assistant response prepared.", extra={"response_length": len(assistant_response)})
        duration = time.time() - start_time
        agent_log.info("generate_response finished.", extra={"total_duration_seconds": round(duration, 2)})
        return assistant_response

    except Exception as e:
        duration = time.time() - start_time
        agent_log.critical(f"Critical error in generate_response", exc_info=True, extra={"total_duration_seconds": round(duration, 2), "error_type": type(e).__name__})
        raise HTTPException(status_code=500, detail=f"Internal server error: {type(e).__name__}")


@app.post("/messages/", response_model=MessageResponse)
async def create_message(message_data: MessageCreate, request: Request, db: Session = Depends(get_db)):
    """
    Handles a new user message, generates a response, and stores both.
    """
    request_log = log.add_context(
        session_id=message_data.session_id,
        request_path=request.url.path,
        request_method=request.method
    )
    request_log.info(f"Received message, role: {message_data.role}")
    if message_data.role.lower() != "user":
         request_log.warning("Received message with non-user role.", extra={"role": message_data.role})
         raise HTTPException(status_code=400, detail="Only user messages can be posted.")

    # --- 1. Store User Message (remains the same) ---
    # ... (store user message in PG and Redis) ...
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
    try:
        existing_history = get_chat_history(message_data.session_id)
        new_history = existing_history + [f"{message_data.role}: {message_data.content}"]
        store_chat_history(message_data.session_id, new_history)
        request_log.info(f"User message added to Redis history.", extra={"history_length": len(new_history)}) # Use request_log
    except Exception as e:
        request_log.error(f"Redis error storing user message", exc_info=True) # Use request_log

    # --- 2. Generate Response ---
    try:
        # Pass the request object to generate_response so it can access app.state.agent_app
        assistant_response_content = await generate_response(message_data.session_id, message_data.content, db, request)
    except HTTPException as e:
        raise e
    except Exception as e:
        request_log.error(f"Unexpected error calling generate_response", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating response.")

    # --- 3. Store Assistant Message (remains the same) ---
    # ... (store assistant message in PG and Redis) ...
    assistant_message_data = MessageCreate(session_id=message_data.session_id, role="assistant", content=assistant_response_content)
    db_assistant_message = Message(**assistant_message_data.dict())
    db.add(db_assistant_message)
    try:
        db.commit()
        db.refresh(db_assistant_message)
        request_log.info(f"Assistant message stored in PostgreSQL.", extra={"message_id": db_assistant_message.id}) # Use request_log
    except Exception as e:
        db.rollback()
        request_log.error(f"Database error storing assistant message", exc_info=True) # Use request_log
    try:
        final_history = new_history + [f"assistant: {assistant_response_content}"]
        store_chat_history(message_data.session_id, final_history)
        request_log.info(f"Assistant message added to Redis history.", extra={"history_length": len(final_history)}) # Use request_log
    except Exception as e:
        request_log.error(f"Redis error storing assistant message", exc_info=True) # Use request_log


    request_log.info("Message request processed successfully.")
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
        return new_user
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