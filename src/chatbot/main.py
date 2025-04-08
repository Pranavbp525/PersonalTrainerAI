import logging
import os
import requests
import openai
from fastapi import FastAPI, Depends, HTTPException, Request # Added Request
from sqlalchemy.orm import Session
# Assuming models.py defines Session as DBSession to avoid conflict
from models import get_db, Message, MessageCreate, MessageResponse, User, Session as DBSession
from redis_utils import store_chat_history, get_chat_history
from config import config
from dotenv import load_dotenv
from pydantic import BaseModel
import json # Added json import
from datetime import datetime # Added datetime import
import re # Added re import

# --- Agent Imports ---
# Make sure AgentState uses typing_extensions.TypedDict if needed
from agent.agent_models import AgentState, UserModel # Import necessary models
# Import build function AND the checkpointer instance
from agent.graph import build_fitness_trainer_graph, agent_checkpointer # <<< MODIFIED IMPORT
# from agent.personal_trainer_agent import end_conversation # Not directly called here
from agent.prompts import ( # Assuming these are needed elsewhere or for setup
    summarize_routine_prompt, get_memory_consolidation_prompt, get_coordinator_prompt
    # push_all_prompts, push_adaptation_prompt, ... # Setup calls likely removed/done elsewhere
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.errors import GraphRecursionError # Import specific error
load_dotenv()

# --- Environment Variables & Logging ---
os.environ['LANGSMITH_API_KEY'] = os.environ.get('LANGSMITH_API')
os.environ['LANGSMITH_TRACING'] = os.environ.get('LANGSMITH_TRACING')
os.environ['LANGSMITH_PROJECT'] = os.environ.get('LANGSMITH_PROJECT')

# Use force=True in basicConfig for FastAPI/Uvicorn compatibility if needed
logging.basicConfig(level=config.LOGGING_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s', force=True)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING) # Quiet down noisy libs


# --- FastAPI App and OpenAI Client ---
app = FastAPI()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set!")
client = openai.OpenAI(api_key=api_key)

# --- Build Agent and Get Checkpointer ---
# Modify build_fitness_trainer_graph to return both app and checkpointer,
# or ensure agent_checkpointer is the correct instance used during compile.
# We assume agent_checkpointer is the correct global instance for this example.
log.info("Building fitness trainer graph...")
agent_app = build_fitness_trainer_graph()
# Ensure the checkpointer is accessible (assuming it's imported as agent_checkpointer)
if agent_checkpointer is None:
     raise RuntimeError("Agent checkpointer instance not available after building graph!")
log.info(f"Graph built. Checkpointer type: {type(agent_checkpointer)}")


# --- Pydantic Models ---
class UserCreate(BaseModel):
    username: str

class SessionCreate(BaseModel):
    user_id: int


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


async def generate_response(session_id: str, latest_human_message_content: str, db: Session) -> str:
    """
    Invokes the agent graph with persistence using the checkpointer.
    """
    log.info(f"Starting generate_response for session: {session_id}")
    try:
        # --- Prepare the INPUT for ainvoke ---
        # Input should only contain the new message for this turn
        invoke_input = {"messages": [HumanMessage(content=latest_human_message_content)]}
        log.info(f"Input for agent invoke: {invoke_input}")

        # --- Configure checkpointer for invocation ---
        # Use the globally accessible checkpointer instance
        config = {"configurable": {"thread_id": session_id, "checkpointer": agent_checkpointer}, "recursion_limit": 100}
        log.debug(f"Invoke config: {config}")

        # --- Invoke the agent ---
        final_state: AgentState = None
        try:
            # Use astream_events for more visibility if needed later, but ainvoke gets final state
            final_state = await agent_app.ainvoke(invoke_input, config=config)
            log.info("Agent invocation completed.")
            log.debug(f"Final agent state type: {type(final_state)}")
            if isinstance(final_state, dict):
                 log.debug(f"Final agent state keys: {list(final_state.keys())}")
            else:
                 log.warning(f"Agent returned non-dict state: {final_state}")
                 raise ValueError("Agent did not return a valid state dictionary.")

        except GraphRecursionError as recursion_error: # Renamed variable
             log.error(f"GraphRecursionError for session {session_id}: {recursion_error}", exc_info=True) # Use new name in log
             # Try to get the state before the error
             # config might be needed here if checkpointer.get requires it
             try:
                 checkpoint_data = agent_checkpointer.get(config) # Use config used for invoke
                 if checkpoint_data and hasattr(checkpoint_data, 'channel_values'):
                     final_state = checkpoint_data.channel_values # Extract state if possible
                 elif checkpoint_data and isinstance(checkpoint_data, dict):
                      final_state = checkpoint_data # Assume it's the state dict
                 else:
                      final_state = None
             except Exception as get_err:
                  log.error(f"Error trying to get state after GraphRecursionError: {get_err}")
                  final_state = None
             # Provide a specific error message to the user
             return "I seem to be stuck in a loop. Could you please rephrase your request or try something slightly different?"
        except Exception as invoke_err:
            log.error(f"Unhandled error during agent invocation for session {session_id}: {invoke_err}", exc_info=True)
            # Try to get last known state
            final_state = agent_checkpointer.get(config)
            if final_state and hasattr(final_state, 'channel_values'):
                final_state = final_state.channel_values # Extract state if possible
            return f"Sorry, I encountered an internal error ({type(invoke_err).__name__}). Please try again later."


        # --- Extract the response ---
        # The final state contains the complete message history
        all_messages = final_state.get("messages", [])
        assistant_response = "I'm sorry, I couldn't generate a response for that." # Default
        if all_messages and isinstance(all_messages[-1], AIMessage):
             # Assume the last message IS the response for the user for this turn
             assistant_response = all_messages[-1].content
             # Extract content within <user> tags if present (from coordinator formatting)
             log.debug(f"Type of 're' before search: {type(re)}") # Add this!
             user_match = re.search(r'<user>(.*?)</user>', assistant_response, re.DOTALL)
             if user_match:
                 assistant_response = user_match.group(1).strip()
        else:
             log.warning(f"No AIMessage found at the end of the state messages for session {session_id}. Last msg: {all_messages[-1] if all_messages else 'None'}")
             # Provide alternative response if last message wasn't AI (e.g., agent waiting for input)
             if final_state.get("agent_state", {}).get("needs_human_input"):
                 assistant_response = "I need more information to proceed. Could you please provide details?"
             # Add other checks if needed

        log.info(f"Final assistant response extracted: {assistant_response}")

        # --- State Saving is handled by the checkpointer automatically ---
        # REMOVED: await save_state(session_id, final_state)

        return assistant_response

    # Keep general exception handling, but log traceback
    except Exception as e:
        log.error(f"Critical error in generate_response for session {session_id}: {type(e).__name__}: {str(e)}", exc_info=True)
        # Re-raise as HTTPException to be caught by FastAPI error handling
        raise HTTPException(status_code=500, detail=f"Internal server error: {type(e).__name__}")

@app.post("/messages/", response_model=MessageResponse)
async def create_message(message_data: MessageCreate, db: Session = Depends(get_db)):
    """
    Handles a new user message, generates a response, and stores both.
    """
    log.info(f"Received message for session_id: {message_data.session_id}, role: {message_data.role}")
    if message_data.role.lower() != "user":
         raise HTTPException(status_code=400, detail="Only user messages can be posted.")

    # --- 1. Store User Message (PostgreSQL and Redis) ---
    # (Keep existing DB/Redis storage logic for user message)
    db_message = Message(**message_data.dict())
    db.add(db_message)
    try:
        db.commit()
        db.refresh(db_message)
        log.info(f"User message stored in PostgreSQL for session: {message_data.session_id}")
    except Exception as e:
        db.rollback()
        log.error(f"Database error storing user message for session {message_data.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error storing user message.")

    # Store in Redis *after* successful DB commit
    try:
        existing_history = get_chat_history(message_data.session_id)
        new_history = existing_history + [f"{message_data.role}: {message_data.content}"]
        store_chat_history(message_data.session_id, new_history)
        log.info(f"User message added to Redis history for session: {message_data.session_id}")
    except Exception as e:
        log.error(f"Redis error storing user message for session {message_data.session_id}: {e}", exc_info=True)
        # Decide if this is critical - maybe just log and continue

    # --- 2. Generate Response ---
    try:
        # Pass only the new user message content to generate_response
        assistant_response_content = await generate_response(message_data.session_id, message_data.content, db)
    except HTTPException as e:
        raise e # Re-raise FastAPI exceptions
    except Exception as e:
        log.error(f"Unexpected error during response generation call for session {message_data.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating response.")

    # --- 3. Store Assistant Message (PostgreSQL and Redis) ---
    # (Keep existing DB/Redis storage logic for assistant message)
    assistant_message_data = MessageCreate(session_id=message_data.session_id, role="assistant", content=assistant_response_content)
    db_assistant_message = Message(**assistant_message_data.dict())
    db.add(db_assistant_message)
    try:
        db.commit()
        db.refresh(db_assistant_message)
        log.info(f"Assistant message stored in PostgreSQL for session: {message_data.session_id}")
    except Exception as e:
        db.rollback()
        log.error(f"Database error storing assistant message for session {message_data.session_id}: {e}", exc_info=True)
        # Don't raise HTTP error here, we already have the response for the user
        # But log it as it indicates a data storage problem

    # Store in Redis *after* successful DB commit
    try:
        # new_history should still be in scope from user message storage
        final_history = new_history + [f"assistant: {assistant_response_content}"]
        store_chat_history(message_data.session_id, final_history)
        log.info(f"Assistant message added to Redis history for session: {message_data.session_id}")
    except Exception as e:
        log.error(f"Redis error storing assistant message for session {message_data.session_id}: {e}", exc_info=True)

    return db_assistant_message # Return the assistant's DB message object



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