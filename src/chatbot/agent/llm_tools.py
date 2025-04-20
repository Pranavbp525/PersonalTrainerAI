import time # Import time for timing
from langchain.tools import tool
# Assuming hevy_api functions are correctly defined and async
from agent.hevy_api import get_workouts, create_routine, update_routine, get_workout_count, get_routines, update_workout
from pydantic import BaseModel, ValidationError # Import ValidationError for Pydantic checks
from pinecone import Pinecone, ServerlessSpec
from fastapi import HTTPException
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from agent.agent_models import ( # Assuming these models are correctly defined
    SetUpdate,
    ExerciseUpdate,
    WorkoutUpdate,
    WorkoutUpdateRequest,
    SetRoutineUpdate,
    ExerciseRoutineUpdate,
    RoutineUpdate,
    RoutineUpdateRequest,
    SetRoutineCreate,
    ExerciseRoutineCreate,
    RoutineCreate,
    RoutineCreateRequest
)
import os
from langchain.tools import StructuredTool

# --- ELK Logging Import ---
try:
    # Assuming llm_tools.py is in the same directory level as elk_logging.py
    # Adjust the path '..' if llm_tools.py is inside a subdirectory like 'agent/'
    # from ..elk_logging import setup_elk_logging # If in subdirectory
    from ..elk_logging import setup_elk_logging # If at same level
except ImportError:
    # Fallback for different execution contexts
    print("Could not import elk_logging, using standard print for tool logs.")
    # Define a dummy logger class if elk_logging is unavailable
    class DummyLogger:
        def add_context(self, **kwargs): return self
        def info(self, msg, *args, **kwargs): print(f"INFO: {msg} | Context: {kwargs.get('extra')}")
        def warning(self, msg, *args, **kwargs): print(f"WARN: {msg} | Context: {kwargs.get('extra')}")
        def error(self, msg, *args, **kwargs): print(f"ERROR: {msg} | Context: {kwargs.get('extra')}")
        def debug(self, msg, *args, **kwargs): print(f"DEBUG: {msg} | Context: {kwargs.get('extra')}")
        def exception(self, msg, *args, **kwargs): print(f"EXCEPTION: {msg} | Context: {kwargs.get('extra')}")
    tools_log = DummyLogger()
else:
    tools_log = setup_elk_logging("fitness-chatbot.llm_tools")
# --- End ELK Logging Import ---

load_dotenv()

# --- Pinecone & Embeddings Initialization with Logging ---
try:
    tools_log.info("Initializing Pinecone...")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "not_found")
    tools_log.info(f"Pinecode api : {pinecone_api_key}")
    if not pinecone_api_key:
        tools_log.error("PINECONE_API_KEY environment variable not set!")
        raise ValueError("PINECONE_API_KEY not found.")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "fitness-chatbot"

    if index_name not in pc.list_indexes().names():
        tools_log.info(f"Pinecone index '{index_name}' not found, creating...")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        tools_log.info(f"Pinecone index '{index_name}' created successfully.")
    else:
        tools_log.info(f"Found existing Pinecone index '{index_name}'.")

    index = pc.Index(index_name)
    tools_log.info("Pinecone initialized successfully.")
except Exception as e:
    tools_log.error("Failed to initialize Pinecone", exc_info=True)
    # Depending on severity, you might raise the exception or try to continue without Pinecone
    index = None # Ensure index is None if initialization failed

try:
    tools_log.info("Initializing HuggingFace embeddings model...")
    # Ensure the model name is correct and accessible
    embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    tools_log.info("Embeddings model initialized successfully.", extra={"model_name": embeddings_model_name})
except Exception as e:
    tools_log.error("Failed to initialize embeddings model", exc_info=True)
    embeddings = None # Ensure embeddings is None if initialization failed
# --- End Initialization ---


@tool
async def tool_fetch_workouts(page: int = 1, page_size: int = 5):
    """
    Tool to fetch user's workout logs.
    """
    tool_name = "tool_fetch_workouts"
    log_ctx = {"tool_name": tool_name, "page": page, "page_size": page_size}
    tools_log.info(f"Executing tool: {tool_name}", extra=log_ctx)
    start_time = time.time()
    try:
        # --- Original Logic ---
        result = await get_workouts(page=page, pageSize=page_size)
        # --- End Original Logic ---
        duration = time.time() - start_time
        # Log success, maybe summarize result
        num_workouts = len(result.get('workouts', [])) if isinstance(result, dict) else 'N/A'
        tools_log.info(f"Tool {tool_name} execution successful.", extra={**log_ctx, "duration_seconds": round(duration, 2), "workouts_fetched": num_workouts})
        return result
    except Exception as e:
        duration = time.time() - start_time
        tools_log.error(f"Error executing tool: {tool_name}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        # Re-raise the original exception or a tool-specific one if needed
        raise e

@tool
async def tool_get_workout_count():
    """
    Tool to retrieve the user's workout count.
    """
    tool_name = "tool_get_workout_count"
    log_ctx = {"tool_name": tool_name}
    tools_log.info(f"Executing tool: {tool_name}", extra=log_ctx)
    start_time = time.time()
    try:
        # --- Original Logic ---
        result = await get_workout_count()
        # --- End Original Logic ---
        duration = time.time() - start_time
        count = result.get('count', 'N/A') if isinstance(result, dict) else 'N/A'
        tools_log.info(f"Tool {tool_name} execution successful.", extra={**log_ctx, "duration_seconds": round(duration, 2), "workout_count": count})
        return result
    except Exception as e:
        duration = time.time() - start_time
        tools_log.error(f"Error executing tool: {tool_name}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise e

@tool
async def tool_update_workout(workout_id: str, update_data: dict):
    """
    Tool to update an existing workout.

    :param workout_id: The ID of the workout to update.
    :param update_data: A dictionary representing the workout update data.
    """
    tool_name = "tool_update_workout"
    log_ctx = {"tool_name": tool_name, "workout_id": workout_id}
    tools_log.info(f"Executing tool: {tool_name}", extra=log_ctx)
    start_time = time.time()
    try:
        # --- Original Logic ---
        tools_log.debug("Validating update_data against WorkoutUpdate model.", extra=log_ctx)
        # Add Pydantic validation log
        try:
            workout_update = WorkoutUpdate(**update_data)
            tools_log.debug("Validation successful.", extra=log_ctx)
        except ValidationError as val_err:
             tools_log.error("Pydantic validation failed for workout update data.", exc_info=val_err, extra=log_ctx)
             raise HTTPException(status_code=400, detail=f"Invalid workout data format: {val_err}")

        update_request = WorkoutUpdateRequest(workout=workout_update)
        result = await update_workout(workout_id, update_request)
        # --- End Original Logic ---
        duration = time.time() - start_time
        tools_log.info(f"Tool {tool_name} execution successful.", extra={**log_ctx, "duration_seconds": round(duration, 2)})
        return result
    except HTTPException as http_exc: # Re-raise HTTP exceptions directly
        duration = time.time() - start_time
        tools_log.error(f"HTTP Error executing tool: {tool_name}", exc_info=http_exc, extra={**log_ctx, "duration_seconds": round(duration, 2), "status_code": http_exc.status_code})
        raise http_exc
    except Exception as e:
        duration = time.time() - start_time
        # Log unexpected errors
        tools_log.error(f"Unexpected error executing tool: {tool_name}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        # Convert to HTTPException for consistent error handling if desired, or re-raise
        raise HTTPException(status_code=500, detail=f"Internal error updating workout: {str(e)}")


@tool
async def tool_fetch_routines(page: int = 1, page_size: int = 5):
    """
    Tool to fetch the user's existing workout routines.
    """
    tool_name = "tool_fetch_routines"
    log_ctx = {"tool_name": tool_name, "page": page, "page_size": page_size}
    tools_log.info(f"Executing tool: {tool_name}", extra=log_ctx)
    start_time = time.time()
    try:
        # --- Original Logic ---
        result = await get_routines(page=page, pageSize=page_size)
        # --- End Original Logic ---
        duration = time.time() - start_time
        num_routines = len(result.get('routines', [])) if isinstance(result, dict) else 'N/A'
        tools_log.info(f"Tool {tool_name} execution successful.", extra={**log_ctx, "duration_seconds": round(duration, 2), "routines_fetched": num_routines})
        return result
    except Exception as e:
        duration = time.time() - start_time
        tools_log.error(f"Error executing tool: {tool_name}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise e

@tool
async def tool_update_routine(routine_id: str, routine_data: dict):
    """
    Tool to update an existing routine.

    :param routine_id: The ID of the routine to update.
    :param routine_data: A dictionary representing the routine update data. Should conform to RoutineUpdate schema.
    """
    tool_name = "tool_update_routine"
    log_ctx = {"tool_name": tool_name, "routine_id": routine_id}
    tools_log.info(f"Executing tool: {tool_name}", extra=log_ctx)
    start_time = time.time()
    try:
        # --- Original Logic ---
        tools_log.debug("Validating routine_data against RoutineUpdate model.", extra=log_ctx)
        # Add Pydantic validation log
        try:
            routine_update = RoutineUpdate(**routine_data)
            tools_log.debug("Validation successful.", extra=log_ctx)
        except ValidationError as val_err:
            tools_log.error("Pydantic validation failed for routine update data.", exc_info=val_err, extra=log_ctx)
            raise HTTPException(status_code=400, detail=f"Invalid routine data format: {val_err}")

        update_request = RoutineUpdateRequest(routine=routine_update)
        result = await update_routine(routine_id, update_request)
        # --- End Original Logic ---
        duration = time.time() - start_time
        tools_log.info(f"Tool {tool_name} execution successful.", extra={**log_ctx, "duration_seconds": round(duration, 2)})
        return result
    except HTTPException as http_exc: # Re-raise HTTP exceptions directly
        duration = time.time() - start_time
        tools_log.error(f"HTTP Error executing tool: {tool_name}", exc_info=http_exc, extra={**log_ctx, "duration_seconds": round(duration, 2), "status_code": http_exc.status_code})
        raise http_exc
    except Exception as e:
        duration = time.time() - start_time
        # Log unexpected errors
        tools_log.error(f"Unexpected error executing tool: {tool_name}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=500, detail=f"Internal error updating routine: {str(e)}")


@tool
async def tool_create_routine(routine_data: dict):
    """
    Tool to create a new routine.

    :param routine_data: A dictionary representing the new routine data. Should conform to RoutineCreate schema.
    """
    tool_name = "tool_create_routine"
    log_ctx = {"tool_name": tool_name} # Avoid logging full routine_data by default unless debugging
    tools_log.info(f"Executing tool: {tool_name}", extra=log_ctx)
    start_time = time.time()
    try:
        # --- Original Logic ---
        tools_log.debug("Validating routine_data against RoutineCreate model.", extra=log_ctx)
        # Add Pydantic validation log
        try:
            routine_create = RoutineCreate(**routine_data)
            log_ctx["routine_title"] = routine_create.title # Add title to context after validation
            tools_log.debug("Validation successful.", extra=log_ctx)
        except ValidationError as val_err:
            tools_log.error("Pydantic validation failed for routine create data.", exc_info=val_err, extra=log_ctx)
            raise HTTPException(status_code=400, detail=f"Invalid routine data format: {val_err}")

        create_request = RoutineCreateRequest(routine=routine_create)
        result = await create_routine(create_request)
        # --- End Original Logic ---
        duration = time.time() - start_time
        # Extract created ID if possible
        created_id = None
        if isinstance(result, dict):
             routine_info = result.get("routine")
             if isinstance(routine_info, list) and routine_info:
                 created_id = routine_info[0].get("id")
             elif isinstance(routine_info, dict):
                 created_id = routine_info.get("id")
        tools_log.info(f"Tool {tool_name} execution successful.", extra={**log_ctx, "duration_seconds": round(duration, 2), "created_routine_id": created_id})
        return result
    except HTTPException as http_exc: # Re-raise HTTP exceptions directly
        duration = time.time() - start_time
        tools_log.error(f"HTTP Error executing tool: {tool_name}", exc_info=http_exc, extra={**log_ctx, "duration_seconds": round(duration, 2), "status_code": http_exc.status_code})
        raise http_exc
    except Exception as e:
        duration = time.time() - start_time
        # Log unexpected errors
        tools_log.error(f"Unexpected error executing tool: {tool_name}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        raise HTTPException(status_code=500, detail=f"Internal error creating routine: {str(e)}")


@tool
async def retrieve_from_rag(query: str) -> str:
    """Retrieves science-based exercise information from Pinecone vector store."""
    tool_name = "retrieve_from_rag"
    log_ctx = {"tool_name": tool_name, "query": query}
    tools_log.info(f"Executing tool: {tool_name}", extra=log_ctx)
    start_time = time.time()

    # Check if embeddings and index were initialized
    if embeddings is None:
        tools_log.error("Embeddings model not initialized. Cannot perform RAG retrieval.", extra=log_ctx)
        return "Error: Embeddings model not available."
    if index is None:
        tools_log.error("Pinecone index not initialized. Cannot perform RAG retrieval.", extra=log_ctx)
        return "Error: Vector database index not available."

    try:
        # --- Original Logic ---
        tools_log.debug("Generating query embedding.", extra=log_ctx)
        embed_start = time.time()
        query_embedding = embeddings.embed_query(query)
        embed_duration = time.time() - embed_start
        tools_log.debug("Query embedding generated.", extra={**log_ctx, "embedding_duration_seconds": round(embed_duration, 2)})

        tools_log.debug("Querying Pinecone index.", extra=log_ctx)
        query_start = time.time()
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        query_duration = time.time() - query_start
        num_matches = len(results.get("matches", []))
        tools_log.debug("Pinecone query complete.", extra={**log_ctx, "query_duration_seconds": round(query_duration, 2), "matches_found": num_matches})

        retrieved_docs = [match["metadata"].get("text", "No text available") for match in results["matches"]]
        result_string = "Retrieved documents:\n" + "\n".join(retrieved_docs)
        # --- End Original Logic ---
        duration = time.time() - start_time
        tools_log.info(f"Tool {tool_name} execution successful.", extra={**log_ctx, "duration_seconds": round(duration, 2), "matches_retrieved": num_matches})
        return result_string
    except Exception as e:
        duration = time.time() - start_time
        tools_log.error(f"Error executing tool: {tool_name}", exc_info=True, extra={**log_ctx, "duration_seconds": round(duration, 2)})
        # Return error message as string, consistent with original code
        return f"Error retrieving information: {str(e)}"