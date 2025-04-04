from langchain.tools import tool
from agent.hevy_api import get_workouts, create_routine, update_routine, get_workout_count, get_routines, update_workout
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from fastapi import HTTPException
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from agent.agent_models import (
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

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "fitness-chatbot"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Matches embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@tool
async def tool_fetch_workouts(page: int = 1, page_size: int = 5):
    """
    Tool to fetch user's workout logs.
    """
    return await get_workouts(page=page, pageSize=page_size)

@tool
async def tool_get_workout_count():
    """
    Tool to retrieve the user's workout count.
    """
    return await get_workout_count()

@tool
async def tool_update_workout(workout_id: str, update_data: dict):
    """
    Tool to update an existing workout.

    :param workout_id: The ID of the workout to update.
    :param update_data: A dictionary representing the workout update data.
    """
    try:
        # Convert dict to Pydantic model
        workout_update = WorkoutUpdate(**update_data)
        update_request = WorkoutUpdateRequest(workout=workout_update)
        return await update_workout(workout_id, update_request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid workout data: {str(e)}")

@tool
async def tool_fetch_routines(page: int = 1, page_size: int = 5):
    """
    Tool to fetch the user's existing workout routines.
    """
    return await get_routines(page=page, pageSize=page_size)

@tool
async def tool_update_routine(routine_id: str, routine_data: dict):
    """
    Tool to update an existing routine.
    
    :param routine_id: The ID of the routine to update.
    :param routine_data: A dictionary representing the routine update data.
    """
    try:
        # Convert dict to Pydantic model
        routine_update = RoutineUpdate(**routine_data)
        update_request = RoutineUpdateRequest(routine=routine_update)
        return await update_routine(routine_id, update_request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid routine data: {str(e)}")

@tool
async def tool_create_routine(routine_data: dict):
    """
    Tool to create a new routine.
    
    :param routine_data: A dictionary representing the new routine data.
    """
    try:
        # Convert dict to Pydantic model
        routine_create = RoutineCreate(**routine_data)
        create_request = RoutineCreateRequest(routine=routine_create)
        return await create_routine(create_request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid routine data: {str(e)}")



@tool
async def retrieve_from_rag(query: str) -> str:
    """Retrieves science-based exercise information from Pinecone vector store."""
    try:
        query_embedding = embeddings.embed_query(query)
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        retrieved_docs = [match["metadata"].get("text", "No text available") for match in results["matches"]]
        return "Retrieved documents:\n" + "\n".join(retrieved_docs)
    except Exception as e:
        return f"Error retrieving information: {str(e)}"

