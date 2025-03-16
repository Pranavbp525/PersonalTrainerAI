from langchain.tools import tool
from hevy_api import get_workouts, create_routine, update_routine, get_workout_count, get_routines
from pydantic import BaseModel
from fastapi import HTTPException
from hevy_api import (
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
    pass
