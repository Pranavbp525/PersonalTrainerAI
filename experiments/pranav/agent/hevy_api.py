from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime

import os
from fastapi import FastAPI, Query, Depends, HTTPException, status, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
from dotenv import load_dotenv
import httpx
from fastapi.encoders import jsonable_encoder


# Data models 
# UPDATE Workout models
class SetUpdate(BaseModel):
    type: str
    weight_kg: float
    reps: int
    distance_meters: Optional[float] = None
    duration_seconds: Optional[float] = None
    rpe: Optional[float] = None

class ExerciseUpdate(BaseModel):
    exercise_template_id: str
    superset_id: Optional[str] = None
    notes: str
    sets: List[SetUpdate]

class WorkoutUpdate(BaseModel):
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    is_private: bool
    exercises: List[ExerciseUpdate]

class WorkoutUpdateRequest(BaseModel):
    workout: WorkoutUpdate

# UPDATE Routine models
class SetRoutineUpdate(BaseModel):
    type: str
    weight_kg: float
    reps: int
    distance_meters: Optional[float] = None
    duration_seconds: Optional[float] = None

class ExerciseRoutineUpdate(BaseModel):
    exercise_template_id: str
    superset_id: Optional[str] = None
    rest_seconds: int
    notes: str
    sets: List[SetRoutineUpdate]

class RoutineUpdate(BaseModel):
    title: str
    notes: str
    exercises: List[ExerciseRoutineUpdate]

class RoutineUpdateRequest(BaseModel):
    routine: RoutineUpdate


# Create Routine models
class SetRoutineCreate(BaseModel):
    type: str
    weight_kg: float
    reps: int
    distance_meters: Optional[float] = None
    duration_seconds: Optional[float] = None

class ExerciseRoutineCreate(BaseModel):
    exercise_template_id: str
    superset_id: Optional[str] = None
    rest_seconds: int
    notes: str
    sets: List[SetRoutineCreate]

class RoutineCreate(BaseModel):
    title: str
    folder_id: Optional[int] = None
    notes: str
    exercises: List[ExerciseRoutineCreate]

class RoutineCreateRequest(BaseModel):
    routine: RoutineCreate


load_dotenv()

# Retrieve the API key from the environment
API_KEY = os.getenv("HEVY_API_KEY")
if not API_KEY:
    raise Exception("HEVY_API_KEY environment variable not set")

# Define the API key header dependency
API_KEY_NAME = "api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)) -> APIKey:
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials"
    )

app = FastAPI(title="Workout API", version="1.0")


## GET REQUESTS ## 

# GET WORKOUTS
@app.get("/v1/workouts")
async def get_workouts(
    api_key: APIKey = Depends(get_api_key),
    page: int = Query(1, ge=1, description="Page number (Must be 1 or greater)"),
    pageSize: int = Query(5, le=10, description="Number of items on the requested page (Max 10)")
):
    external_api_url = "https://api.hevyapp.com/v1/workouts"
    params = {"page": page, "pageSize": pageSize}
    headers = {
        "api-key": API_KEY,
        "accept": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(external_api_url, headers=headers, params=params)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching data from external API")
    
    return response.json()

# GET WORKOUT COUNT
@app.get("/v1/workouts/count")
async def get_workout_count(api_key: APIKey = Depends(get_api_key)):
    external_api_url = "https://api.hevyapp.com/v1/workouts/count"
    headers = {
        "api-key": API_KEY,
        "accept": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(external_api_url, headers=headers)
        
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, 
            detail="Error fetching data from external API"
        )
    
    return response.json()

# GET WORKOUT BY ID
@app.get("/v1/workouts/{workoutId}")
async def get_workout(
    workoutId: str, 
    api_key: APIKey = Depends(get_api_key)
):
    external_api_url = f"https://api.hevyapp.com/v1/workouts/{workoutId}"
    headers = {"api-key": API_KEY, "accept": "application/json"}

    async with httpx.AsyncClient() as client:
        response = await client.get(external_api_url, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, 
            detail="Error fetching data from external API"
        )
    
    return response.json()

# GET ROUTINES
@app.get("/v1/routines")
async def get_routines(
    api_key: APIKey = Depends(get_api_key),
    page: int = Query(1, ge=1, description="Page number (Must be 1 or greater)"),
    pageSize: int = Query(5, le=10, description="Number of items on the requested page (Max 10)")
):
    external_api_url = "https://api.hevyapp.com/v1/routines"
    params = {"page": page, "pageSize": pageSize}
    headers = {
        "api-key": API_KEY,
        "accept": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(external_api_url, headers=headers, params=params)
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, 
            detail="Error fetching data from external API"
        )
    
    return response.json()



## PUT REQUESTS ## 
# Update workout
@app.put("/v1/workouts/{workoutId}")
async def update_workout(
    workoutId: str,
    update_data: WorkoutUpdateRequest,
    api_key: APIKey = Depends(get_api_key)
):
    external_api_url = f"https://api.hevyapp.com/v1/workouts/{workoutId}"
    headers = {
        "api-key": API_KEY,
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = jsonable_encoder(update_data)
    
    async with httpx.AsyncClient() as client:
        response = await client.put(external_api_url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail="Error updating workout in external API"
        )
    
    return response.json()


# Update routine
@app.put("/v1/routines/{routineId}")
async def update_routine(
    routineId: str,
    update_data: RoutineUpdateRequest,
    api_key: APIKey = Depends(get_api_key)
):
    external_api_url = f"https://api.hevyapp.com/v1/routines/{routineId}"
    headers = {
        "api-key": API_KEY,
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = jsonable_encoder(update_data)
    
    async with httpx.AsyncClient() as client:
        response = await client.put(external_api_url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail="Error updating routine in external API"
        )
    
    return response.json()


## POST REQUESTS ##
# Create routine
@app.post("/v1/routines")
async def create_routine(
    routine_data: RoutineCreateRequest,
    api_key: APIKey = Depends(get_api_key)
):
    external_api_url = "https://api.hevyapp.com/v1/routines"
    headers = {
        "api-key": API_KEY,
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = jsonable_encoder(routine_data)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(external_api_url, headers=headers, json=payload)
    
    if response.status_code not in (200, 201):
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error creating routine in external API: {response.text}"
        )
    
    return response.json()

