from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

import os
import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from agent.agent_models import (
    SetUpdate, 
    ExerciseUpdate, 
    WorkoutUpdate,
    WorkoutUpdateRequest,
    SetRoutineUpdate,
    RoutineUpdateRequest,
    RoutineCreateRequest
)

load_dotenv()

API_KEY = os.getenv("HEVY_API_KEY")
if not API_KEY:
    raise Exception("HEVY_API_KEY environment variable not set")

HEADERS = {
    "api-key": API_KEY,
    "accept": "application/json",
    "Content-Type": "application/json"
}


# === Workout GET requests ===

async def get_workouts(page: int = 1, pageSize: int = 5):
    url = "https://api.hevyapp.com/v1/workouts"
    params = {"page": page, "pageSize": pageSize}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=HEADERS, params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching workouts")
    
    return response.json()


async def get_workout_count():
    url = "https://api.hevyapp.com/v1/workouts/count"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching workout count")
    
    return response.json()


async def get_workout(workout_id: str):
    url = f"https://api.hevyapp.com/v1/workouts/{workout_id}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching workout by ID")
    
    return response.json()


# === Routine GET ===

async def get_routines(page: int = 1, pageSize: int = 5):
    url = "https://api.hevyapp.com/v1/routines"
    params = {"page": page, "pageSize": pageSize}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=HEADERS, params=params)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching routines")
    
    return response.json()
