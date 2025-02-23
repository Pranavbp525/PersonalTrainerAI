from pydantic import BaseModel, Field
from typing import List

class Exercise(BaseModel):
    name: str = Field(..., description="Name of the exercise")
    sets: int = Field(..., ge=1, description="Number of sets")
    reps: int = Field(..., ge=1, description="Number of repetitions per set")
    description: str = Field(..., description="Brief description of the exercise")

class WorkoutRoutine(BaseModel):
    goal: str = Field(..., description="Goal of the workout, e.g., 'strength', 'endurance'")
    exercises: List[Exercise] = Field(..., description="List of exercises in the routine")