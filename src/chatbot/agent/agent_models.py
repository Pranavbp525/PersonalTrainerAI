from typing import Annotated, List, Dict, Any, Optional, Literal, Union, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.graph.message import add_messages
from datetime import datetime, timedelta
import json
import os
import uuid
from pydantic import BaseModel, Field



class UserProfile(BaseModel):
    name: Optional[str] = Field(None, description="User's name")
    age: Optional[int] = Field(None, description="User's age")
    gender: Optional[str] = Field(None, description="User's gender")
    goals: Optional[List[str]] = Field(None, description="User's fitness goals")
    preferences: Optional[List[str]] = Field(None, description="User's preferences")
    constraints: Optional[List[str]] = Field(None, description="User's constraints")
    fitness_level: Optional[str] = Field(None, description="User's estimated fitness level")
    motivation_factors: Optional[List[str]] = Field(None, description="User's motivation factors")
    learning_style: Optional[str] = Field(None, description="User's preferred learning style")
    confidence_scores: Optional[Dict[str, float]] = Field(None, description="Confidence scores for each aspect")
    available_equipment: Optional[List[str]] = Field(None, description="Equipment available to the user")
    training_environment: Optional[str] = Field(None, description="Where the user prefers to train")
    schedule: Optional[Dict[str, str]] = Field(None, description="User's available schedule")
    measurements: Optional[Dict[str, float]] = Field(None, description="Body measurements in cm/inches")
    height: Optional[float] = Field(None, description="User's height in cm/inches")
    weight: Optional[float] = Field(None, description="User's weight in kg/lbs")
    workout_history: Optional[List[str]] = Field(None, description="User's past workout routines")



class UserModel(BaseModel):
    name: Optional[str] = Field(None, description="User's name")
    age: Optional[int] = Field(None, description="User's age")
    gender: Optional[str] = Field(None, description="User's gender")
    goals: Optional[List[str]] = Field(None, description="User's fitness goals")
    preferences: Optional[List[str]] = Field(None, description="User's preferences")
    constraints: Optional[List[str]] = Field(None, description="User's constraints")
    fitness_level: Optional[str] = Field(None, description="User's estimated fitness level")
    motivation_factors: Optional[List[str]] = Field(None, description="User's motivation factors")
    learning_style: Optional[str] = Field(None, description="User's preferred learning style")
    confidence_scores: Optional[Dict[str, float]] = Field(None, description="Confidence scores for each aspect")
    available_equipment: Optional[List[str]] = Field(None, description="Equipment available to the user")
    training_environment: Optional[str] = Field(None, description="Where the user prefers to train")
    schedule: Optional[Dict[str, str]] = Field(None, description="User's available schedule")
    measurements: Optional[Dict[str, float]] = Field(None, description="Body measurements in cm/inches")
    height: Optional[float] = Field(None, description="User's height in cm/inches")
    weight: Optional[float] = Field(None, description="User's weight in kg/lbs")
    workout_history: Optional[List[str]] = Field(None, description="User's past workout routines")
    last_updated: Optional[datetime] = Field(None, description="Current Date and Time")
    model_version: Optional[int] = Field(None, description="The version of the User Model")
    missing_fields: Optional[List[str]] = Field(None, description="Missing fields of User Model")



# Define comprehensive state model
class AgentState(TypedDict):
    # Core state
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    
    # Memory components
    memory: Dict[str, Any]  # Long-term memory storage
    working_memory: Dict[str, Any]  # Short-term contextual memory
    
    # User and fitness data
    user_model: UserModel = Field(default_factory=UserModel)  # Comprehensive user profile
    fitness_plan: Dict[str, Any]  # Structured fitness plan
    progress_data: Dict[str, Any]  # Progress tracking data
    
    # Agent management
    reasoning_trace: List[Dict[str, Any]]  # Traces of reasoning steps
    agent_state: Dict[str, str]  # Current state of each agent
    current_agent: str  # Currently active agent
    
    # Tool interaction
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[Dict]


# Scientific & Research Models
class TrainingPrinciples(BaseModel):
    """Scientific training principles extracted from text."""
    principles: List[str] = Field(description="List of scientific fitness principles mentioned in the text.")

class TrainingApproach(BaseModel):
    """A training approach with name and description."""
    name: str = Field(description="The name of the training approach.")
    description: str = Field(description="A description of what the training approach entails.")

class TrainingApproaches(BaseModel):
    """Collection of training approaches extracted from text."""
    approaches: List[TrainingApproach] = Field(description="List of training approaches mentioned in the text.")

class Citation(BaseModel):
    """A citation from a scientific or expert source."""
    source: str = Field(description="The name of the source (author, publication, influencer).")
    content: str = Field(description="The cited content or key point.")

class Citations(BaseModel):
    """Collection of citations extracted from text."""
    citations: List[Citation] = Field(description="List of citations mentioned in the text.")

# Progress Analysis Models
class AdherenceRate(BaseModel):
    """Workout adherence rate extracted from analysis text."""
    rate: float = Field(description="The adherence rate as a decimal between 0 and 1.")

class ProgressMetrics(BaseModel):
    """Collection of progress metrics extracted from analysis."""
    metrics: Dict[str, float] = Field(description="Dictionary of metric names and their values.")

class IssuesList(BaseModel):
    """Collection of issues identified in a progress analysis."""
    issues: List[str] = Field(description="List of issues identified in the analysis.")

class Adjustment(BaseModel):
    """A suggested workout adjustment."""
    target: str = Field(description="The target of the adjustment (exercise, volume, etc).")
    change: str = Field(description="The specific change to make.")

class AdjustmentsList(BaseModel):
    """Collection of suggested adjustments from analysis."""
    adjustments: List[Adjustment] = Field(description="List of suggested adjustments.")

# Routine Models
class BasicRoutine(BaseModel):
    """A simplified routine structure."""
    name: str = Field(description="Name of the routine.")
    description: str = Field(description="Description of the routine.")
    workouts: List[str] = Field(description="List of workout names or descriptions.")

class SetExtract(BaseModel):
    """A set within an exercise for extraction."""
    type: str = Field(description="Type of set (normal, warmup, drop, etc).")
    weight: float = Field(description="Weight used in kg.")
    reps: int = Field(description="Number of repetitions.")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds for timed exercises.")
    distance_meters: Optional[float] = Field(None, description="Distance in meters for distance-based exercises.")

class ExerciseExtract(BaseModel):
    """An exercise within a workout routine for extraction."""
    exercise_name: str = Field(description="Name of the exercise.")
    exercise_id: str = Field(description="ID of the exercise in the database.")
    exercise_type: str = Field(description="Type of exercise (strength, cardio, etc).")
    sets: List[SetExtract] = Field(description="List of sets for this exercise.")
    notes: str = Field(description="Notes or instructions for this exercise.")

class RoutineExtract(BaseModel):
    """A complete workout routine structure for extraction."""
    title: str = Field(description="Title of the routine.")
    notes: str = Field(description="Overall notes for the routine.")
    exercises: List[ExerciseExtract] = Field(description="List of exercises in this routine.")


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
