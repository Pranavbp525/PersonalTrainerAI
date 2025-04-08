from typing import Annotated, List, Dict, Any, Optional, Literal, Union, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.graph.message import add_messages
from datetime import datetime, timedelta
import json
import os
import uuid
from pydantic import BaseModel, Field
import operator



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
    superset_id: Optional[int] = None
    rest_seconds: Optional[int] = None
    notes: Optional[str] = None
    sets: List[SetRoutineUpdate]

class RoutineUpdate(BaseModel):
    title: str
    notes: Optional[str] = None
    exercises: List[ExerciseRoutineUpdate]

class RoutineUpdateRequest(BaseModel):
    routine: RoutineUpdate


# Create Routine models
class SetRoutineCreate(BaseModel):
    type: str = "normal"
    weight_kg: Optional[float] = 0.0 # Default to 0.0 if None
    reps: Optional[int] = 0 # Default to 0 if None
    distance_meters: Optional[float] = None
    duration_seconds: Optional[float] = None

class ExerciseRoutineCreate(BaseModel):
    exercise_template_id: str
    superset_id: Optional[int] = None
    rest_seconds: int = 60
    notes: str = "" # Default to empty string
    sets: List[SetRoutineCreate]

class RoutineCreate(BaseModel):
    title: str
    folder_id: Optional[str] = None # Hevy API uses string for folder_id
    notes: str = "" # Default to empty string
    exercises: List[ExerciseRoutineCreate]

class RoutineCreateRequest(BaseModel):
    routine: RoutineCreate

class HevyRoutineApiPayload(BaseModel):
     routine_data: RoutineCreate


class PlannerSetRoutineCreate(BaseModel):
    # MODIFIED: Removed default from Field, made Optional
    type: Optional[str] = Field(None, description="Type of set (e.g., normal, warmup, drop). Default to 'normal' if unsure.")
    weight_kg: Optional[float] = Field(None, description="Weight used in kilograms")
    reps: int = Field(..., description="Number of repetitions")
    distance_meters: Optional[float] = Field(None, description="Distance in meters for cardio/distance exercises")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds for timed exercises")
    notes: Optional[str] = Field(None, description="Specific notes for this set")

class PlannerExerciseRoutineCreate(BaseModel):
    exercise_name: str = Field(..., description="The specific name of the exercise, including equipment (e.g., 'Bench Press (Barbell)', 'Squat (Barbell)'). Be precise.")
    superset_id: Optional[int] = Field(None, description="Identifier (e.g., 0, 1, 2) to group exercises performed back-to-back in a superset. Null or absent if not part of a superset.")
    # MODIFIED: Removed default from Field, made Optional
    rest_seconds: Optional[int] = Field(None, description="Rest time in seconds after completing all sets of this exercise. Default to 60 if unsure.")
    notes: Optional[str] = Field(None, description="Overall notes or instructions for the exercise")
    sets: List[PlannerSetRoutineCreate] = Field(..., description="List of sets for this exercise")

class PlannerRoutineCreate(BaseModel):
    title: str = Field(..., description="Title/name of the workout routine (e.g., 'Push Day 1', 'Full Body Strength A')")
    notes: Optional[str] = Field(None, description="Overall notes or description for the routine")
    exercises: List[PlannerExerciseRoutineCreate] = Field(..., description="List of exercises in this routine")

# Wrapper remains the same
class PlannerOutputContainer(BaseModel):
    """Container for the workout plan generated by the AI planner."""
    routines: List[PlannerRoutineCreate] = Field(..., description="A list of individual workout routines making up the plan.")



class AnalysisFindings(BaseModel):
    """Structured findings from workout log analysis."""
    summary: str = Field(description="A brief text summary of the progress report.")
    key_observations: List[str] = Field(description="List of key positive or negative observations.")
    areas_for_potential_adjustment: List[str] = Field(description="Specific areas identified for adjustment/research.")

class IdentifiedRoutineTarget(TypedDict):
    """Represents a routine identified for adaptation."""
    routine_data: Dict[str, Any] # Full routine data from Hevy (structure matches GET /routines/{id})
    reason_for_selection: str # Why this routine was chosen

class RoutineAdaptationResult(TypedDict):
    """Stores the outcome of adapting a single routine."""
    routine_id: str
    original_title: str
    status: Literal["Success", "Failed", "Skipped (No Changes)", "Skipped (Error)"]
    message: str # e.g., Hevy response, error details, "No changes needed"
    updated_routine_data: Optional[Dict[str, Any]] # The full data after successful update

class ProgressAnalysisAdaptationStateV2(TypedDict):
    """State for subgraph V2 handling identification, analysis, and adaptation."""

    # --- Inputs (Required from parent graph) ---
    user_model: UserModel
    user_request_context: Optional[str] # Specific user request, e.g., "Make my leg day harder"

    # --- Internal State (Managed by the subgraph nodes) ---
    fetched_routines_list: Optional[List[Dict[str, Any]]]
    workout_logs: Optional[List[Dict[str, Any]]]
    identified_targets: Optional[List[IdentifiedRoutineTarget]]
    # Use List[] default factory if supported, otherwise initialize in first node
    processed_results: List[RoutineAdaptationResult] # Results for each processed target

    # Error tracking across the whole process
    process_error: Optional[str]

    # --- Outputs (Returned to parent graph) ---
    final_report_and_notification: Optional[str]
    cycle_completed_successfully: bool


class StreamlinedRoutineState(TypedDict):
    # messages: Sequence[BaseMessage]
    user_model: Dict[str, Any]
    working_memory: Dict[str, Any]  # Short-term contextual memory
    
    # Outputs from nodes
    planner_structured_output: Optional[List[PlannerRoutineCreate]] # List of routines from planner
    hevy_payloads: Optional[List[Dict[str, Any]]] # List of dicts ready for the tool (HevyRoutineApiPayload format)
    hevy_results: Optional[List[Dict[str, Any]]] # List of results from tool calls
    
    # Error tracking
    errors: List[str]


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
    hevy_results: Optional[List[Dict[str, Any]]]  # Structured fitness plan
    progress_data: Dict[str, Any]  # Progress tracking data
    
    # Agent management
    reasoning_trace: List[Dict[str, Any]]  # Traces of reasoning steps
    agent_state: Dict[str, str]  # Current state of each agent
    current_agent: str  # Currently active agent

    
    research_topic: Optional[str]       # Input: Topic for the deep research subgraph
    user_profile_str: Optional[str]     # Input: User model as JSON string for subgraph context
    final_report: Optional[str]         # Output: Report generated by the subgraph
    
    # Tool interaction
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[Dict]

    user_request_context: Optional[str]
    # Outputs (Coordinator checks these upon return from subgraph)
    final_report_and_notification: Optional[str] # Use a distinct name
    cycle_completed_successfully: Optional[bool] # Use a distinct name
    processed_results: Optional[List[RoutineAdaptationResult]]

    


class DeepFitnessResearchState(TypedDict):
    # Input/Shared from Parent Graph
    research_topic: str
    user_profile_str: str

    # Internal State for the Loop
    sub_questions: Optional[List[str]]
    current_sub_question_idx: int
    current_rag_query: Optional[str]
    rag_results: Optional[str]
    accumulated_findings: str
    reflections: Annotated[List[str], operator.add]
    iteration_count: int               # Overall loop counterx
    research_complete: bool
    queries_this_sub_question: int     # Counter for queries on current sub_question
    sub_question_complete_flag: bool   # Flag set by reflection node

    # Configuration (Set by plan_steps, potentially overridden by RunnableConfig)
    max_iterations: int
    max_queries_per_sub_question: int

    # Output/Shared back to Parent Graph
    final_report: Optional[str]
