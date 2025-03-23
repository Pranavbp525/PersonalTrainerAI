"""
Enhanced Multi-Agent Architecture for Personal Fitness Trainer AI

This implementation combines the strengths of the multi-agent system with
specialized components for fitness training, RAG integration, and Hevy API usage.
"""

from typing import Annotated, List, Dict, Any, Optional, Literal, Union, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from datetime import datetime, timedelta
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
import json
import os
import logging
import re
from pinecone import Pinecone, ServerlessSpec

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import uuid
from pydantic import BaseModel, Field

# Import Hevy API models directly
from hevy_api import (
    SetRoutineCreate, ExerciseRoutineCreate, RoutineCreate, RoutineCreateRequest,
    SetRoutineUpdate, ExerciseRoutineUpdate, RoutineUpdate, RoutineUpdateRequest
)

# Import LLM tools directly
from llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))

# Configure LLM with tools
llm_with_tools = llm.bind_tools([
    retrieve_from_rag,
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine
])

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


from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain.schema.runnable import RunnableLambda

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




def extract_principles(text: str) -> List[str]:
    """Extract scientific principles from text."""
    extraction_chain = llm.with_structured_output(TrainingPrinciples)
    
    result = extraction_chain.invoke(
        "Extract the scientific fitness principles from the following text. Return a list of specific principles mentioned:\n\n" + text
    )
    
    return result.principles

def extract_approaches(text: str) -> List[Dict]:
    """Extract training approaches from text."""
    extraction_chain = llm.with_structured_output(TrainingApproaches)
    
    result = extraction_chain.invoke(
        "Extract the training approaches with their names and descriptions from the following text:\n\n" + text
    )
    
    return [{"name": approach.name, "description": approach.description} 
            for approach in result.approaches]

def extract_citations(text: str) -> List[Dict]:
    """Extract citations from text."""
    extraction_chain = llm.with_structured_output(Citations)
    
    result = extraction_chain.invoke(
        "Extract the citations and their sources from the following text. For each citation, identify the source (author, influencer, publication) and the main content or claim:\n\n" + text
    )
    
    return [{"source": citation.source, "content": citation.content} 
            for citation in result.citations]

def extract_routine_data(text: str) -> Dict:
    """Extract structured routine data for Hevy API."""
    extraction_chain = llm.with_structured_output(BasicRoutine)
    
    result = extraction_chain.invoke(
        "Extract the basic workout routine information from the following text. Identify the name, description, and list of workout days or sessions:\n\n" + text
    )
    
    return {
        "name": result.name,
        "description": result.description,
        "workouts": result.workouts
    }

def extract_adherence_rate(text: str) -> float:
    """Extract adherence rate from analysis text."""
    extraction_chain = llm.with_structured_output(AdherenceRate)
    
    result = extraction_chain.invoke(
        "Extract the workout adherence rate as a decimal between 0 and 1 from the following analysis text. This should represent the percentage of planned workouts that were completed:\n\n" + text
    )
    
    return result.rate

def extract_progress_metrics(text: str) -> Dict:
    """Extract progress metrics from analysis text."""
    extraction_chain = llm.with_structured_output(ProgressMetrics)
    
    result = extraction_chain.invoke(
        "Extract the progress metrics with their values from the following analysis text. Identify metrics like strength gains, endurance improvements, weight changes, etc. and their numeric values:\n\n" + text
    )
    
    return result.metrics

def extract_issues(text: str) -> List[str]:
    """Extract identified issues from analysis text."""
    extraction_chain = llm.with_structured_output(IssuesList)
    
    result = extraction_chain.invoke(
        "Extract the identified issues or problems from the following workout analysis text. List each distinct issue that needs attention:\n\n" + text
    )
    
    return result.issues

def extract_adjustments(text: str) -> List[Dict]:
    """Extract suggested adjustments from analysis text."""
    extraction_chain = llm.with_structured_output(AdjustmentsList)
    
    result = extraction_chain.invoke(
        "Extract the suggested workout adjustments from the following analysis text. For each adjustment, identify the target (exercise, schedule, etc.) and the specific change recommended:\n\n" + text
    )
    
    return [{"target": adj.target, "change": adj.change} 
            for adj in result.adjustments]


def extract_routine_structure(text: str) -> Dict:
    """Extract detailed routine structure from text for Hevy API."""
    extraction_chain = llm.with_structured_output(RoutineCreate)
    
    result = extraction_chain.invoke("""
    Extract a detailed workout routine structure from the following text, suitable for the Hevy API:
    
    """ + text + """
    
    Create a structured workout routine with:
    - A title for the routine
    - Overall notes or description
    - A list of exercises, each with:
      - Exercise name
      - Exercise ID (use a placeholder if not specified)
      - Exercise type (strength, cardio, etc)
      - Sets with reps, weight, and type
      - Any specific notes for the exercise
    """)

    logger.info(f'EXTRACTED ROUTINE : {result}')
    
    return result.model_dump()

def extract_routine_updates(text: str) -> Dict:
    """Extract routine updates from text for Hevy API."""
    extraction_chain = llm.with_structured_output(RoutineExtract)
    
    result = extraction_chain.invoke("""
    Extract updates to a workout routine from the following text, suitable for the Hevy API:
    
    """ + text + """
    
    Create a structured representation of the updated workout routine with:
    - The updated title for the routine
    - Updated overall notes
    - The list of exercises with their updated details, each with:
      - Exercise name
      - Exercise ID (use a placeholder if not specified)
      - Exercise type (strength, cardio, etc)
      - Updated sets with reps, weight, and type
      - Any updated notes for the exercise
    """)
    
    return {
        "title": result.title,
        "notes": result.notes,
        "exercises": [
            {
                "exercise_name": ex.exercise_name,
                "exercise_id": ex.exercise_id,
                "exercise_type": ex.exercise_type,
                "sets": [
                    {
                        "type": s.type, 
                        "weight": s.weight,
                        "reps": s.reps,
                        "duration_seconds": s.duration_seconds,
                        "distance_meters": s.distance_meters
                    } for s in ex.sets
                ],
                "notes": ex.notes
            } for ex in result.exercises
        ]
    }


# User modeler for comprehensive user understanding
async def user_modeler(state: AgentState) -> AgentState:
    """Builds and maintains a comprehensive model of the user."""
    logger.info(f"User Modeler - Input State: {state}") #Log input state

    parser = PydanticOutputParser(pydantic_object=UserProfile)
    format_instructions = parser.get_format_instructions()

    # Define the prompt template with format instructions
    prompt_template = PromptTemplate(
        input_variables=["user_model", "recent_exchanges"],
        template="""
        You are a user modeling specialist for a fitness training system. Analyze all available information about the user to build a comprehensive model:
        1. Extract explicit information (stated goals, preferences, constraints)
        2. Infer implicit information (fitness level, motivation factors, learning style)
        3. Identify gaps in our understanding that need to be addressed
        4. Update confidence levels for different aspects of the model
        
        Current user model: {user_model}
        Recent exchanges: {recent_exchanges}
        
        Return an updated user model with confidence scores for each attribute in the following JSON format:
        {format_instructions}
        """,
        partial_variables={"format_instructions": format_instructions}
    )
    
    current_user_model = state.get("user_model", {})
    recent_exchanges = state.get("working_memory", {}).get("recent_exchanges", [])
    
    # Format the prompt
    formatted_prompt = prompt_template.format(
        user_model=json.dumps(current_user_model),
        recent_exchanges=json.dumps(recent_exchanges)
    )
    
    # Use LangChain to get structured output
    messages = [SystemMessage(content=formatted_prompt)]
    response = await llm.ainvoke(messages)
    
    # Parse the response content into the structured model
    parsed_model = parser.parse(response.content)
    
    # Update user model
    user_model = state.get("user_model", {})
    user_model["last_updated"] = datetime.now().isoformat()
    user_model["model_version"] = user_model.get("model_version", 0) + 1
    
    # Update user model fields from parsed response, preserving existing values if new ones are None
    if parsed_model.name is not None:
        user_model["name"] = parsed_model.name
    if parsed_model.age is not None:
        user_model["age"] = parsed_model.age
    if parsed_model.gender is not None:
        user_model["gender"] = parsed_model.gender
    if parsed_model.goals is not None:
        user_model["goals"] = parsed_model.goals
    if parsed_model.preferences is not None:
        user_model["preferences"] = parsed_model.preferences
    if parsed_model.constraints is not None:
        user_model["constraints"] = parsed_model.constraints
    if parsed_model.fitness_level is not None:
        user_model["fitness_level"] = parsed_model.fitness_level
    if parsed_model.motivation_factors is not None:
        user_model["motivation_factors"] = parsed_model.motivation_factors
    if parsed_model.learning_style is not None:
        user_model["learning_style"] = parsed_model.learning_style
    if parsed_model.confidence_scores is not None:
        user_model["confidence_scores"] = parsed_model.confidence_scores
    if parsed_model.available_equipment is not None:
        user_model["available_equipment"] = parsed_model.available_equipment
    if parsed_model.training_environment is not None:
        user_model["training_environment"] = parsed_model.training_environment
    if parsed_model.schedule is not None:
        user_model["schedule"] = parsed_model.schedule
    if parsed_model.measurements is not None:
        user_model["measurements"] = parsed_model.measurements
    if parsed_model.height is not None:
        user_model["height"] = parsed_model.height
    if parsed_model.weight is not None:
        user_model["weight"] = parsed_model.weight
    if parsed_model.workout_history is not None:
        user_model["workout_history"] = parsed_model.workout_history
    
    # Check if assessment is complete
    required_fields = ["goals", "fitness_level", "available_equipment", 
                        "training_environment", "schedule", "constraints"]
    missing_fields = [field for field in required_fields 
                        if field not in user_model or not user_model.get(field)]
    
    user_model["missing_fields"] = missing_fields
    user_model["assessment_complete"] = len(missing_fields) == 0
    
    logger.info(f"User model updated successfully via PydanticOutputParser")

    
    
    updated_state = {
        **state,
        "user_model": user_model,
        "current_agent": "coordinator"  # Return to coordinator after updating
    }
    
    logger.info(f"User Modeler - Output State: {updated_state}")  # Log output state
    return updated_state

async def coordinator(state: AgentState) -> AgentState:
    """Central coordinator that manages the overall interaction flow, assessment, and memory."""
    logger.info(f"Coordinator Agent - Input State: {state}")
    
    # =================== MEMORY MANAGEMENT SECTION ===================
    # Process recent messages for context
    recent_exchanges = []
    for msg in state["messages"][-10:]:  # Last 10 messages
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            content = msg.content
            # Extract user-facing content from AI messages if needed
            if isinstance(msg, AIMessage):
                user_match = re.search(r'<user>(.*?)</user>', content, re.DOTALL)
                if user_match:
                    content = user_match.group(1).strip()
            
            recent_exchanges.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": content
            })
    
    # Update working memory with recent context
    working_memory = state.get("working_memory", {})
    working_memory["recent_exchanges"] = recent_exchanges
    working_memory["last_updated"] = datetime.now().isoformat()
    
    # Track agent interactions in memory
    memory = state.get("memory", {})
    memory_key = f"interaction_{datetime.now().isoformat()}"
    memory[memory_key] = {
        "agent_states": state.get("agent_state", {}),
        "current_agent": state.get("current_agent", "coordinator"),
        "user_intent": working_memory.get("current_user_intent", "unknown")
    }
    
    # Periodic memory consolidation (every 10 interactions)
    memory_size = len(memory)
    if memory_size > 0 and memory_size % 10 == 0:
        logger.info(f"Performing memory consolidation (memory size: {memory_size})")
        try:
            memory_prompt = """You are the memory manager for a fitness training system. Review the conversation history and current agent states to:
            1. Identify key information that should be stored in long-term memory
            2. Update the user model with new insights
            3. Consolidate redundant information
            4. Prune outdated or superseded information
            5. Ensure critical context is available in working memory
            
            Current long-term memory: {{memory}}
            Current user model: {{user_model}}
            Current working memory: {{working_memory}}
            
            Return a structured update of what should be stored, updated, or removed.
            """
            
            # Fill template with state data
            filled_prompt = memory_prompt.replace("{{memory}}", json.dumps(memory))
            filled_prompt = filled_prompt.replace("{{user_model}}", json.dumps(state.get("user_model", {})))
            filled_prompt = filled_prompt.replace("{{working_memory}}", json.dumps(working_memory))
            
            # Invoke LLM for memory management decisions
            memory_messages = [SystemMessage(content=filled_prompt)]
            memory_response = await llm.ainvoke(memory_messages)
            
            # Log memory consolidation for debugging
            logger.info(f"Memory consolidation: {memory_response.content}")
            
            # Store the consolidation result
            working_memory["last_consolidation"] = {
                "timestamp": datetime.now().isoformat(),
                "result": memory_response.content
            }
        except Exception as e:
            logger.error(f"Memory consolidation error: {str(e)}")
    # =================== END MEMORY MANAGEMENT SECTION ===================
    
    # Normal flow - use LLM for ALL routing decisions including to user modeler
    coordinator_prompt = """You are the coordinator for a personal fitness trainer AI. Your role is to:
    1. Understand the user's current needs and context
    2. Determine which specialized agent should handle the interaction
    3. Provide a coherent experience across different interactions
    4. Ensure all user needs are addressed appropriately
    5. Conduct user assessment when needed

    You have direct access to these specialized capabilities:
    - Research: Retrieve scientific fitness knowledge using the retrieve_from_rag tool
    - Planning: Create personalized workout routines
    - Progress Analysis: Analyze workout data and track progress
    - Adaptation: Modify workout plans based on progress and feedback
    - Coach: Provide motivation and adherence strategies
    - User Modeler: Updates the user model with new information from the user

    IMPORTANT ROUTING INSTRUCTIONS:
    - When a user responds to an assessment question, ALWAYS route to [User Modeler] first
    - The User Modeler will update the profile and then route back to you for next steps
    - If assessment is complete but research_findings is empty, route to [Research]
    - If assessment is complete and research_findings exists but seems irrelevant to current user goals, route to [Research]
    - Only route to [Planning] when assessment is complete AND relevant research is available

    Assessment process:
    - If user profile is incomplete, you should ask assessment questions
    - Required fields for assessment: goals, fitness_level, available_equipment, training_environment, schedule, constraints

    RESPONSE FORMAT:
    1. First provide your internal reasoning (not shown to user)
    2. If choosing [Research], include specific research needs in format: <research_needs>specific research topics and information needed based on user profile</research_needs>
    3. End your internal reasoning with one of these agent tags:
    [Assessment] - If you need to ask an assessment question
    [Research] - If research information is needed
    [Planning] - If workout routine creation is needed
    [Progress] - If progress analysis is needed
    [Adaptation] - If routine modification is needed
    [Coach] - If motivation/coaching is needed
    [User Modeler] - If the user's message contains information that should update their profile, or if the user's message is a response to an assessment question with information in it, then choose this agent.
    [Complete] - If you can directly handle the response
    4. Then wrap your user-facing response in <user>...</user> tags
    5. If the previous response has PLANNERS RESPONSE with a fitness plan/routine, just copy paste the routine in your response (within the user facing tags).

    Current user model: {{user_model}}
    Current fitness plan: {{fitness_plan}}
    Recent interactions: {{recent_exchanges}}
    Research findings: {{research_findings}}
    """

    
    # Fill template with state data
    filled_prompt = coordinator_prompt.replace("{{user_model}}", json.dumps(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{fitness_plan}}", json.dumps(state.get("fitness_plan", {})))
    filled_prompt = filled_prompt.replace("{{recent_exchanges}}", 
                                     json.dumps(working_memory.get("recent_exchanges", [])))
    filled_prompt = filled_prompt.replace("{{research_findings}}", 
                                     json.dumps(working_memory.get("research_findings", {})))
    
    # Invoke LLM with the filled prompt and conversation history
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = await llm.ainvoke(messages)
    
    # Process the response to separate internal reasoning, agent tag, and user response
    content = response.content
    user_response = ""
    internal_reasoning = content
    selected_agent = "coordinator"  # Default
    research_needs = None
    
    # Extract user-facing response
    user_match = re.search(r'<user>(.*?)</user>', content, re.DOTALL)
    if user_match:
        user_response = user_match.group(1).strip()
        internal_reasoning = content.replace(f"<user>{user_response}</user>", "").strip()

    # Extract research needs if present
    research_needs_match = re.search(r'<research_needs>(.*?)</research_needs>', internal_reasoning, re.DOTALL)
    if research_needs_match:
        research_needs = research_needs_match.group(1).strip()
        internal_reasoning = internal_reasoning.replace(f"<research_needs>{research_needs}</research_needs>", "").strip()
    
    # Extract agent tag
    agent_tags = {
        "[Assessment]": "assessment",
        "[Research]": "research_agent",
        "[Planning]": "planning_agent",
        "[Progress]": "progress_analysis_agent",
        "[Adaptation]": "adaptation_agent",
        "[Coach]": "coach_agent",
        "[User Modeler]": "user_modeler",
        "[Complete]": "end_conversation"
    }
    
    for tag, agent in agent_tags.items():
        if tag in internal_reasoning:
            selected_agent = agent
            internal_reasoning = internal_reasoning.replace(tag, "").strip()
            logger.info(f'Coordinator agent decided on the agent: {selected_agent}')
            break
    
    # Update agent state
    agent_state = state.get("agent_state", {})
    
    # Safety check - if assessment is incomplete and we're not routing to user_modeler,
    # force assessment to ensure we collect all required information
    if selected_agent != "user_modeler":
        # Check if assessment is complete
        user_model = state.get("user_model", {})
        required_fields = ["goals", "fitness_level", "available_equipment", "training_environment", "schedule", "constraints"]
        missing_fields = [field for field in required_fields if field not in user_model or not user_model.get(field)]
        
        if missing_fields:
            # Force assessment if fields are missing
            selected_agent = "assessment"
            logger.info(f"Forcing assessment due to missing fields: {missing_fields}")

    # If research is selected, store research needs in working memory
    if selected_agent == "research_agent" and research_needs:
        working_memory["research_needs"] = research_needs
        logger.info(f"Added research needs to working memory: {research_needs}")
    
    if selected_agent == "assessment":
        agent_state["needs_human_input"] = True
    else:
        agent_state["needs_human_input"] = False
    
    # Return updated state with memory components
    updated_state = {
        **state,
        "messages": state["messages"] + [AIMessage(content=user_response)],
        "current_agent": selected_agent,
        "agent_state": agent_state,
        "memory": memory,  # Include updated memory
        "working_memory": working_memory | {  # Merge with new working_memory data
            "internal_reasoning": internal_reasoning,
            "selected_agent": selected_agent
        }
    }
    
    logger.info(f"Coordinator Agent - Output State: {updated_state}")
    return updated_state


async def retrieve_data(query: str) -> str:
    
    """Retrieves science-based exercise information from Pinecone vector store."""
    logger.info(f"RAG query: {query}")
    query_embedding = embeddings.embed_query(query)
    logger.info(f"Generated query embedding: {query_embedding[:5]}... (truncated)")
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    logger.info(f"Pinecone query results: {results}")
    retrieved_docs = [match["metadata"].get("text", "No text available") for match in results["matches"]]
    logger.info(f"Retrieved documents: {retrieved_docs}")
    return "\n".join(retrieved_docs)
          

# Research agent for scientific knowledge retrieval
async def research_agent(state: AgentState) -> AgentState:
    """Retrieves and synthesizes scientific fitness knowledge from RAG."""
    logger.info(f"Research Agent - Input State: {state}") #Log input state
    research_prompt = """You are a fitness research specialist. Based on the user's profile and current needs:
    1. Identify key scientific principles relevant to their goals
    2. Retrieve evidence-based approaches from the rag knowledge base
    3. Synthesize this information into actionable insights
    4. Provide citations to specific sources
    
    Current user profile: {{user_profile}}
    Current research needs: {{research_needs}}
    
    Use the retrieve_from_rag tool to access scientific fitness information.
    """
    
    # Generate targeted queries based on user profile
    user_goals = state.get("user_model", {}).get("goals", ["general fitness"])
    experience_level = state.get("user_model", {}).get("fitness_level", "beginner")
    limitations = state.get("user_model", {}).get("constraints", [])
    
    # Create specific, focused queries for better RAG results
    queries = []
    
    # Goal-specific queries
    for goal in user_goals:
        queries.append(f"optimal training frequency for {goal}")
        queries.append(f"best exercises for {goal}")
        queries.append(f"progression schemes for {goal}")
    
    # Experience-level queries
    queries.append(f"training volume for {experience_level} lifters")
    queries.append(f"intensity recommendations for {experience_level} lifters")
    
    # Limitation-specific queries
    for limitation in limitations:
        queries.append(f"exercise modifications for {limitation}")
        queries.append(f"safe training with {limitation}")
    
    # Fill template with state data
    filled_prompt = research_prompt.replace("{{user_profile}}", json.dumps(state.get("user_model", {})))
    filled_prompt = research_prompt.replace("{{research_needs}}", 
                                         json.dumps(state.get("working_memory", {}).get("research_needs", [])))
    
    # Add research topics to prompt
    filled_prompt += f"\n\nSpecific research topics to investigate: {', '.join(queries)}"
    
    # Invoke LLM with research prompt and conversation history
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = await llm_with_tools.ainvoke(messages)
    
    # Execute multiple targeted RAG queries
    research_results = []
    for query in queries[:3]:  # Limit to first 3 queries for demonstration
        try:
            result = await retrieve_data(query)
            if result:  # Check if result is not None or empty
                research_results.append({
                    "query": query,
                    "result": result,
                    "source": result.get("source", "Unknown")
                })
        except Exception as e:
            research_results.append({
                "query": query,
                "error": str(e),
                "result": "Error retrieving information"
            })
    
    # Update working memory with structured research findings
    working_memory = state.get("working_memory", {})
    working_memory["research_findings"] = {
        "principles": extract_principles(response.content),
        "approaches": extract_approaches(response.content),
        "citations": extract_citations(response.content),
        "raw_results": research_results,
        "timestamp": datetime.now().isoformat()
    }
    
    # Return updated state
    updated_state = {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "working_memory": working_memory
    }
    logger.info(f"Research Agent - Output State: {updated_state}")  # Log output state
    return updated_state

# Planning agent for workout routine creation
async def planning_agent(state: AgentState) -> AgentState:
    """Creates personalized workout routines and exports to Hevy."""

    logger.info(f"Planning Agent - Input State: {state}") #Log input state
    planning_prompt = """You are a workout programming specialist. Create a detailed, personalized workout plan:
    1. Design a structured routine based on scientific principles and user profile
    2. Format the routine specifically for Hevy app integration
    3. Include exercise selection, sets, reps, rest periods, and progression scheme
    4. Provide clear instructions for implementation
    
    User profile: {{user_profile}}
    Research findings: {{research_findings}}

     
    The routine you provided will be converted into pydantic base classes and accessed this way by the user:
    exercises = []
    for exercise_data in routine_structure.get("exercises", []):
        sets = []
        for set_data in exercise_data.get("sets", []):
            sets.append(SetRoutineCreate(
                type=set_data.get("type", "normal"),
                weight_kg=set_data.get("weight", 0.0),
                reps=set_data.get("reps", 0),
                duration_seconds=set_data.get("duration", None),
                distance_meters=set_data.get("distance", None)
            ))
        
        exercises.append(ExerciseRoutineCreate(
            exercise_template_id=exercise_data.get("exercise_id", ""),
            exercise_name=exercise_data.get("exercise_name", ""),
            exercise_type=exercise_data.get("exercise_type", "strength"),
            sets=sets,
            notes=exercise_data.get("notes", ""),
            rest_seconds=60  # Default rest time
        ))
    
    # Create the full routine object
    routine = RoutineCreate(
        title=routine_structure.get("title", "Personalized Routine"),
        notes=routine_structure.get("notes", "AI-generated routine"),
        exercises=exercises
    )
    So make sure you include all the fields in your response
    """
    
    # Fill template with state data
    filled_prompt = planning_prompt.replace("{{user_profile}}", json.dumps(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{research_findings}}", 
                                        json.dumps(state.get("working_memory", {}).get("research_findings", {})))

    
    # Invoke LLM with planning prompt and conversation history
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = await llm.ainvoke(messages)

    logger.info(f"\033[91m PLANNERS RESPONSE: {response} \033[0m")  # Red text in logs
    
    
    try:
        
        fitness_plan = state.get("fitness_plan", {})
        fitness_plan["content"] = response.content
        
        fitness_plan["created_at"] = datetime.now().isoformat()
        fitness_plan["version"] = fitness_plan.get("version", 0) + 1
        logger.info(f'Fitness Plan: {fitness_plan}')
        #
        
        # Return updated state
        updated_state = {
            **state,
            "messages": state["messages"] + [AIMessage(content="PLANNERS OUTPUT:\n"+response.content)],
            "fitness_plan": fitness_plan
        }
        logger.info(f"Planning Agent - Output State: {updated_state}")
        return updated_state
    except Exception as e:
        # Handle errors gracefully
        updated_state = {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=response.content),
                AIMessage(content=f"I've designed this routine for you, but there was an issue saving it to Hevy: {str(e)}")
            ]
        }
        error_message = f"Error creating routine in Hevy: {str(e)}"
        logging.error(f"{error_message} in Planning Agent. Output State: {updated_state}")
        return updated_state

# Progress analysis agent for workout log analysis
async def progress_analysis_agent(state: AgentState) -> AgentState:
    """Analyzes workout logs to track progress and suggest adjustments."""
    logger.info(f"Progress Analysis Agent - Input State: {state}") #Log input state
    analysis_prompt = """You are a fitness progress analyst. Examine the user's workout logs to:
    1. Track adherence to the planned routine
    2. Identify trends in performance (improvements, plateaus, regressions)
    3. Compare actual progress against expected progress
    4. Suggest specific adjustments to optimize results
    
    User profile: {{user_profile}}
    Current fitness plan: {{fitness_plan}}
    Recent workout logs: {{workout_logs}}
    
    Use the tool_fetch_workouts tool to access workout logs from Hevy.
    """
    
    # Fetch recent workout logs from Hevy API
    try:
        workout_logs = await tool_fetch_workouts(page=1, page_size=10)
    except Exception as e:
        workout_logs = {"error": str(e), "message": "Unable to fetch workout logs"}
    
    # Fill template with state data
    filled_prompt = analysis_prompt.replace("{{user_profile}}", json.dumps(state.get("user_model", {})))
    filled_prompt = analysis_prompt.replace("{{fitness_plan}}", json.dumps(state.get("fitness_plan", {})))
    filled_prompt = analysis_prompt.replace("{{workout_logs}}", json.dumps(workout_logs))
    
    # Invoke LLM with analysis prompt and conversation history
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = await llm_with_tools.ainvoke(messages)
    
    # Extract structured analysis results
    analysis_results = {
        "adherence_rate": extract_adherence_rate(response.content),
        "progress_metrics": extract_progress_metrics(response.content),
        "identified_issues": extract_issues(response.content),
        "suggested_adjustments": extract_adjustments(response.content),
        "analysis_date": datetime.now().isoformat()
    }
    
    # Update working memory with analysis date
    working_memory = state.get("working_memory", {})
    working_memory["last_analysis_date"] = datetime.now().isoformat()
    
    # Return updated state
    updated_state = {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "progress_data": state.get("progress_data", {}) | {"latest_analysis": analysis_results},
        "working_memory": working_memory
    }
    logger.info(f"Progress Analysis Agent - Output State: {updated_state}") #Log input state
    return updated_state

# Adaptation agent for routine modification
async def adaptation_agent(state: AgentState) -> AgentState:
    """Modifies workout routines based on progress and feedback."""
    logger.info(f"Adaptation Agent - Input State: {state}") #Log input state
    adaptation_prompt = """You are a workout adaptation specialist. Based on progress data and user feedback:
    1. Identify specific aspects of the routine that need modification
    2. Apply scientific principles to make appropriate adjustments
    3. Ensure changes align with the user's goals and constraints
    4. Update the routine in Hevy
    
    User profile: {{user_profile}}
    Current fitness plan: {{fitness_plan}}
    Progress data: {{progress_data}}
    Suggested adjustments: {{suggested_adjustments}}
    
    Use the tool_update_routine tool to update the routine in Hevy.
    """
    
    # Fill template with state data
    filled_prompt = adaptation_prompt.replace("{{user_profile}}", json.dumps(state.get("user_model", {})))
    filled_prompt = adaptation_prompt.replace("{{fitness_plan}}", json.dumps(state.get("fitness_plan", {})))
    filled_prompt = adaptation_prompt.replace("{{progress_data}}", json.dumps(state.get("progress_data", {})))
    filled_prompt = adaptation_prompt.replace("{{suggested_adjustments}}", 
                                         json.dumps(state.get("progress_data", {})
                                                   .get("latest_analysis", {})
                                                   .get("suggested_adjustments", [])))
    
    # Invoke LLM with adaptation prompt and conversation history
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = await llm_with_tools.ainvoke(messages)
    
    # Extract routine updates from response
    routine_updates = extract_routine_updates(response.content)
    
    # Update routine in Hevy
    try:
        # Get current routine ID
        routine_id = state.get("fitness_plan", {}).get("hevy_routine_id")
        
        if routine_id:
            # Convert to appropriate format for Hevy API
            exercises = []
            for exercise_data in routine_updates.get("exercises", []):
                sets = []
                for set_data in exercise_data.get("sets", []):
                    sets.append(SetRoutineUpdate(
                        type=set_data.get("type", "normal"),
                        weight_kg=set_data.get("weight", 0.0),
                        reps=set_data.get("reps", 0),
                        duration_seconds=set_data.get("duration", None),
                        distance_meters=set_data.get("distance", None)
                    ))
                
                exercises.append(ExerciseRoutineUpdate(
                    exercise_template_id=exercise_data.get("exercise_id", ""),
                    exercise_name=exercise_data.get("exercise_name", ""),
                    exercise_type=exercise_data.get("exercise_type", "strength"),
                    sets=sets,
                    notes=exercise_data.get("notes", ""),
                    rest_seconds=60  # Default rest time
                ))
            
            # Create the update object
            routine_update = RoutineUpdate(
                title=routine_updates.get("title", "Updated Routine"),
                notes=routine_updates.get("notes", "AI-updated routine"),
                exercises=exercises
            )
            
            # Create the request object
            update_request = RoutineUpdateRequest(routine=routine_update)
            
            # Call the API
            hevy_result = await tool_update_routine(routine_id, update_request)
            
            # Update fitness plan with modifications
            fitness_plan = state.get("fitness_plan", {})
            fitness_plan["content"] = response.content
            fitness_plan["updated_at"] = datetime.now().isoformat()
            fitness_plan["version"] = fitness_plan.get("version", 0) + 1
            
            # Return updated state
            updated_state = {
                **state,
                "messages": state["messages"] + [AIMessage(content=response.content)],
                "fitness_plan": fitness_plan
            }
            logger.info(f"Adaptation Agent - Output State: {updated_state}") #Log input state
            return updated_state
        else:
            # Handle missing routine ID
            updated_state = {
                **state,
                "messages": state["messages"] + [
                    AIMessage(content=response.content),
                    AIMessage(content="I've designed these updates for your routine, but I couldn't find your existing routine in Hevy. Would you like me to create a new routine instead?")
                ]
            }
            logger.info(f"Adaptation Agent - Output State: {updated_state}") #Log input state
            return updated_state
    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error updating routine in Hevy: {str(e)}"
    
        updated_state = {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=response.content),
                AIMessage(content=f"I've designed these updates for your routine, but there was an issue saving them to Hevy: {str(e)}")
            ]
        }
        logger.error(f"{error_message} in Adaptation Agent. Output State: {updated_state}") #Log input state
        return updated_state

# Coach agent for motivation and adherence
async def coach_agent(state: AgentState) -> AgentState:
    """Provides motivation, adherence strategies, and behavioral coaching."""
    logger.info(f"Coach Agent - Input State: {state}") #Log input state
    coach_prompt = """You are a fitness motivation coach. Your role is to:
    1. Provide encouragement and motivation tailored to the user's profile
    2. Offer strategies to improve adherence and consistency
    3. Address psychological barriers to fitness progress
    4. Celebrate achievements and milestones
    
    User profile: {{user_profile}}
    Progress data: {{progress_data}}
    Recent exchanges: {{recent_exchanges}}
    
    Be supportive, empathetic, and science-based in your approach.
    """
    
    # Fill template with state data
    filled_prompt = coach_prompt.replace("{{user_profile}}", json.dumps(state.get("user_model", {})))
    filled_prompt = coach_prompt.replace("{{progress_data}}", json.dumps(state.get("progress_data", {})))
    filled_prompt = coach_prompt.replace("{{recent_exchanges}}", 
                                         json.dumps(state.get("working_memory", {}).get("recent_exchanges", [])))
    
    # Invoke LLM with coaching prompt and conversation history
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = await llm.ainvoke(messages)
    
    # Return updated state
    updated_state = {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }
    logger.info(f"Coach Agent - Output State: {updated_state}") #Log input state
    return 

# Agent selector function
async def agent_selector(state: AgentState, reasoning_text: str = "") -> str:
    """Determines which agent should handle the current interaction."""

    # For new users, ensure comprehensive assessment
    if state.get("agent_state", {}).get("status") == "complete":
        return "end_conversation"

    if not state.get("user_model") or state.get("user_model", {}).get("assessment_complete") != True:
        return "assessment_agent"
    
    # For users with assessment but no plan, prioritize research and planning
    if state.get("user_model") and not state.get("fitness_plan"):
        if not state.get("working_memory", {}).get("research_findings"):
            return "research_agent"
        else:
            return "planning_agent"
    
    # For users with existing plans, check if analysis is needed
    if state.get("fitness_plan") and state.get("working_memory", {}).get("last_analysis_date"):
        last_analysis = datetime.fromisoformat(state.get("working_memory", {}).get("last_analysis_date"))
        if (datetime.now() - last_analysis).days >= 7:  # Weekly analysis
            return "progress_analysis_agent"
    
    # Use reasoning text to determine appropriate agent
    if reasoning_text:
        if "assessment" in reasoning_text.lower() or "profile" in reasoning_text.lower():
            return "assessment_agent"
        elif "research" in reasoning_text.lower() or "knowledge" in reasoning_text.lower():
            return "research_agent"
        elif "plan" in reasoning_text.lower() or "routine" in reasoning_text.lower():
            return "planning_agent"
        elif "progress" in reasoning_text.lower() or "analyze" in reasoning_text.lower():
            return "progress_analysis_agent"
        elif "adjust" in reasoning_text.lower() or "modify" in reasoning_text.lower():
            return "adaptation_agent"
        elif "motivate" in reasoning_text.lower() or "coach" in reasoning_text.lower():
            return "coach_agent"
    
    # Default to coordinator for general interactions
    return "coordinator"

# State management utilities
async def get_or_create_state(session_id: str) -> AgentState:
    """Retrieve existing state or create a new one."""
    # In a real implementation, this would check a database
    # For now, just create a new state
    return {
        "messages": [],
        "session_id": session_id,
        "memory": {},
        "working_memory": {},
        "user_model": {},
        "fitness_plan": {},
        "progress_data": {},
        "reasoning_trace": [],
        "agent_state": {"status": "new"},
        "current_agent": "coordinator",
        "tool_calls": None,
        "tool_results": None
    }

async def save_state(session_id: str, state: AgentState) -> None:
    """Save state to persistent storage."""
    # In a real implementation, this would save to a database
    # For now, just print a message
    print(f"State saved for session {session_id}")

# Enhanced error handling wrapper
def agent_with_error_handling(agent_func):
    """Wrapper to add error handling to agent functions."""
    async def wrapped_agent(state: AgentState) -> AgentState:
        try:
            return await agent_func(state)
        except Exception as e:
            # Log the error
            print(f"Error in {agent_func.__name__}: {str(e)}")
            
            # Add error message to state
            error_message = f"I encountered an issue while processing your request: {str(e)}"
            state["messages"].append(AIMessage(content=error_message))
            
            # Update agent state
            state["agent_state"] = state.get("agent_state", {}) | {"error": str(e), "status": "error"}
            
            # Return to coordinator for recovery
            state["current_agent"] = "coordinator"
            
            return state
    
    return wrapped_agent

async def end_conversation(state: AgentState) -> AgentState:

    """Marks the conversation as complete."""
    logging.info(f"End Conversation. Input State: {state}")
    state["agent_state"] = state.get("agent_state", {}) | {"status": "complete"}
    state["conversation_complete"] = True
    logging.info(f"End Conversation. Output State: {state}")
    return state  # Return the modified state, not END


def coordinator_condition(state):
    """Determine the next node based on state conditions."""
    selected_agent = state.get("current_agent", "coordinator")
    
    if selected_agent == "assessment":
        return "end_conversation"  # Route to end node for user input
    elif selected_agent == "end_conversation":
        return "end_conversation"
    else:
        # Route to the selected agent
        return selected_agent


# Build the graph
def build_fitness_trainer_graph():
    """Constructs the simplified multi-agent graph for the fitness trainer."""
    workflow = StateGraph(AgentState)
    checkpointer = MemorySaver()
    
    # Add nodes
    workflow.add_node("coordinator", agent_with_error_handling(coordinator))
    workflow.add_node("user_modeler", agent_with_error_handling(user_modeler))
    workflow.add_node("research_agent", agent_with_error_handling(research_agent))
    workflow.add_node("planning_agent", agent_with_error_handling(planning_agent))
    workflow.add_node("progress_analysis_agent", agent_with_error_handling(progress_analysis_agent))
    workflow.add_node("adaptation_agent", agent_with_error_handling(adaptation_agent))
    workflow.add_node("coach_agent", agent_with_error_handling(coach_agent))
    
    
    # Add tools node
    workflow.add_node("tools", ToolNode([
        retrieve_from_rag,
        tool_fetch_workouts,
        tool_get_workout_count,
        tool_fetch_routines,
        tool_update_routine,
        tool_create_routine
    ]))
    
    # Add the end node for handling user input
    workflow.add_node("end_conversation", agent_with_error_handling(end_conversation))
    
    # Define the edges - route based on coordinator's decision
    workflow.add_conditional_edges(
        "coordinator",
        coordinator_condition,
        {
            "research_agent": "research_agent",
            "planning_agent": "planning_agent",
            "progress_analysis_agent": "progress_analysis_agent",
            "adaptation_agent": "adaptation_agent",
            "coach_agent": "coach_agent",
            "user_modeler": "user_modeler",
            "end_conversation": "end_conversation"
        }
    )
    
    # Connect everything back to coordinator
    workflow.add_edge("research_agent", "coordinator")
    workflow.add_edge("planning_agent", "coordinator")
    workflow.add_edge("progress_analysis_agent", "coordinator")
    workflow.add_edge("adaptation_agent", "coordinator")
    workflow.add_edge("coach_agent", "coordinator")
    workflow.add_edge("user_modeler", "coordinator")
    
    
    # Connect end_conversation to END for final responses
    workflow.add_edge("end_conversation", END)
    
    workflow.set_entry_point("coordinator")
    
    # Compile the graph
    return workflow.compile(checkpointer=checkpointer)



