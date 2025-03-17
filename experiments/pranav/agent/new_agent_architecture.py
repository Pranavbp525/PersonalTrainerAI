"""
Enhanced Multi-Agent Architecture for Personal Fitness Trainer AI

This implementation combines the strengths of the multi-agent system with
specialized components for fitness training, RAG integration, and Hevy API usage.
"""

from typing import Annotated, List, Dict, Any, Optional, Literal, Union, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from datetime import datetime, timedelta
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
import json
import os
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

# Define comprehensive state model
class AgentState(TypedDict):
    # Core state
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    
    # Memory components
    memory: Dict[str, Any]  # Long-term memory storage
    working_memory: Dict[str, Any]  # Short-term contextual memory
    
    # User and fitness data
    user_model: Dict[str, Any]  # Comprehensive user profile
    fitness_plan: Dict[str, Any]  # Structured fitness plan
    progress_data: Dict[str, Any]  # Progress tracking data
    
    # Agent management
    reasoning_trace: List[Dict[str, Any]]  # Traces of reasoning steps
    agent_state: Dict[str, str]  # Current state of each agent
    current_agent: str  # Currently active agent
    
    # Tool interaction
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[Dict]

# Helper functions for data extraction
def extract_principles(text: str) -> List[str]:
    """Extract scientific principles from text."""
    # Implementation would parse the text to identify key principles
    # For now, return a placeholder
    return ["principle1", "principle2"]

def extract_approaches(text: str) -> List[Dict]:
    """Extract training approaches from text."""
    # Implementation would parse the text to identify approaches
    # For now, return a placeholder
    return [{"name": "approach1", "description": "description1"}]

def extract_citations(text: str) -> List[Dict]:
    """Extract citations from text."""
    # Implementation would parse the text to identify citations
    # For now, return a placeholder
    return [{"source": "Jeff Nippard", "content": "content1"}]

def extract_routine_data(text: str) -> Dict:
    """Extract structured routine data for Hevy API."""
    # Implementation would parse the text to create a structured routine
    # For now, return a placeholder
    return {
        "name": "Routine",
        "description": "Description",
        "workouts": []
    }

def extract_adherence_rate(text: str) -> float:
    """Extract adherence rate from analysis text."""
    # Implementation would parse the text to identify adherence rate
    # For now, return a placeholder
    return 0.85

def extract_progress_metrics(text: str) -> Dict:
    """Extract progress metrics from analysis text."""
    # Implementation would parse the text to identify progress metrics
    # For now, return a placeholder
    return {"strength": 0.1, "endurance": 0.05}

def extract_issues(text: str) -> List[str]:
    """Extract identified issues from analysis text."""
    # Implementation would parse the text to identify issues
    # For now, return a placeholder
    return ["issue1", "issue2"]

def extract_adjustments(text: str) -> List[Dict]:
    """Extract suggested adjustments from analysis text."""
    # Implementation would parse the text to identify adjustments
    # For now, return a placeholder
    return [{"target": "exercise1", "change": "increase weight"}]

def extract_routine_structure(text: str) -> Dict:
    """Extract detailed routine structure from text for Hevy API."""
    # This would be a more sophisticated version of extract_routine_data
    # that creates properly structured data for the Hevy API
    
    # For demonstration, return a simplified structure
    return {
        "title": "Personalized Routine",
        "notes": "AI-generated routine",
        "exercises": [
            {
                "exercise_name": "Bench Press",
                "exercise_id": "bench_press_id",
                "exercise_type": "strength",
                "sets": [
                    {"reps": 10, "weight": 135.0, "type": "normal"}
                ],
                "notes": "Focus on form"
            }
        ]
    }

def extract_routine_updates(text: str) -> Dict:
    """Extract routine updates from text for Hevy API."""
    # Similar to extract_routine_structure but for updates
    return {
        "title": "Updated Routine",
        "notes": "AI-updated routine",
        "exercises": [
            {
                "exercise_name": "Bench Press",
                "exercise_id": "bench_press_id",
                "exercise_type": "strength",
                "sets": [
                    {"reps": 8, "weight": 145.0, "type": "normal"}
                ],
                "notes": "Increased weight"
            }
        ]
    }

# Memory manager for sophisticated state management
async def memory_manager(state: AgentState) -> AgentState:
    """Manages long-term and working memory, consolidating information and pruning as needed."""
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
    filled_prompt = memory_prompt.replace("{{memory}}", json.dumps(state.get("memory", {})))
    filled_prompt = filled_prompt.replace("{{user_model}}", json.dumps(state.get("user_model", {})))
    filled_prompt = filled_prompt.replace("{{working_memory}}", json.dumps(state.get("working_memory", {})))
    
    # Invoke LLM for memory management decisions
    messages = [SystemMessage(content=filled_prompt)]
    response = await llm.ainvoke(messages)
    
    # Process recent messages for context
    recent_exchanges = []
    for msg in state["messages"][-10:]:  # Last 10 messages
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            recent_exchanges.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
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
    
    # Return updated state
    return {
        **state,
        "memory": memory,
        "working_memory": working_memory
    }

# Reasoning engine for sophisticated decision-making
async def reasoning_engine(state: AgentState) -> AgentState:
    """Advanced reasoning to determine the optimal next steps and actions."""
    reasoning_prompt = """You are the reasoning engine for a fitness training system. Analyze the current situation using sophisticated reasoning to determine:
    1. What is the user's current need or intent?
    2. What information do we currently have and what's missing?
    3. What potential approaches could address the user's needs?
    4. What are the tradeoffs between different approaches?
    5. What is the optimal next action given all constraints?
    
    User model: {{user_model}}
    Working memory: {{working_memory}}
    Fitness plan: {{fitness_plan}}
    
    Think step by step and document your reasoning process.
    """
    
    # Fill template with state data
    filled_prompt = reasoning_prompt.replace("{{user_model}}", json.dumps(state.get("user_model", {})))
    filled_prompt = reasoning_prompt.replace("{{working_memory}}", json.dumps(state.get("working_memory", {})))
    filled_prompt = reasoning_prompt.replace("{{fitness_plan}}", json.dumps(state.get("fitness_plan", {})))
    
    # Invoke LLM with the reasoning prompt
    messages = [SystemMessage(content=filled_prompt)]
    response = await llm.ainvoke(messages)
    
    # Extract reasoning trace and store in state
    reasoning_trace = state.get("reasoning_trace", [])
    reasoning_trace.append({
        "timestamp": datetime.now().isoformat(),
        "reasoning": response.content,
        "context": {
            "current_agent": state.get("current_agent"),
            "working_memory_snapshot": state.get("working_memory", {})
        }
    })
    
    # Determine next agent based on reasoning
    agent_selection = await agent_selector(state, response.content)
    
    # Update working memory with reasoning results
    working_memory = state.get("working_memory", {})
    working_memory["reasoning_summary"] = response.content
    working_memory["selected_agent"] = agent_selection
    
    # Return updated state
    return {
        **state,
        "reasoning_trace": reasoning_trace,
        "current_agent": agent_selection,
        "working_memory": working_memory
    }

# User modeler for comprehensive user understanding
async def user_modeler(state: AgentState) -> AgentState:
    """Builds and maintains a comprehensive model of the user."""
    modeling_prompt = """You are a user modeling specialist for a fitness training system. Analyze all available information about the user to build a comprehensive model:
    1. Extract explicit information (stated goals, preferences, constraints)
    2. Infer implicit information (fitness level, motivation factors, learning style)
    3. Identify gaps in our understanding that need to be addressed
    4. Update confidence levels for different aspects of the model
    
    Current user model: {{user_model}}
    Recent exchanges: {{recent_exchanges}}
    
    Return an updated user model with confidence scores for each attribute.
    """
    
    # Fill template with state data
    filled_prompt = modeling_prompt.replace("{{user_model}}", json.dumps(state.get("user_model", {})))
    filled_prompt = modeling_prompt.replace("{{recent_exchanges}}", 
                                         json.dumps(state.get("working_memory", {}).get("recent_exchanges", [])))
    
    # Invoke LLM for user modeling
    messages = [SystemMessage(content=filled_prompt)]
    response = await llm.ainvoke(messages)
    
    # Update user model
    user_model = state.get("user_model", {})
    user_model["last_updated"] = datetime.now().isoformat()
    user_model["model_version"] = user_model.get("model_version", 0) + 1
    
    # In a real implementation, we would parse the response to update specific fields
    # For now, we'll just store the raw response
    user_model["latest_analysis"] = response.content
    
    # Return updated state
    return {
        **state,
        "user_model": user_model
    }

# Coordinator agent for overall orchestration
async def coordinator_agent(state: AgentState) -> AgentState:
    """Central coordinator that manages the overall interaction flow."""
    coordinator_prompt = """You are the coordinator for a personal fitness trainer AI. Your role is to:
    1. Understand the user's current needs and context
    2. Determine which specialized agent should handle the interaction
    3. Provide a coherent experience across different agents
    4. Ensure all user needs are addressed appropriately
    
    You have access to these specialized agents:
    - Assessment Agent: Gathers user information and builds profiles
    - Research Agent: Retrieves scientific fitness knowledge
    - Planning Agent: Creates personalized workout routines
    - Progress Analysis Agent: Analyzes workout data and tracks progress
    - Adaptation Agent: Modifies workout plans based on progress and feedback
    - Coach Agent: Provides motivation and adherence strategies
    
    Current user model: {{user_model}}
    Current fitness plan: {{fitness_plan}}
    Recent interactions: {{recent_exchanges}}
    
    Respond to the user and indicate which agent should handle the next step.
    """
    
    # Fill template with state data
    filled_prompt = coordinator_prompt.replace("{{user_model}}", json.dumps(state.get("user_model", {})))
    filled_prompt = coordinator_prompt.replace("{{fitness_plan}}", json.dumps(state.get("fitness_plan", {})))
    filled_prompt = coordinator_prompt.replace("{{recent_exchanges}}", 
                                         json.dumps(state.get("working_memory", {}).get("recent_exchanges", [])))
    
    # Invoke LLM with the filled prompt and conversation history
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = await llm.ainvoke(messages)
    
    # Return updated state with the coordinator's response added to messages
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }

# Assessment agent for user profiling
async def assessment_agent(state: AgentState) -> AgentState:
    """Handles comprehensive user assessment and profile building."""
    assessment_prompt = """You are a fitness assessment specialist. Your goal is to build a comprehensive user profile by asking targeted questions about:
    1. Fitness goals (strength, hypertrophy, endurance, weight loss, etc.)
    2. Training experience and current fitness level
    3. Available equipment and training environment
    4. Schedule and time availability
    5. Physical limitations, injuries, or health concerns
    6. Dietary preferences and restrictions
    7. Measurement data (if willing to share)
    8. Previous workout history and experiences
    
    Current profile: {{user_profile}}
    
    Be conversational but thorough. Focus on gathering missing information based on what you already know.
    If this is the first interaction, introduce yourself and begin the assessment process.
    """
    
    # Fill template with state data
    filled_prompt = assessment_prompt.replace("{{user_profile}}", json.dumps(state.get("user_model", {})))
    
    # Invoke LLM with assessment prompt and conversation history
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = await llm_with_tools.ainvoke(messages)
    
    # Update user model with assessment information
    user_model = state.get("user_model", {})
    
    # Mark assessment as complete if we have sufficient information
    # This is a simplified check - in reality, we would verify specific fields
    if len(state["messages"]) > 5:  # Arbitrary threshold for demonstration
        user_model["assessment_complete"] = True
    
    # Return updated state
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "user_model": user_model
    }

# Research agent for scientific knowledge retrieval
async def research_agent(state: AgentState) -> AgentState:
    """Retrieves and synthesizes scientific fitness knowledge from RAG."""
    research_prompt = """You are a fitness research specialist. Based on the user's profile and current needs:
    1. Identify key scientific principles relevant to their goals
    2. Retrieve evidence-based approaches from fitness influencers like Jeff Nippard, Dr. Mike Isratel, and Jeremy Ethier
    3. Synthesize this information into actionable insights
    4. Provide citations to specific sources
    
    Current user profile: {{user_profile}}
    Current research needs: {{research_needs}}
    
    Use the retrieve_from_rag tool to access scientific fitness information.
    """
    
    # Generate targeted queries based on user profile
    user_goals = state.get("user_model", {}).get("goals", ["general fitness"])
    experience_level = state.get("user_model", {}).get("experience_level", "beginner")
    limitations = state.get("user_model", {}).get("limitations", [])
    
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
            result = await retrieve_from_rag(query)
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
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "working_memory": working_memory
    }

# Planning agent for workout routine creation
async def planning_agent(state: AgentState) -> AgentState:
    """Creates personalized workout routines and exports to Hevy."""
    planning_prompt = """You are a workout programming specialist. Create a detailed, personalized workout plan:
    1. Design a structured routine based on scientific principles and user profile
    2. Format the routine specifically for Hevy app integration
    3. Include exercise selection, sets, reps, rest periods, and progression scheme
    4. Provide clear instructions for implementation
    
    User profile: {{user_profile}}
    Research findings: {{research_findings}}
    
    Use the tool_create_routine tool to save the routine to Hevy.
    """
    
    # Fill template with state data
    filled_prompt = planning_prompt.replace("{{user_profile}}", json.dumps(state.get("user_model", {})))
    filled_prompt = planning_prompt.replace("{{research_findings}}", 
                                         json.dumps(state.get("working_memory", {}).get("research_findings", {})))
    
    # Invoke LLM with planning prompt and conversation history
    messages = state["messages"] + [SystemMessage(content=filled_prompt)]
    response = await llm_with_tools.ainvoke(messages)
    
    # Extract structured routine data for Hevy API
    routine_structure = extract_routine_structure(response.content)
    
    # Create properly structured Hevy API objects
    try:
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
        
        # Create the request object
        create_request = RoutineCreateRequest(routine=routine)
        
        # Call the API
        hevy_result = await tool_create_routine(create_request)
        
        # Update fitness plan with both content and Hevy reference
        fitness_plan = state.get("fitness_plan", {})
        fitness_plan["content"] = response.content
        fitness_plan["hevy_routine_id"] = hevy_result.get("id")
        fitness_plan["created_at"] = datetime.now().isoformat()
        fitness_plan["version"] = fitness_plan.get("version", 0) + 1
        
        # Return updated state
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "fitness_plan": fitness_plan
        }
    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error creating routine in Hevy: {str(e)}"
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=response.content),
                AIMessage(content=f"I've designed this routine for you, but there was an issue saving it to Hevy: {str(e)}")
            ]
        }

# Progress analysis agent for workout log analysis
async def progress_analysis_agent(state: AgentState) -> AgentState:
    """Analyzes workout logs to track progress and suggest adjustments."""
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
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "progress_data": state.get("progress_data", {}) | {"latest_analysis": analysis_results},
        "working_memory": working_memory
    }

# Adaptation agent for routine modification
async def adaptation_agent(state: AgentState) -> AgentState:
    """Modifies workout routines based on progress and feedback."""
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
            return {
                **state,
                "messages": state["messages"] + [AIMessage(content=response.content)],
                "fitness_plan": fitness_plan
            }
        else:
            # Handle missing routine ID
            return {
                **state,
                "messages": state["messages"] + [
                    AIMessage(content=response.content),
                    AIMessage(content="I've designed these updates for your routine, but I couldn't find your existing routine in Hevy. Would you like me to create a new routine instead?")
                ]
            }
    except Exception as e:
        # Handle errors gracefully
        error_message = f"Error updating routine in Hevy: {str(e)}"
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=response.content),
                AIMessage(content=f"I've designed these updates for your routine, but there was an issue saving them to Hevy: {str(e)}")
            ]
        }

# Coach agent for motivation and adherence
async def coach_agent(state: AgentState) -> AgentState:
    """Provides motivation, adherence strategies, and behavioral coaching."""
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
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }

# Agent selector function
async def agent_selector(state: AgentState, reasoning_text: str = "") -> str:
    """Determines which agent should handle the current interaction."""
    # For new users, ensure comprehensive assessment
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
    return "coordinator_agent"

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
        "current_agent": "coordinator_agent",
        "tool_calls": None,
        "tool_results": None
    }

async def save_state(session_id: str, state: AgentState) -> None:
    """Save state to persistent storage."""
    # In a real implementation, this would save to a database
    # For now, just print a message
    print(f"State saved for session {session_id}")

# Enhanced error handling wrapper
async def agent_with_error_handling(agent_func, state: AgentState) -> AgentState:
    """Wrapper to add error handling to agent functions."""
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
        state["current_agent"] = "coordinator_agent"
        
        return state

# Build the graph
def build_fitness_trainer_graph():
    """Constructs the multi-agent graph for the fitness trainer."""
    workflow = StateGraph(AgentState)
    
    # Add core nodes
    workflow.add_node("coordinator", agent_with_error_handling(coordinator_agent, {}))
    workflow.add_node("reasoning", agent_with_error_handling(reasoning_engine, {}))
    workflow.add_node("memory", agent_with_error_handling(memory_manager, {}))
    workflow.add_node("user_modeler", agent_with_error_handling(user_modeler, {}))
    
    # Add specialized agent nodes
    workflow.add_node("assessment", agent_with_error_handling(assessment_agent, {}))
    workflow.add_node("research", agent_with_error_handling(research_agent, {}))
    workflow.add_node("planning", agent_with_error_handling(planning_agent, {}))
    workflow.add_node("progress_analysis", agent_with_error_handling(progress_analysis_agent, {}))
    workflow.add_node("adaptation", agent_with_error_handling(adaptation_agent, {}))
    workflow.add_node("coach", agent_with_error_handling(coach_agent, {}))
    
    # Add tools node
    workflow.add_node("tools", ToolNode([
        retrieve_from_rag,
        tool_fetch_workouts,
        tool_get_workout_count,
        tool_fetch_routines,
        tool_update_routine,
        tool_create_routine
    ]))
    
    # Define the edges
    # Start with coordinator
    workflow.add_edge("coordinator", "reasoning")
    workflow.add_edge("reasoning", "memory")
    
    # From memory to agent selection
    workflow.add_conditional_edges(
        "memory",
        lambda state: state["current_agent"],
        {
            "assessment_agent": "assessment",
            "research_agent": "research",
            "planning_agent": "planning",
            "progress_analysis_agent": "progress_analysis",
            "adaptation_agent": "adaptation",
            "coach_agent": "coach",
            "coordinator_agent": "coordinator"
        }
    )
    
    # From each agent back to coordinator
    workflow.add_edge("assessment", "user_modeler")
    workflow.add_edge("user_modeler", "coordinator")
    workflow.add_edge("research", "coordinator")
    workflow.add_edge("planning", "coordinator")
    workflow.add_edge("progress_analysis", "coordinator")
    workflow.add_edge("adaptation", "coordinator")
    workflow.add_edge("coach", "coordinator")
    
    # Add tool handling
    workflow.add_conditional_edges(
        "coordinator",
        tools_condition,
        {
            True: "tools",
            False: "reasoning"
        }
    )
    workflow.add_edge("tools", "coordinator")
    
    # Add conditional end
    workflow.add_conditional_edges(
        "coordinator",
        lambda state: "END" if state.get("agent_state", {}).get("status") == "complete" else "reasoning",
        {
            "END": END,
            "reasoning": "reasoning"
        }
    )
    
    # Compile the graph
    return workflow.compile()

# Main agent execution function
async def run_agent(user_input: str, session_id: str = None):
    """Main entry point for agent execution with proper async handling."""
    # Initialize or retrieve state
    if not session_id:
        session_id = str(uuid.uuid4())
    
    state = await get_or_create_state(session_id)
    
    # Add user input to messages
    state["messages"].append(HumanMessage(content=user_input))
    
    # Create the agent graph
    graph = build_fitness_trainer_graph()
    
    # Run the graph
    result = await graph.ainvoke(state)
    
    # Save state
    await save_state(session_id, result)
    
    # Return the latest assistant message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            return msg.content
    
    return "I processed your request but encountered an issue generating a response."

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        response = await run_agent("Hi, I want to start a new workout routine to build muscle.")
        print(response)
    
    asyncio.run(main())
