"""
Enhanced Multi-Agent Architecture for Personal Fitness Trainer AI

This implementation combines the strengths of the multi-agent system with
specialized components for fitness training, RAG integration, and Hevy API usage.
"""

from typing import Annotated, List, Dict, Any, Optional, Literal, Union, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.types import Command, interrupt

from langgraph.graph.message import add_messages
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
import json
import os
import logging
import re

import uuid
from pydantic import BaseModel, Field
from .agent_models import (
    SetRoutineCreate, ExerciseRoutineCreate, RoutineCreate, RoutineCreateRequest,
    SetRoutineUpdate, ExerciseRoutineUpdate, RoutineUpdate, RoutineUpdateRequest
)
from .agent_models import AgentState, UserProfile
from .utils import (extract_adherence_rate,
                   extract_adjustments,
                   extract_approaches,
                   extract_citations,
                   extract_issues,
                   extract_principles,
                   extract_progress_metrics,
                   extract_routine_data,
                   extract_routine_structure,
                   extract_routine_updates,
                   retrieve_data)
from .prompts import (
    get_adaptation_prompt,
    get_analysis_prompt,
    get_coach_prompt,
    get_coordinator_prompt,
    get_memory_consolidation_prompt,
    get_planning_prompt,
    get_research_prompt,
    get_user_modeler_prompt
)


from .llm_tools import (
    tool_fetch_workouts,
    tool_get_workout_count,
    tool_fetch_routines,
    tool_update_routine,
    tool_create_routine,
    retrieve_from_rag
)



from dotenv import load_dotenv


from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain.schema.runnable import RunnableLambda


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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



# User modeler for comprehensive user understanding
async def user_modeler(state: AgentState) -> AgentState:
    """Builds and maintains a comprehensive model of the user."""
    logger.info(f"User Modeler - Input State: {state}") #Log input state

    parser = PydanticOutputParser(pydantic_object=UserProfile)
    format_instructions = parser.get_format_instructions()

    # Define the prompt template with format instructions
    prompt_template = get_user_modeler_prompt()
    
    current_user_model = state.get("user_model", {})
    recent_exchanges = state.get("working_memory", {}).get("recent_exchanges", [])
    
    # Format the prompt
    formatted_prompt = prompt_template.format(
        user_model=json.dumps(current_user_model),
        recent_exchanges=json.dumps(recent_exchanges),
        format_instructions=format_instructions
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
            memory_prompt = get_memory_consolidation_prompt()
            
            # Fill template with state data
            filled_prompt = memory_prompt.format(
                memory=json.dumps(memory),
                user_model=json.dumps(state.get("user_model", {})),
                working_memory=json.dumps(working_memory)
            )
            
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
    coordinator_prompt = get_coordinator_prompt()

    filled_prompt = coordinator_prompt.format(
        user_model=json.dumps(state.get("user_model", {})),
        fitness_plan=json.dumps(state.get("fitness_plan", {})),
        recent_exchanges=json.dumps(working_memory.get("recent_exchanges", [])),
        research_findings=json.dumps(working_memory.get("research_findings", {}))
    )

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



          

# Research agent for scientific knowledge retrieval
async def research_agent(state: AgentState) -> AgentState:
    """Retrieves and synthesizes scientific fitness knowledge from RAG."""
    logger.info(f"Research Agent - Input State: {state}") #Log input state
    research_prompt = get_research_prompt()
    
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
    filled_prompt = research_prompt.format(
    user_profile=json.dumps(state.get("user_model", {})),
    research_needs=json.dumps(state.get("working_memory", {}).get("research_needs", [])))
    
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
    planning_prompt = get_planning_prompt()  # Latest version
    # Or for specific version: planning_prompt = get_planning_prompt("production") 

    # Format the prompt properly using the template's format method
    filled_prompt = planning_prompt.format(
        user_profile=json.dumps(state.get("user_model", {})),
        research_findings=json.dumps(state.get("working_memory", {}).get("research_findings", {}))
    )
    
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
    analysis_prompt = get_analysis_prompt()
    
    # Fetch recent workout logs from Hevy API
    try:
        workout_logs = await tool_fetch_workouts(page=1, page_size=10)
    except Exception as e:
        workout_logs = {"error": str(e), "message": "Unable to fetch workout logs"}
    

    filled_prompt = analysis_prompt.format(
        user_profile = json.dumps(state.get("user_model", {})),
        fitness_plan = json.dumps(state.get("fitness_plan", {})),
        workout_logs = json.dumps(workout_logs)
    )
    
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
    adaptation_prompt = get_adaptation_prompt()
    
    
    filled_prompt = adaptation_prompt.format(
        user_profile = json.dumps(state.get("user_model", {})),
        fitness_plan = json.dumps(state.get("fitness_plan", {})),
        progress_data = json.dumps(state.get("progress_data", {})),
        suggested_adjustments = json.dumps(state.get("progress_data", {})
                                                   .get("latest_analysis", {})
                                                   .get("suggested_adjustments", []))
    )
    
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
    coach_prompt = get_coach_prompt()
    
    filled_prompt = coach_prompt.format(
        user_profile=json.dumps(state.get("user_model", {})),
        progress_data=json.dumps(state.get("progress_data", {})),
        recent_exchanges=json.dumps(state.get("working_memory", {}).get("recent_exchanges", []))
    )
    
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

async def end_conversation(state: AgentState) -> AgentState:

    """Marks the conversation as complete."""
    logging.info(f"End Conversation. Input State: {state}")
    state["agent_state"] = state.get("agent_state", {}) | {"status": "complete"}
    state["conversation_complete"] = True
    logging.info(f"End Conversation. Output State: {state}")
    return state  # Return the modified state, not END


