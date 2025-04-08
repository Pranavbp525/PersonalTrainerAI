"""
Enhanced Multi-Agent Architecture for Personal Fitness Trainer AI

This implementation combines the strengths of the multi-agent system with
specialized components for fitness training, RAG integration, and Hevy API usage.
"""

from typing import Annotated, List, Dict, Any, Optional, Literal, Union, TypedDict, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.types import Command, interrupt
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
# from langchain_core.pydantic_v1 import BaseModel, Field # Use v1 Pydantic with LangChain/Graph typically
from langchain.output_parsers.pydantic import PydanticOutputParser
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
import json
import os
import logging
import re
import asyncio

import uuid
from pydantic import BaseModel, Field
from agent.agent_models import (
    SetRoutineCreate, ExerciseRoutineCreate, RoutineCreate, RoutineCreateRequest,
    SetRoutineUpdate, ExerciseRoutineUpdate, RoutineUpdate, RoutineUpdateRequest,
    PlannerExerciseRoutineCreate, PlannerRoutineCreate, PlannerSetRoutineCreate, PlannerOutputContainer, HevyRoutineApiPayload,
    AnalysisFindings, IdentifiedRoutineTarget,
    RoutineAdaptationResult
)
from agent.agent_models import AgentState, UserProfile, DeepFitnessResearchState, StreamlinedRoutineState, ProgressAnalysisAdaptationStateV2
from agent.utils import (extract_adherence_rate,
                   extract_adjustments,
                   extract_approaches,
                   extract_citations,
                   extract_issues,
                   extract_principles,
                   extract_progress_metrics,
                   extract_routine_data,
                   extract_routine_structure,
                   extract_routine_updates,
                   retrieve_data,
                   get_exercise_template_by_title_fuzzy,
                   validate_and_lookup_exercises)
from agent.prompts import (
    get_adaptation_prompt,
    get_analysis_prompt,
    get_coach_prompt,
    get_coordinator_prompt,
    get_memory_consolidation_prompt,
    get_planning_prompt,
    get_research_prompt,
    get_user_modeler_prompt,
    summarize_routine_prompt,
    get_analysis_v2_template,
    get_final_cycle_report_template_v2,
    get_reasoning_generation_template,
    get_routine_identification_prompt,
    get_routine_modification_template_v2,
    get_targeted_rag_query_template
)


from agent.llm_tools import (
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

    # --- <<<< ADD CHECK FOR RETURN FROM PROGRESS/ADAPTATION SUBGRAPH >>>> ---
    # Use the distinct state key names if you added them
    progress_notification = state.get("final_report_and_notification") 
    if progress_notification is not None:
        logger.info("Coordinator: Detected return from progress_adaptation_subgraph.")
        working_memory = state.get("working_memory", {})
        processed_results = state.get("processed_results") # Optional details
        success_status = state.get("cycle_completed_successfully") # Optional status

        # Log the event in working memory
        working_memory[f"progress_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = {
            "status": "Success" if success_status else "Failed/Partial",
            "notification_sent": progress_notification,
            # Optionally include summary of processed_results here
        }

        # Prepare state update
        next_state = {**state}
        # Add notification message
        next_state["messages"] = add_messages(state.get("messages", []), [AIMessage(content=f"<user>{progress_notification}</user>")])
        # Update working memory
        next_state["working_memory"] = working_memory
        # Clean up subgraph outputs from state
        next_state["final_report_and_notification"] = None
        next_state["cycle_completed_successfully"] = None
        next_state["processed_results"] = None
        # next_state["process_error"] = None # Clear specific subgraph error too
        # Clear the trigger context if it was set
        next_state["user_request_context"] = None
        # Route to end conversation to show the user
        next_state["current_agent"] = "end_conversation"

        logger.info("Coordinator: Processed progress subgraph return. Routing to end_conversation.")
        return next_state
    
    # --- Check if returning from Planning Subgraph ---
    # Detect return by checking if hevy_results exists and is not None
    # (It would be None otherwise or before the subgraph runs)
    returned_hevy_results = state.get("hevy_results")
    if returned_hevy_results is not None:
        logger.info("Coordinator: Detected return from planning_subgraph (hevy_results is present).")

        # Store the payloads in working memory for persistence/future reference
        working_memory = state.get("working_memory", {})
        generated_routines_key = f"generated_hevy_routines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        working_memory[generated_routines_key] = returned_hevy_results
        logger.info(f"Stored generated routines in working_memory under key: {generated_routines_key}")

        # Prepare state for summarization
        payloads_json = json.dumps(returned_hevy_results, indent=2)
        summary_prompt_filled = summarize_routine_prompt.format(hevy_results_json=payloads_json)

        user_facing_summary = "An error occurred while summarizing the generated plan." # Default error message
        try:
            logger.info("Coordinator: Generating user-facing summary of routines...")
            summary_response = await llm.ainvoke([SystemMessage(content=summary_prompt_filled)])
            user_facing_summary = summary_response.content.strip()
            logger.info("Coordinator: Summary generated successfully.")

        except Exception as summary_err:
            logger.error(f"Coordinator: Error generating routine summary: {summary_err}", exc_info=True)
            # Keep the default error message

        # --- Prepare the state update ---
        # Start with the current state
        next_state = {**state}
        # Update messages with the summary
        next_state["messages"] = add_messages(state.get("messages", []), [AIMessage(content=f"<user>{user_facing_summary}</user>")])
        # Store the updated working memory
        next_state["working_memory"] = working_memory
        # Clear the subgraph output from the main state level
        next_state["hevy_results"] = None
        # Keep errors for now, might need review
        # next_state["errors"] = state.get("errors", [])
        # Route back to coordinator to decide the *next* step after summarizing
        next_state["current_agent"] = "end_conversation"

        logger.info("Coordinator Agent - Output State (After Planning Summary): Routing back to end-conversation to show results to user.")
        return next_state


    # --- Check if returning from Deep Research --- <<<< ADD THIS BLOCK AT START >>>>
    final_report = state.get("final_report")
    if final_report is not None: # Check specifically for non-None
        logger.info("Coordinator: Received final report from deep_research subgraph.")
        # Integrate the report into working memory
        working_memory = state.get("working_memory", {})
        research_findings = working_memory.get("research_findings", {}) # Get current findings dict
        research_findings["report"] = final_report
        research_findings["report_timestamp"] = datetime.now().isoformat()

        working_memory["research_findings"] = research_findings
        working_memory["research_needs"] = None # Clear the needs trigger that started research

        # Clean up subgraph communication keys from the main state
        clean_state = {k: v for k, v in state.items() if k not in ['final_report', 'research_topic', 'user_profile_str']}
        clean_state["working_memory"] = working_memory # Add updated WM back

        logger.info("Coordinator: Integrated report, clearing subgraph keys.")

        # Decide next step - likely planning now that research is done
        ai_message = AIMessage(content="<user>Okay, I've completed the research based on your profile and needs. Now I can create a plan, or would you like to discuss the findings?</user>")
        clean_state["messages"] = add_messages(clean_state.get("messages", []), [ai_message])
        clean_state["current_agent"] = "planning_agent" # Route to planning

        logger.info(f"Coordinator Agent - Output State (After Research): Routing to {clean_state['current_agent']}")
        return clean_state # Return updated state

    
    logger.info("Coordinator: Detected nothing. Resuming Normal Flow.")

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

    # coordinator_prompt_obj = get_coordinator_prompt()
    # logger.info(f"Coordinator received prompt object of type: {type(coordinator_prompt_obj)}") # Verify type before format



    filled_prompt = coordinator_prompt.format(
        user_model=json.dumps(state.get("user_model", {})),
        fitness_plan=json.dumps(state.get("hevy_results", {})),
        recent_exchanges=json.dumps(working_memory.get("recent_exchanges", [])),
        research_findings=json.dumps(working_memory.get("research_findings", {}))
    )

    # logger.error(f"\033[91mCO-ORDINATOR PROMPT CHECK - \n {filled_prompt}\033[0m")

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
        "<Assessment>": "assessment",
        "<Research>": "deep_research",
        "<Planning>": "planning_agent",
        "<Progress_and_Adaptation>": "progress_analysis_adaptation_agent",
        "<Coach>": "coach_agent",
        "<User Modeler>": "user_modeler",
        "<Complete>": "end_conversation"
    }
    
    for tag, agent in agent_tags.items():
        if tag in internal_reasoning:
            selected_agent = agent
            internal_reasoning = internal_reasoning.replace(tag, "").strip()
            logger.info(f'Coordinator agent decided on the agent: {selected_agent}')
            break
    
    # Update agent state
    agent_state = state.get("agent_state", {})

    # --- Prepare Subgraph Input if Routing to Deep Research --- <<<< MODIFY/ADD THIS BLOCK >>>>
    next_state_update = {} # Prepare dictionary for state updates
    working_memory = state.get("working_memory", {}) # Get current WM

    if selected_agent == "deep_research":
        # Determine the research topic
        topic = research_needs if research_needs else working_memory.get('research_needs')
        if not topic:
             # Fallback: base topic on user goals if no specific needs identified
             user_goals = state.get("user_model", {}).get("goals")
             if user_goals:
                 topic = f"Research fitness principles for achieving goals: {', '.join(user_goals)}"
             else:
                 topic = "General fitness planning based on user profile."
             logger.warning(f"No specific research needs found, using derived/default topic: {topic}")

        logger.info(f"Coordinator: Preparing state for deep_research subgraph. Topic: '{topic}'")

        # Set the shared state keys for the subgraph
        next_state_update["research_topic"] = topic
        next_state_update["user_profile_str"] = json.dumps(state.get("user_model", {})) # Use the JSON string created earlier

        # Ensure working memory has the trigger cleared/updated if needed
        working_memory["research_needs"] = topic # Store the definitive topic used

        # Clear any old report from previous runs
        next_state_update["final_report"] = None

    elif selected_agent == "planning_agent":
         logger.info("Coordinator: Preparing state for planning_subgraph.")
         
         # Clear previous outputs explicitly before calling subgraph
         next_state_update["hevy_results"] = None

         # Ensure other subgraph inputs are cleared
         next_state_update["research_topic"] = None
         next_state_update["user_profile_str"] = None
         next_state_update["final_report"] = None

    # --- <<<< ADD PREPARATION FOR PROGRESS/ADAPTATION SUBGRAPH >>>> ---
    elif selected_agent == "progress_adaptation_agent":
        logger.info("Coordinator: Preparing state for progress_adaptation_subgraph.")
        # Extract context for the subgraph if available
        last_message = state["messages"][-1]
        request_context = None
        if isinstance(last_message, HumanMessage):
             # A simple approach - use the last human message as context
             request_context = last_message.content
             logger.debug(f"Using last human message as user_request_context: {request_context}")
        elif working_memory.get("system_trigger_review"): # Check for external trigger flag
             request_context = working_memory.pop("system_trigger_review") # Use and remove flag
             logger.debug(f"Using system trigger as user_request_context: {request_context}")

        next_state_update["user_request_context"] = request_context
        # Clear outputs from other subgraphs
        next_state_update["final_report"] = None
        next_state_update["hevy_results"] = None
        next_state_update["final_report_and_notification"] = None # Clear its own previous output


    else:
        # Clear subgraph keys if routing elsewhere (important for clean state)
        next_state_update["research_topic"] = None
        next_state_update["user_profile_str"] = None
        next_state_update["final_report"] = None
        next_state_update["hevy_results"] = None
        next_state_update["user_request_context"] = None
        next_state_update["progress_report_and_notification"] = None # Typo
        next_state_update["final_report_and_notification"] = None # Corrected key
        next_state_update["cycle_completed_successfully"] = None # Clear progress output
        next_state_update["processed_results"] = None # Clear progress output
        # Clear research needs from WM if not routing to research? Optional.
        # working_memory["research_needs"] = None

    
    # Safety check - if assessment is incomplete and we're not routing to user_modeler,
    # force assessment to ensure we collect all required information
    if selected_agent != "user_modeler":
        # Check if assessment is complete
        user_model = state.get("user_model", {})
        required_fields = ["goals", "fitness_level", "available_equipment", "training_environment", "schedule", "constraints"]
        missing_fields = [field for field in required_fields if field not in user_model or not user_model.get(field)]
        
        if missing_fields and selected_agent!= "assessment":
            # Force assessment if fields are missing
            selected_agent = "assessment"
            logger.info(f"Forcing assessment due to missing fields: {missing_fields}")

    # If research is selected, store research needs in working memory
    # if selected_agent == "deep_research" and research_needs:
    #     working_memory["research_needs"] = research_needs
    #     logger.info(f"Added research needs to working memory: {research_needs}")
    
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
        },
        "research_topic": next_state_update.get("research_topic"), # Use .get() which defaults to None
        "user_profile_str": next_state_update.get("user_profile_str"),
        "final_report": next_state_update.get("final_report"),
        "hevy_results": next_state_update.get("hevy_results"),
        "user_request_context": next_state_update.get("user_request_context"),
        "final_report_and_notification": next_state_update.get("final_report_and_notification"),
        "cycle_completed_successfully": next_state_update.get("cycle_completed_successfully"),
        "processed_results": next_state_update.get("processed_results"),

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
                    "source": "Pinecone RAG"
                    # "source": result.get("source", "Unknown") # Add source field to pinecone metadata later for citations
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
        
        fitness_plan = state.get("hevy_payloads", {})
        fitness_plan["content"] = response.content
        
        fitness_plan["created_at"] = datetime.now().isoformat()
        fitness_plan["version"] = fitness_plan.get("version", 0) + 1
        logger.info(f'Fitness Plan: {fitness_plan}')
        #
        
        # Return updated state
        updated_state = {
            **state,
            "messages": state["messages"] + [AIMessage(content="PLANNERS OUTPUT:\n"+response.content)],
            "hevy_payloads": fitness_plan
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
        fitness_plan = json.dumps(state.get("hevy_payloads", {})),
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
        fitness_plan = json.dumps(state.get("hevy_payloads", {})),
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
        routine_id = state.get("hevy_payloads", {}).get("hevy_routine_id")
        
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
            fitness_plan = state.get("hevy_payloads", {})
            fitness_plan["content"] = response.content
            fitness_plan["updated_at"] = datetime.now().isoformat()
            fitness_plan["version"] = fitness_plan.get("version", 0) + 1
            
            # Return updated state
            updated_state = {
                **state,
                "messages": state["messages"] + [AIMessage(content=response.content)],
                "hevy_payloads": fitness_plan
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



##################################### Deep Research Agent ######################################
async def plan_research_steps(state: DeepFitnessResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Uses LLM to break down the research topic into sub-questions.
    Initializes the internal state for the research loop, including limits.
    """
    logger.info("--- Deep Research: Planning Steps ---")
    topic = state['research_topic']
    user_profile = state['user_profile_str']

    prompt = f"""Given the main research topic: '{topic}' for a user with this profile:
<user_profile>
{user_profile}
</user_profile>

Break this down into 3-5 specific, actionable sub-questions relevant to fitness science that can likely be answered using our internal knowledge base (RAG system). Focus on aspects like training principles, exercise selection, progression, nutrition timing, recovery, etc., as relevant to the topic.

Output ONLY a JSON list of strings, where each string is a sub-question. Example:
["What are the optimal rep ranges for muscle hypertrophy based on recent studies?", "How does protein timing affect muscle protein synthesis post-workout?", "What are common exercise modifications for individuals with lower back pain?"]
"""
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        sub_questions_str = response.content.strip()
        logger.debug(f"LLM response for planning: {sub_questions_str}")
        # Basic parsing, consider adding more robust JSON cleaning if needed
        sub_questions = json.loads(sub_questions_str)
        if not isinstance(sub_questions, list) or not all(isinstance(q, str) for q in sub_questions):
            raise ValueError("LLM did not return a valid JSON list of strings.")
        logger.info(f"Planned sub-questions: {sub_questions}")

    except Exception as e:
        logger.error(f"Error planning research steps: {e}. Using topic as single question.")
        sub_questions = [topic] # Fallback

    # Initialize state, getting limits from config or using defaults
    run_config = config.get("configurable", {}) if config else {}
    max_iterations = run_config.get("max_iterations", 5)
    max_queries_per_sub_q = run_config.get("max_queries_per_sub_question", 2) # Default to 2 queries/sub-question

    return {
        "sub_questions": sub_questions,
        "accumulated_findings": f"Initial research topic: {topic}\nUser Profile Summary: {user_profile}\n\nStarting research...\n",
        "iteration_count": 0,
        "current_sub_question_idx": 0,
        "reflections": [],
        "research_complete": False,
        "queries_this_sub_question": 0,
        "sub_question_complete_flag": False, # Initialize flag
        "max_iterations": max_iterations,
        "max_queries_per_sub_question": max_queries_per_sub_q
    }

async def generate_rag_query_v2(state: DeepFitnessResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Generates RAG query for the current sub-question and increments counters.
    """
    logger.info("--- Deep Research: Generating RAG Query (v2) ---")
    # Increment overall iteration count
    new_iteration_count = state.get('iteration_count', 0) + 1
    # Increment per-sub-question query count
    queries_count_this_sub_q = state.get('queries_this_sub_question', 0) + 1

    if not state.get('sub_questions'):
        logger.warning("No sub_questions found, cannot generate query.")
        return {"current_rag_query": None, "iteration_count": new_iteration_count, "queries_this_sub_question": queries_count_this_sub_q}

    current_idx = state['current_sub_question_idx']
    if current_idx >= len(state['sub_questions']):
         logger.info("All sub-questions processed based on index.")
         # Should be caught by router, but handle defensively
         return {"current_rag_query": None, "research_complete": True, "iteration_count": new_iteration_count, "queries_this_sub_question": queries_count_this_sub_q}

    current_sub_question = state['sub_questions'][current_idx]
    findings = state['accumulated_findings']
    reflections_str = "\n".join(state.get('reflections', []))

    prompt = f"""You are a research assistant formulating queries for an internal fitness science knowledge base (RAG system accessed via the `retrieve_data` function).

Current Research Sub-Question: "{current_sub_question}"
Query attempt number {queries_count_this_sub_q} for this sub-question.

Accumulated Findings So Far:
<findings>
{findings}
</findings>

Previous Reflections on Progress:
<reflections>
{reflections_str}
</reflections>

Based on the *current sub-question* and the information gathered or reflected upon so far, formulate the single, most effective query string to retrieve the *next piece* of relevant scientific information from our fitness RAG system. Be specific and targeted. If previous attempts failed to yield useful info, try a different angle.

Output *only* the query string itself, without any explanation or preamble.
"""
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        query = response.content.strip()
        # Basic check for empty query
        if not query:
            raise ValueError("LLM returned an empty query.")
        logger.info(f"Generated RAG query: {query}")
        query_to_run = query
    except Exception as e:
        logger.error(f"Error generating RAG query: {e}. Using fallback.")
        query_to_run = f"Details about {current_sub_question}" # Fallback

    return {
        "current_rag_query": query_to_run,
        "iteration_count": new_iteration_count, # Return updated overall count
        "queries_this_sub_question": queries_count_this_sub_q # Return updated per-question count
        }

async def execute_rag_direct(state: DeepFitnessResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Directly executes the retrieve_data function using the current_rag_query.
    """
    logger.info("--- Deep Research: Executing RAG Query ---")
    query = state.get('current_rag_query')

    if not query:
        logger.warning("No RAG query provided in state. Skipping execution.")
        return {"rag_results": "No query was provided."}

    try:
        # Directly call the imported async RAG function
        logger.debug(f"Calling retrieve_data with query: {query}")
        rag_output = await retrieve_data(query=query)

        logger.info(f"RAG execution successful. Result length: {len(rag_output)}")
        # Handle potential empty results from RAG
        if not rag_output or rag_output.strip() == "":
            logger.warning(f"RAG query '{query}' returned empty results.")
            return {"rag_results": f"No information found in knowledge base for query: '{query}'"}

        return {"rag_results": rag_output}
    except Exception as e:
        logger.error(f"Error executing RAG query '{query}': {e}", exc_info=True)
        return {"rag_results": f"Error retrieving information for query '{query}': {str(e)}"}

async def synthesize_rag_results(state: DeepFitnessResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Uses LLM to integrate the latest RAG results into the accumulated findings.
    """
    logger.info("--- Deep Research: Synthesizing RAG Results ---")
    rag_results = state.get('rag_results')
    if not rag_results or "No query was provided" in rag_results or "No information found" in rag_results or "Error retrieving information" in rag_results:
        logger.warning(f"Skipping synthesis due to missing or problematic RAG results: {rag_results}")
        # Optionally add a marker about the failed query
        error_marker = f"\n\n[Skipped synthesis for query: '{state.get('current_rag_query', 'N/A')}' due to RAG result: {rag_results}]\n"
        return {"accumulated_findings": state['accumulated_findings'] + error_marker}

    current_idx = state['current_sub_question_idx']
    current_sub_question = state['sub_questions'][current_idx] if state.get('sub_questions') and current_idx < len(state['sub_questions']) else "the current topic"
    findings = state['accumulated_findings']

    prompt = f"""You are a research assistant synthesizing information for a fitness report.

Current Research Sub-Question: "{current_sub_question}"

Existing Accumulated Findings:
<existing_findings>
{findings}
</existing_findings>

Newly Retrieved Information from Knowledge Base (RAG):
<new_info>
{rag_results}
</new_info>

Task: Integrate the key points from the "Newly Retrieved Information" into the "Existing Accumulated Findings". Focus *only* on information directly relevant to the "Current Research Sub-Question". Update the findings concisely and maintain a logical flow. Avoid redundancy. If the new info isn't relevant or adds nothing substantially new, state that briefly within the updated findings.

Output *only* the complete, updated accumulated findings text. Do not include headers like "Updated Findings".
"""
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        updated_findings = response.content.strip()
        logger.debug(f"Synthesized findings length: {len(updated_findings)}")
        # Append a marker indicating which query produced this update
        query_marker = f"\n\n[Synthesized info based on RAG query: '{state.get('current_rag_query', 'N/A')}']\n"
        return {"accumulated_findings": updated_findings + query_marker}
    except Exception as e:
        logger.error(f"Error synthesizing RAG results: {e}")
        # Append error message instead of synthesizing
        error_marker = f"\n\n[Error synthesizing results for query: '{state.get('current_rag_query', 'N/A')}'. RAG Results: {rag_results}. Error: {e}]\n"
        return {"accumulated_findings": findings + error_marker}

async def reflect_on_progress_v2(state: DeepFitnessResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Reflects on progress, decides if sub-question is complete (naturally or by force),
    and updates the index and per-question query count accordingly.
    """
    logger.info("--- Deep Research: Reflecting on Progress (v2) ---")
    current_idx = state['current_sub_question_idx']
    sub_questions = state.get('sub_questions', [])

    # Pre-check: If index is already out of bounds, something went wrong before, signal completion
    if not sub_questions or current_idx >= len(sub_questions):
        logger.warning("Cannot reflect, index out of bounds or no sub-questions.")
        return {
            "reflections": ["Reflection skipped: Index out of bounds."],
            "sub_question_complete_flag": True,
            "current_sub_question_idx": current_idx, # Keep index as is
            "queries_this_sub_question": 0 # Reset counter
            }

    current_sub_question = sub_questions[current_idx]
    findings = state['accumulated_findings']
    queries_this_sub_q = state.get('queries_this_sub_question', 0)
    max_queries_sub_q = state.get('max_queries_per_sub_question', 2)

    # Check if the per-question limit is hit *before* asking the LLM
    force_next_question = queries_this_sub_q >= max_queries_sub_q

    prompt = f"""You are an expert research assistant evaluating the progress on a specific fitness research sub-question.

Current Sub-Question: "{current_sub_question}"
Number of queries made for this sub-question: {queries_this_sub_q} (Max recommended: {max_queries_sub_q})

Accumulated Findings Gathered So Far:
<findings>
{findings}
</findings>

Task: Critically evaluate the findings related *specifically* to the current sub-question.
1.  Assess Sufficiency: Is the sub-question adequately answered based on the findings?
2.  Identify Gaps: What specific, crucial information related to this sub-question is still missing or unclear?
3.  Suggest Next Step: Based on the gaps, should we:
    a) Perform another RAG query for *this* sub-question? (If yes, briefly suggest *what* to query for).
    b) Conclude this sub-question and move to the next?

{"Note: Max queries for this sub-question reached. You should lean towards concluding unless a major, critical gap remains." if force_next_question else ""}

Format your response clearly, addressing points 1, 2, and 3. Start your response with "CONCLUSION:" followed by either "CONTINUE_SUB_QUESTION" or "SUB_QUESTION_COMPLETE".

Example 1 (Needs More):
CONCLUSION: CONTINUE_SUB_QUESTION
Sufficiency: Partially answered...
Gaps: Need details on...
Next Step: Perform another RAG query focusing on...

Example 2 (Sufficient):
CONCLUSION: SUB_QUESTION_COMPLETE
Sufficiency: Yes, the findings cover the core aspects adequately.
Gaps: Minor details could be explored, but not critical.
Next Step: Conclude this sub-question.
"""
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        reflection_text = response.content.strip()
        logger.info(f"Reflection on '{current_sub_question}':\n{reflection_text}")

        # Determine natural completion based on LLM response
        natural_completion = "CONCLUSION: SUB_QUESTION_COMPLETE" in reflection_text.upper()

        # Final decision: complete if natural OR if forced by limit
        is_complete = natural_completion or force_next_question

        if force_next_question and not natural_completion:
             logger.warning(f"Forcing completion of sub-question {current_idx + 1} due to query limit ({max_queries_sub_q}).")
             # Optionally add note to reflection
             reflection_text += "\n\n[Note: Sub-question concluded due to query limit.]"

        # Update index and reset counter *only* if moving to next question
        next_idx = current_idx + 1 if is_complete else current_idx
        queries_reset = 0 if is_complete else queries_this_sub_q # Reset only if index advances

        return {
            "reflections": [reflection_text], # Add new reflection
            "sub_question_complete_flag": is_complete, # Use final decision
            "current_sub_question_idx": next_idx,
            "queries_this_sub_question": queries_reset
            }
    except Exception as e:
        logger.error(f"Error reflecting on progress: {e}")
        # Force completion on error to avoid loops
        return {
            "reflections": [f"Reflection Error: {e}. Assuming sub-question complete."],
            "sub_question_complete_flag": True,
            "current_sub_question_idx": current_idx + 1, # Advance index
            "queries_this_sub_question": 0 # Reset counter
            }

async def finalize_research_report(state: DeepFitnessResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Uses LLM to generate the final research report based on all findings.
    """
    logger.info("--- Deep Research: Finalizing Report ---")
    topic = state['research_topic']
    findings = state['accumulated_findings']
    reflections = state.get('reflections', [])
    sub_questions = state.get('sub_questions', [])

    prompt = f"""You are a research assistant compiling a final report based *only* on information gathered from our internal fitness science knowledge base.

Main Research Topic: "{topic}"

Original Research Plan (Sub-questions):
{json.dumps(sub_questions, indent=2)}

Accumulated Findings (Synthesized from RAG results):
<findings>
{findings}
</findings>

Reflections During Research:
<reflections>
{json.dumps(reflections, indent=2)}
</reflections>

Task: Generate a comprehensive, well-structured research report addressing the main topic.
- Use *only* the information presented in the "Accumulated Findings". Do not add external knowledge.
- Structure the report logically, perhaps following the flow of the sub-questions, synthesizing related points.
- Incorporate insights or limitations mentioned in the "Reflections" where appropriate (e.g., mention if a topic was concluded due to limits or lack of info).
- Ensure the report is clear, concise, and scientifically grounded based *only* on the provided findings.
- Start the report directly. Do not include a preamble like "Here is the final report".

Output the final report text.
"""
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        final_report = response.content.strip()
        logger.info("Generated final research report.")

        return {
            "final_report": final_report,
            "research_complete": True # Mark overall research as complete
            }
    except Exception as e:
        logger.error(f"Error generating final report: {e}")
        return {
            "final_report": f"Error generating report: {e}\n\nRaw Findings:\n{findings}",
            "research_complete": True # Mark complete even on error to exit loop
            }




##################################### Routine Planner and Creation Agent ######################################

async def structured_planning_node(state: StreamlinedRoutineState) -> StreamlinedRoutineState:
    """Generates the workout plan directly as a validated Pydantic object using with_structured_output."""
    logger.info("--- Executing Structured Planning Node (using with_structured_output) ---")
    # messages = state["messages"]
    user_model_str = json.dumps(state.get("user_model", {}), indent=2)
    research_findings_str = json.dumps(state.get("working_memory", {}).get("research_findings", {}), indent=2)
    errors = state.get("errors", [])

    # Prompt Template - Simplified: No explicit format instructions needed here usually
    # Rely on the model descriptions and the structure passed via with_structured_output
    PLANNING_PROMPT_TEMPLATE = """You are an expert personal trainer specializing in evidence-based workout programming.
Based on the user profile and research findings provided below, create a detailed workout plan.

**Critical Instructions:**
1.  **Exercise Names:** For `exercise_name` in each exercise, use the MOST SPECIFIC name possible, including the equipment used (e.g., 'Bench Press (Barbell)', 'Squat (Barbell)', 'Lat Pulldown (Cable)', 'Arnold Press (Dumbbell)'). This is crucial for matching with the exercise database. Do NOT use generic names like 'Bench Press' or 'Row' without specifying equipment.
2.  **Completeness:** Fill in all required fields based on the requested output structure. Provide reasonable defaults for optional fields if appropriate (e.g., rest times = 60s).
3.  **Multiple Routines:** If the plan involves multiple workout days (e.g., Push/Pull/Legs), create a separate routine object within the `routines` list for each day.

User Profile:
{user_profile}

Research Findings:
{research_findings}

Generate the workout plan adhering strictly to the required output structure based on the user profile and research findings.
"""

    prompt = PromptTemplate(
        template=PLANNING_PROMPT_TEMPLATE,
        input_variables=["user_profile", "research_findings"],
        # No partial_variables for format_instructions needed here
    )

    prompt_str = prompt.format(user_profile=user_model_str, research_findings=research_findings_str)

    # Prepare messages for the LLM
    # Use the state's messages if you want history, or just the system prompt + human prompt
    # For structured output, sometimes just the latest instruction is cleaner
    current_messages_for_llm = [
        SystemMessage(content=prompt_str),
        # You might optionally include the initial human message from state['messages'][0]
        # HumanMessage(content=state['messages'][0].content)
    ]


    try:
        # *** Use with_structured_output ***
        # Pass the desired Pydantic output model class
        # Ensure your 'llm' instance supports this method and function/tool calling
        structured_llm = llm.with_structured_output(PlannerOutputContainer)

        # Ainvoke with the prepared messages
        parsed_output_container = await structured_llm.ainvoke(current_messages_for_llm)

        # Check the type just in case, although it should be the Pydantic object
        if isinstance(parsed_output_container, PlannerOutputContainer):
            planner_routines = parsed_output_container.routines
            logger.info(f"Successfully received structured output with {len(planner_routines)} routine(s) via with_structured_output.")

            structured_message_content = f"Planner generated {len(planner_routines)} routine(s) via structured output:\n" + \
                                         "\n".join([f"- {r.title}" for r in planner_routines])
            structured_message = AIMessage(content=structured_message_content) # Add summary to messages

            return {
                **state,
                # "messages": messages + [structured_message], # Add summary to original messages
                "planner_structured_output": planner_routines,
                "errors": errors
            }
        else:
            # This case should ideally not happen if with_structured_output works correctly
            logger.error(f"llm.with_structured_output returned unexpected type: {type(parsed_output_container)}")
            error_message = AIMessage(content="⚠️ Error: Planner returned an unexpected data type after requesting structured output.")
            return {
                **state,
                # "messages": messages + [error_message],
                "planner_structured_output": None,
                "errors": errors + ["Structured Planning Node Failed: Unexpected return type from with_structured_output"]
                }

    # Catch potential errors during the LLM call or structuring process
    except Exception as e:
        logger.error(f"Error using llm.with_structured_output: {e}", exc_info=True)
        # Check if the error message indicates a Pydantic schema issue (common if V1/V2 conflict)
        error_detail = str(e)
        if "model_json_schema" in error_detail or "schema_json" in error_detail:
             error_detail += " (Possible Pydantic V1/V2 incompatibility with with_structured_output. Consider migrating models to standard Pydantic V2.)"

        error_message = AIMessage(content=f"⚠️ Error: Failed to generate structured plan using with_structured_output.\nDetails: {error_detail}")
        return {
            **state,
            # "messages": messages + [error_message],
            "planner_structured_output": None,
            "errors": errors + [f"Structured Planning Node Failed (with_structured_output): {error_detail}"]
        }

async def format_and_lookup_node(state: StreamlinedRoutineState) -> StreamlinedRoutineState:
    """Looks up exercise IDs and formats routines for the Hevy API tool."""
    logger.info("--- Executing Format & Lookup Node ---")
    planner_routines: Optional[List[PlannerRoutineCreate]] = state.get("planner_structured_output")
    # messages = state["messages"]
    errors = state.get("errors", [])
    hevy_payloads = []

    if not planner_routines:
        logger.warning("No structured planner output found to format.")
        if not state.get("errors"): # Only add error if no previous error occurred
             errors.append("Format/Lookup Node Error: No structured plan available from planner.")
        return {**state, "hevy_payloads": [], "errors": errors}

    routine_errors = []
    processed_routine_count = 0
    skipped_exercises = []

    for planner_routine in planner_routines:
        logger.info(f"Formatting routine: '{planner_routine.title}'")
        hevy_exercises = []
        superset_mapping = {}
        next_superset_id_numeric = 0
        exercise_skipped_in_this_routine = False

        for planner_ex in planner_routine.exercises:
            exercise_name = planner_ex.exercise_name # Name from Planner output
            logger.debug(f"Processing exercise: '{exercise_name}'")

            template_info = await get_exercise_template_by_title_fuzzy(exercise_name)

            if not template_info or not template_info.get("id"):
                err_msg = f"Exercise '{exercise_name}' in routine '{planner_routine.title}' could not be matched to a Hevy exercise template. Skipping exercise."
                logger.warning(err_msg)
                routine_errors.append(err_msg)
                skipped_exercises.append(f"'{exercise_name}' (in routine '{planner_routine.title}')")
                exercise_skipped_in_this_routine = True
                continue

            exercise_template_id = template_info["id"]
            matched_title = template_info["title"] # Official Hevy title
            logger.debug(f"Matched '{exercise_name}' to '{matched_title}' (ID: {exercise_template_id})")

            # --- Handle Sets (Mapping PlannerSet -> SetRoutineCreate) ---
            # ... (set handling code remains the same) ...
            hevy_sets = []
            for planner_set in planner_ex.sets:
                 weight = planner_set.weight_kg if planner_set.weight_kg is not None else 0
                 reps = planner_set.reps
                 set_type = planner_set.type if planner_set.type is not None else "normal"
                 hevy_sets.append(SetRoutineCreate(
                     type=set_type,
                     weight_kg=weight,
                     reps=reps,
                     distance_meters=planner_set.distance_meters,
                     duration_seconds=planner_set.duration_seconds,
                 ))
            if not hevy_sets and planner_ex.sets:
                 logger.warning(f"No valid sets created for exercise '{matched_title}' although sets were defined conceptually. Skipping exercise.")
                 exercise_skipped_in_this_routine = True
                 continue


            # --- Handle Supersets ---
            # ... (superset handling code remains the same) ...
            hevy_superset_id = None
            conceptual_superset_group = planner_ex.superset_id
            if conceptual_superset_group is not None and str(conceptual_superset_group).strip():
                group_key = str(conceptual_superset_group).strip()
                if group_key not in superset_mapping:
                    superset_mapping[group_key] = str(next_superset_id_numeric)
                    next_superset_id_numeric += 1
                hevy_superset_id = superset_mapping[group_key]
                logger.debug(f"Assigning Hevy superset_id '{hevy_superset_id}' for planner group '{group_key}'")


            # --- **** MODIFIED DEFAULT FOR NOTES **** ---
            # If planner_ex.notes is None or empty, default to the matched_title
            exercise_notes = planner_ex.notes or matched_title
            # --- **** END OF MODIFICATION **** ---
            # --- *** ADD DEFAULT HANDLING FOR REST SECONDS *** ---
            rest_time = planner_ex.rest_seconds if planner_ex.rest_seconds is not None else 60 # Or use target default: ExerciseRoutineCreate.__fields__['rest_seconds'].default


            # --- Create Hevy Exercise Object (ExerciseRoutineCreate) ---
            hevy_exercises.append(ExerciseRoutineCreate(
                exercise_template_id=exercise_template_id,
                superset_id=hevy_superset_id,
                rest_seconds=rest_time,
                notes=exercise_notes, # Use the modified notes
                sets=hevy_sets
            ))

        # --- Construct Final Routine Payload ---
        # ... (rest of the node remains the same) ...
        if hevy_exercises:
            final_hevy_routine = RoutineCreate(
                title=planner_routine.title,
                notes=planner_routine.notes or "", # Routine notes can maybe be empty? Check API. Defaulting to ""
                exercises=hevy_exercises
            )
            api_payload = HevyRoutineApiPayload(routine_data=final_hevy_routine)
            hevy_payloads.append(api_payload.model_dump(exclude_none=True))
            processed_routine_count += 1
            logger.info(f"Successfully formatted routine '{planner_routine.title}' for Hevy API.")
        elif planner_routine.exercises:
             err_msg = f"Routine '{planner_routine.title}' had no valid/matched exercises after formatting. Skipping routine export."
             logger.warning(err_msg)
             routine_errors.append(err_msg)


    final_errors = errors + routine_errors

    
    # Add summary message
    summary_parts = []
    if processed_routine_count > 0:
        summary_parts.append(f"Successfully prepared {processed_routine_count} routine(s) for Hevy export.")
    if skipped_exercises:
         # Use set to avoid duplicate exercise names in summary
         unique_skipped = sorted(list(set(skipped_exercises)))
         summary_parts.append(f"Skipped exercises due to matching issues: {', '.join(unique_skipped)}.")
    if planner_routines and len(planner_routines) > processed_routine_count:
         num_skipped_routines = len(planner_routines) - processed_routine_count
         summary_parts.append(f"{num_skipped_routines} routine(s) could not be fully prepared or were empty.")

    summary_message = " ".join(summary_parts) if summary_parts else "Formatting complete. No routines to process or format."
    
    return {
        **state,
        # "messages": messages + [AIMessage(content=summary_message)],
        "hevy_payloads": hevy_payloads,
        "errors": final_errors
    }

async def tool_execution_node(state: StreamlinedRoutineState) -> StreamlinedRoutineState:
    """Executes the tool_create_routine for each formatted payload."""
    logger.info("--- Executing Tool Execution Node ---")
    payloads = state.get("hevy_payloads", [])
    # messages = state["messages"]
    errors = state.get("errors", [])
    hevy_results = []

    if not payloads:
        logger.info("No Hevy payloads to execute.")
        return {**state, "hevy_results": []}

    logger.info(f"Attempting to create {len(payloads)} routines in Hevy...")

    # --- **** MODIFIED TASK CREATION **** ---
    tasks = []
    # Get the actual tool object (assuming it's globally accessible or passed appropriately)
    # If tool_create_routine is defined globally as shown, this works.
    # If it's part of a class, you'd need access to the instance.
    hevy_tool = tool_create_routine 

    for i, payload in enumerate(payloads):
        routine_title = payload.get("routine_data", {}).get("title", f"Routine {i+1}")
        logger.info(f"Scheduling tool_create_routine call for: '{routine_title}' using ainvoke")
        # Explicitly use the tool's async invocation method
        tasks.append(hevy_tool.ainvoke(input=payload)) 
    # --- **** END OF MODIFIED TASK CREATION **** ---


    # Gather results (same as before)
    tool_outputs = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results (same as before)
    for i, result in enumerate(tool_outputs):
        routine_title = payloads[i].get("routine_data", {}).get("title", f"Routine {i+1}")
        if isinstance(result, Exception):
            # Handle exceptions (same logic)
            logger.error(f"Error during tool ainvoke for '{routine_title}': {result}", exc_info=result)
            error_detail = str(result)
            # Add more specific exception handling if needed
            if isinstance(result, NotImplementedError): # Specifically catch this if it persists
                 error_detail += " (Sync invocation attempted on async tool?)"
            elif hasattr(result, 'detail'):
                 error_detail = f"{getattr(result, 'status_code', 'Unknown Status')}: {result.detail}"
            
            errors.append(f"Tool Execution Failed for '{routine_title}': {error_detail}")
            hevy_results.append({"error": f"Tool Execution Failed: {error_detail}", "routine_title": routine_title, "status": "failed"})
        else:
            # Process successful calls or error dicts (same logic)
            logger.info(f"Tool result for '{routine_title}': {result}")
            hevy_results.append(result)
            if isinstance(result, dict) and result.get("error"):
                 error_msg = f"Hevy API Error for '{routine_title}': {result.get('error')} (Status: {result.get('status_code', 'N/A')})"
                 logger.error(error_msg)
                 errors.append(error_msg)
                 if "status" not in result: result["status"] = "failed"

    # Add summary message (same as before)
    success_count = sum(1 for r in hevy_results if isinstance(r, dict) and "id" in r and not r.get("error"))
    failure_count = len(hevy_results) - success_count
    result_summary = f"Hevy Routine Creation Attempt Results: {success_count} succeeded, {failure_count} failed."
    logger.info(result_summary)

    return {
        **state,
        # "messages": messages + [AIMessage(content=result_summary)],
        "hevy_results": hevy_results,
        "errors": errors
    }


##################################### Progress Analysis and Adaptation Agent ###################################



async def fetch_all_routines_node(state: ProgressAnalysisAdaptationStateV2) -> Dict[str, Any]:
    """Fetches all available routines from Hevy."""
    logger.info("--- Progress Cycle V2: Fetching All Routines ---")
    all_routines = []
    page = 1
    page_size = 20 # Fetch in batches
    try:
        while True:
            logger.debug(f"Fetching routines page {page}...")
            result = await tool_fetch_routines.ainvoke({"page": page, "pageSize": page_size})
            if isinstance(result, dict) and result.get("routines"):
                fetched_page = result["routines"]
                all_routines.extend(fetched_page)
                # Check if this was the last page (Hevy API might indicate this,
                # otherwise, stop if fewer routines than page_size were returned)
                if len(fetched_page) < page_size:
                    break
                page += 1
            elif isinstance(result, dict) and result.get("error"):
                 raise Exception(f"Hevy API error fetching routines: {result.get('error')}")
            else:
                 # Assume empty list or unexpected format means no more routines
                 break
        logger.info(f"Successfully fetched {len(all_routines)} routines.")
        return {"fetched_routines_list": all_routines, "process_error": None}
    except Exception as e:
        logger.error(f"Error fetching routines: {e}", exc_info=True)
        return {"fetched_routines_list": None, "process_error": f"Failed to fetch routines: {e}"}

async def fetch_logs_node(state: ProgressAnalysisAdaptationStateV2) -> Dict[str, Any]:
    """Fetches workout logs from Hevy."""
    if state.get("process_error"): return {} # Skip on prior error
    logger.info("--- Progress Cycle V2: Fetching Logs ---")
    try:
        # Fetch a reasonable number of logs for context
        workout_count_data = await tool_get_workout_count.ainvoke({})
        total_workouts = workout_count_data.get('count', 20)
        page_size = 30 # Increase slightly for better identification context?
        logger.info(f"Fetching latest {page_size} workout logs (total: {total_workouts})...")
        fetched_logs = await tool_fetch_workouts.ainvoke({"page": 1, "pageSize": page_size})
        workout_logs_data = fetched_logs.get('workouts', [])

        if not workout_logs_data:
            logger.warning("No workout logs found.")
            # Don't set process_error, identification might proceed without logs or fail gracefully
            return {"workout_logs": []}
        logger.info(f"Successfully fetched {len(workout_logs_data)} logs.")
        return {"workout_logs": workout_logs_data, "process_error": None}
    except Exception as e:
        logger.error(f"Error fetching workout logs: {e}", exc_info=True)
        # Treat log fetch failure as potentially recoverable, don't halt immediately
        return {"workout_logs": None, "process_error": f"Warning: Failed to fetch logs: {e}"}


async def identify_target_routines_node(state: ProgressAnalysisAdaptationStateV2) -> Dict[str, Any]:
    """Identifies target routines based on logs, request, and available routines."""
    if state.get("process_error") and "Failed to fetch routines" in state["process_error"]:
        # Cannot proceed without routines
        logger.error("Halting identification: Failed to fetch routines list.")
        return {"process_error": state["process_error"]}
    logger.info("--- Progress Cycle V2: Identifying Target Routines ---")

    routines_list = state.get("fetched_routines_list")
    logs = state.get("workout_logs")
    user_model = state.get("user_model")
    user_request = state.get("user_request_context", "")

    if not routines_list:
        logger.warning("No routines available to identify targets from.")
        return {"identified_targets": [], "process_error": state.get("process_error")} # Return empty list, keep potential log error

    # Prepare prompt
    id_prompt_template = get_routine_identification_prompt()
    try:
        # Limit size of JSON passed to LLM if necessary
        routines_json = json.dumps(routines_list[:10], indent=2) # Limit to first 10 routines?
        logs_json = json.dumps(logs[:10] if logs else [], indent=2) # Limit logs context

        filled_prompt = id_prompt_template.format(
            user_profile=json.dumps(user_model.model_dump() if user_model else {}),
            user_request_context=user_request,
            routines_list_json=routines_json,
            logs_list_json=logs_json
        )
    except Exception as e:
         logger.error(f"Error formatting identification prompt: {e}")
         return {"process_error": f"Internal error formatting identification prompt: {e}"}

    # Call LLM
    messages = [SystemMessage(content=filled_prompt)]
    try:
        response = await llm.ainvoke(messages)
        output_str = response.content.strip()

        # Parse the JSON list output
        identified_targets_raw = json.loads(output_str)

        # Validate the structure (basic check)
        if not isinstance(identified_targets_raw, list):
            raise ValueError("LLM did not return a JSON list.")

        # Re-structure slightly to match IdentifiedRoutineTarget TypedDict if needed
        # The prompt asks for {"routine_data": {...}, "reason": "..."} structure
        # This already matches IdentifiedRoutineTarget
        identified_targets: List[IdentifiedRoutineTarget] = identified_targets_raw

        # Further validation can be added here if needed

        logger.info(f"Identified {len(identified_targets)} target routine(s).")
        # Filter targets to ensure 'routine_data' and 'id' exist
        valid_targets = [
            t for t in identified_targets
            if isinstance(t, dict) and isinstance(t.get("routine_data"), dict) and t["routine_data"].get("id")
        ]
        if len(valid_targets) != len(identified_targets):
             logger.warning("Some identified targets had invalid structure and were filtered.")

        return {"identified_targets": valid_targets, "process_error": state.get("process_error")} # Keep potential log error

    except json.JSONDecodeError as e:
         logger.error(f"Error parsing identification JSON from LLM: {e}\nLLM Response:\n{output_str}", exc_info=True)
         return {"process_error": f"Failed to parse valid JSON targets from LLM: {e}"}
    except Exception as e:
        logger.error(f"Error during routine identification LLM call: {e}\nLLM Response:\n{output_str if 'output_str' in locals() else 'N/A'}", exc_info=True)
        return {"process_error": f"Error during routine identification: {e}"}


async def process_targets_node(state: ProgressAnalysisAdaptationStateV2) -> Dict[str, Any]:
    """Iterates through identified targets and attempts adaptation for each."""
    if state.get("process_error"): return {} # Skip if critical error happened before
    logger.info("--- Progress Cycle V2: Processing Identified Targets ---")

    targets = state.get("identified_targets", [])
    if not targets:
        logger.info("No targets to process.")
        return {"processed_results": []} # Return empty list if no targets

    logs = state.get("workout_logs")
    user_model = state.get("user_model")
    processed_results: List[RoutineAdaptationResult] = []
    any_target_failed = False # Flag if any individual adaptation fails

    # --- Get Prompt Templates Once ---
    analysis_prompt_template = get_analysis_v2_template()
    rag_query_prompt_template = get_targeted_rag_query_template()
    modification_prompt_template = get_routine_modification_template_v2()
    reasoning_prompt_template = get_reasoning_generation_template()
    analysis_parser = PydanticOutputParser(pydantic_object=AnalysisFindings)
    analysis_format_instructions = analysis_parser.get_format_instructions()

    # --- Loop Through Each Target ---
    for target in targets:
        routine_data = target["routine_data"]
        routine_id = routine_data["id"]
        original_title = routine_data.get("title", "Unknown Title")
        logger.info(f"--- Processing Target: '{original_title}' (ID: {routine_id}) ---")

        # Initialize result for this target
        current_result = RoutineAdaptationResult(
            routine_id=routine_id,
            original_title=original_title,
            status="Skipped (Error)", # Default to error
            message="Processing started",
            updated_routine_data=None
        )

        try:
            # 1. Analyze Report for Target
            logger.debug(f"Analyzing target '{original_title}'...")
            analysis_findings: Optional[AnalysisFindings] = None
            analysis_error = None
            try:
                analysis_filled_prompt = analysis_prompt_template.format(
                    user_profile=json.dumps(user_model.model_dump() if user_model else {}),
                    target_routine_details=json.dumps(routine_data),
                    workout_logs=json.dumps(logs[:20] if logs else []), # Limit context
                    format_instructions=analysis_format_instructions
                )
                analysis_response = await llm.ainvoke([SystemMessage(content=analysis_filled_prompt)])
                analysis_findings = analysis_parser.parse(analysis_response.content)
                logger.debug(f"Analysis successful for '{original_title}'. Areas found: {len(analysis_findings.areas_for_potential_adjustment)}")
                logger.debug(f"Analysis for {original_title} are {analysis_findings}")
            except Exception as e:
                analysis_error = f"Failed analysis step: {e}"
                logger.error(analysis_error, exc_info=True)
                current_result["message"] = analysis_error
                any_target_failed = True
                processed_results.append(current_result)
                continue # Skip to next target if analysis fails

            # Check if adjustments are needed
            if not analysis_findings or not analysis_findings.areas_for_potential_adjustment:
                logger.info(f"No adjustment areas identified for '{original_title}'. Skipping further steps.")
                current_result["status"] = "Skipped (No Changes)"
                current_result["message"] = "Analysis completed, no modifications needed for this routine."
                processed_results.append(current_result)
                continue # Skip to next target

            # 2. Research Gaps for Target
            logger.debug(f"Researching gaps for '{original_title}'...")
            adaptation_rag_results = {}
            rag_error = None
            try:
                user_profile_json = json.dumps(user_model.model_dump() if user_model else {})
                for area in analysis_findings.areas_for_potential_adjustment:
                    try:
                        query_gen_prompt = rag_query_prompt_template.format(
                            user_profile=user_profile_json, area_for_adjustment=area,
                            previous_query="N/A", previous_result="N/A"
                        )
                        query_response = await llm.ainvoke([SystemMessage(content=query_gen_prompt)])
                        rag_query = query_response.content.strip()
                        logger.debug(f"Querying RAG with generated query for {area}: {query_response}")
                        if rag_query:
                            rag_result = await retrieve_from_rag.ainvoke({"query": rag_query})
                            logger.debug(f"Queried results for RAG with generated query for {area} with {query_response}: {rag_result}")
                            adaptation_rag_results[area] = rag_result or "No specific info."
                        else:
                            adaptation_rag_results[area] = "Could not generate query."
                    except Exception as inner_e:
                         logger.warning(f"Failed RAG for area '{area}': {inner_e}")
                         adaptation_rag_results[area] = f"Error: {inner_e}"
                         rag_error = f"Partial RAG failure on area '{area}'" # Flag partial failure
            except Exception as e:
                 rag_error = f"Failed RAG step: {e}"
                 logger.error(rag_error, exc_info=True)
                 # Continue processing, but note the RAG failure

            if rag_error and not current_result["message"].startswith("Failed"): # Don't overwrite analysis error
                 current_result["message"] = rag_error # Report partial failure


            # 3. Generate Modifications for Target
            logger.debug(f"Generating modifications for '{original_title}'...")
            proposed_mods_dict: Optional[Dict[str, Any]] = None
            modification_reasoning = "N/A"
            mod_output_str = "N/A" # Initialize for error logging
            generation_error = None
            try:
                mod_filled_prompt = modification_prompt_template.format( # Uses updated template
                    user_profile=json.dumps(user_model.model_dump() if user_model else {}),
                    analysis_findings=json.dumps(analysis_findings.model_dump()),
                    adaptation_rag_results=json.dumps(adaptation_rag_results),
                    current_routine_json=json.dumps(routine_data, indent=2)
                )
                mod_response = await llm.ainvoke([SystemMessage(content=mod_filled_prompt)])
                mod_output_str = mod_response.content.strip()
                proposed_mods_dict = json.loads(mod_output_str)
                logger.debug(f"Proposed Modifications are {proposed_mods_dict}")
                if not isinstance(proposed_mods_dict, dict) or 'exercises' not in proposed_mods_dict:
                    raise ValueError("LLM output was not a valid routine JSON dict.")

                # Generate reasoning
                try:
                     reasoning_filled_prompt = reasoning_prompt_template.format(
                         original_routine_snippet=json.dumps(routine_data.get('exercises', [])[:2], indent=2),
                         modified_routine_snippet=json.dumps(proposed_mods_dict.get('exercises', [])[:2], indent=2),
                         analysis_findings=json.dumps(analysis_findings.model_dump()),
                         adaptation_rag_results=json.dumps(adaptation_rag_results)
                     )
                     reasoning_response = await llm.ainvoke([SystemMessage(content=reasoning_filled_prompt)])
                     modification_reasoning = reasoning_response.content.strip()

                     logger.debug(f"Modification reasoning for proposed modifications are : {modification_reasoning}")
                except Exception as reason_e:
                     logger.warning(f"Failed to generate reasoning: {reason_e}")
                     modification_reasoning = "(Failed to generate reasoning)"

                logger.debug(f"Modification generation successful for '{original_title}'.")

            except Exception as e:
                generation_error = f"Failed modification step: {e}"
                logger.error(f"{generation_error}\nLLM Response:\n{mod_output_str if 'mod_output_str' in locals() else 'N/A'}", exc_info=True)
                current_result["message"] = generation_error
                any_target_failed = True
                processed_results.append(current_result)
                continue # Skip to next target

            # --- *** NEW STEP: Validate and Lookup Exercises *** ---
            logger.debug(f"Validating and looking up exercises for '{original_title}'...")
            validated_routine_dict, validation_errors = await validate_and_lookup_exercises(
                proposed_mods_dict,
                original_title
            )

            if validation_errors:
                # Log errors, decide if they are fatal for this routine
                errors_string = "; ".join(validation_errors)
                logger.warning(f"Validation issues found for routine '{original_title}': {errors_string}")
                # If validation returned None, it's a fatal error for this routine
                if validated_routine_dict is None:
                    fatal_error_msg = f"Fatal validation error for '{original_title}', cannot update Hevy. Issues: {errors_string}"
                    logger.error(fatal_error_msg)
                    current_result["message"] = fatal_error_msg
                    current_result["status"] = "Skipped (Error)" # Ensure status reflects failure
                    any_target_failed = True
                    processed_results.append(current_result)
                    continue # Skip Hevy update for this target

            # If validation passed (even with minor warnings), proceed with the validated data
            logger.info(f"Validation successful for '{original_title}'. Proceeding to update Hevy.")

            # --- *** ADD THIS LOGGING *** ---
            logger.debug(f"Payload prepared for Hevy update (routine: '{original_title}', ID: {routine_id}):")
            try:
                # Pretty print the dictionary for readability in DEBUG logs
                logger.debug(json.dumps(validated_routine_dict, indent=2, default=str))
            except Exception as log_e:
                logger.error(f"Failed to serialize validated_routine_dict for logging: {log_e}")
                logger.debug(f"Raw validated_routine_dict: {validated_routine_dict}")
            # --- *** END OF ADDED LOGGING *** ---
            # 4. Update Hevy for Target
            logger.debug(f"Updating Hevy for '{original_title}'...")
            hevy_error = None
            updated_data_from_hevy = None
            try:
                # *** USE THE VALIDATED DICTIONARY ***
                hevy_result = await tool_update_routine.ainvoke({
                    "routine_id": routine_id,
                    "routine_data": validated_routine_dict # Send the validated/corrected dict
                })

                logger.info(f"Hevy API update result (raw type: {type(hevy_result)}): {hevy_result}") # Log type

                # --- FIX V2: Check expected structures ---
                updated_routine_dict = None
                is_success = False

                # Check if result is {'routine': [ {routine_dict} ] }
                if isinstance(hevy_result, dict) and \
                   isinstance(hevy_result.get("routine"), list) and \
                   len(hevy_result["routine"]) > 0 and \
                   isinstance(hevy_result["routine"][0], dict) and \
                   hevy_result["routine"][0].get("id"):

                    updated_routine_dict = hevy_result["routine"][0] # Extract from list
                    is_success = True
                    logger.info(f"Hevy routine '{original_title}' updated successfully (via dict wrapping list).")

                # Check if result is { "routine": {routine_dict} } (Original expectation)
                elif isinstance(hevy_result, dict) and \
                     isinstance(hevy_result.get("routine"), dict) and \
                     hevy_result["routine"].get("id"):

                     updated_routine_dict = hevy_result["routine"] # Directly use the dict
                     is_success = True
                     logger.info(f"Hevy routine '{original_title}' updated successfully (via dict wrapping dict).")

                # Check if result is [ {routine_dict} ] (Previous attempt's expectation)
                elif isinstance(hevy_result, list) and \
                     len(hevy_result) > 0 and \
                     isinstance(hevy_result[0], dict) and \
                     hevy_result[0].get("id"):

                    updated_routine_dict = hevy_result[0] # Extract from list
                    is_success = True
                    logger.info(f"Hevy routine '{original_title}' updated successfully (via direct list).")

                # --- End Fix V2 ---

                if is_success:
                    logger.info(f"Hevy routine '{original_title}' updated successfully.")
                    current_result["status"] = "Success"
                    current_result["message"] = f"Routine '{original_title}' updated successfully."
                    current_result["updated_routine_data"] = hevy_result.get("routine")
                    updated_data_from_hevy = hevy_result.get("routine")
                else:
                    hevy_error_detail = json.dumps(hevy_result.get("error", hevy_result) if isinstance(hevy_result, dict) else hevy_result)
                    hevy_error = f"Hevy update failed for '{original_title}': {hevy_error_detail}"
                    logger.error(hevy_error)
                    current_result["message"] = hevy_error
                    any_target_failed = True

            except Exception as e:
                hevy_error = f"Internal error during Hevy update for '{original_title}': {e}"
                logger.error(hevy_error, exc_info=True)
                current_result["message"] = hevy_error
                current_result["status"] = "Skipped (Error)" # Ensure status reflects failure
                any_target_failed = True

            # Append result for this target
            processed_results.append(current_result)

        except Exception as outer_e:
             # Catch unexpected errors in the loop logic itself
            logger.error(f"Unexpected error processing target '{original_title}': {outer_e}", exc_info=True)
            current_result["message"] = f"Unexpected error: {outer_e}"
            current_result["status"] = "Skipped (Error)" # Ensure status reflects failure
            any_target_failed = True
            processed_results.append(current_result)
            continue # Ensure loop continues

    # --- Return Accumulated Results ---
    final_state_update = {"processed_results": processed_results}
    # If any individual target failed, propagate a general process error message
    # but don't overwrite a more specific earlier error (like routine fetch failure)
    if any_target_failed and not state.get("process_error"):
         final_state_update["process_error"] = "One or more routines failed during the adaptation process."

    logger.info(f"Finished processing all targets. Results count: {len(processed_results)}")
    return final_state_update


async def compile_final_report_node_v2(state: ProgressAnalysisAdaptationStateV2) -> Dict[str, Any]:
    """Compiles the final user-facing report for potentially multiple routines."""
    logger.info("--- Progress Cycle V2: Compiling Final Report ---")

    processed_results = state.get("processed_results", [])
    initial_process_error = state.get("process_error") # Error from early stages (fetch, identify)
    user_model = state.get("user_model")

    # Determine overall status
    overall_status = "Failed" # Default
    overall_message = initial_process_error or "An unknown error occurred." # Start with initial error if present
    processed_summary_lines = []
    success_count = 0
    skipped_no_changes_count = 0
    failure_count = 0
    total_processed = len(processed_results)

    if not initial_process_error:
        if not state.get("identified_targets") and not processed_results:
            overall_status = "Completed (No Targets)"
            overall_message = "I looked at your routines and recent logs, but didn't identify any specific routines needing adaptation right now based on the analysis."
            cycle_completed_successfully = True
        elif processed_results:
            for res in processed_results:
                status_icon = "✅" if res["status"] == "Success" else ("⚠️" if res["status"] == "Skipped (No Changes)" else "❌")
                processed_summary_lines.append(f"{status_icon} **{res['original_title']}**: {res['message']}")
                if res["status"] == "Success": success_count += 1
                if res["status"] == "Skipped (No Changes)": skipped_no_changes_count += 1
                if res["status"] == "Failed" or res["status"] == "Skipped (Error)": failure_count += 1

            if failure_count == 0 and success_count > 0:
                overall_status = "Success"
                overall_message = "I've finished analyzing your progress and successfully updated the relevant routines!"
            elif failure_count == 0 and success_count == 0 and skipped_no_changes_count > 0:
                overall_status = "Completed (No Changes)"
                overall_message = "I've analyzed your progress for the relevant routines, and everything looks on track - no changes needed this time!"
            elif success_count > 0 and failure_count > 0:
                overall_status = "Partial Success"
                overall_message = f"I analyzed your routines. I was able to update {success_count} routine(s), but encountered issues with {failure_count}."
            else: # All failed or skipped with errors
                overall_status = "Failed"
                overall_message = "I tried to analyze and adapt your routines, but encountered errors during the process."

            cycle_completed_successfully = failure_count == 0
        else:
             # This case shouldn't be reached if logic is right (targets identified but no results)
             overall_message = "An unexpected state occurred after identifying targets."
             cycle_completed_successfully = False
    else:
         # Initial process error occurred (e.g., fetch routines failed)
         overall_status = "Failed"
         overall_message = f"I couldn't complete the progress review due to an initial error: {initial_process_error}"
         cycle_completed_successfully = False


    # Format summary string
    processed_results_summary = "\n".join(processed_summary_lines) if processed_summary_lines else "No routines were processed."

    # Generate the final user message using LLM
    final_report_prompt_template = get_final_cycle_report_template_v2()
    user_name = user_model.name if user_model else "there"
    final_message = overall_message # Fallback

    try:
         filled_prompt = final_report_prompt_template.format(
             user_name=user_name,
             processed_results_summary=processed_results_summary,
             overall_status=overall_status,
             overall_message=overall_message
         )
         response = await llm.ainvoke([SystemMessage(content=filled_prompt)])
         final_message = response.content.strip()
    except Exception as e:
         logger.error(f"Error generating final report message: {e}", exc_info=True)
         # Use the constructed message as fallback
         final_message = f"Hi {user_name}, {overall_message}\n\nDetails:\n{processed_results_summary}"


    logger.info(f"Final Cycle Report V2: Status={overall_status}, Successful={cycle_completed_successfully}")
    return {
        "final_report_and_notification": final_message,
        "cycle_completed_successfully": cycle_completed_successfully,
        "process_error": initial_process_error if initial_process_error else state.get("process_error") # Persist any error msg
    }
